import spacemap
from spacemap import Slice
import numpy as np

class LDMMgrMulti:
    def __init__(self, slices: list[Slice], 
                 initJKey=Slice.rawKey,
                 enhance=None, gpu=None):
        self.slices = slices
        self.initJKey = initJKey
        self.alignKey = Slice.align1Key
        self.enhance=enhance
        self.enhanceKernel = -1
        self.gpu=gpu
        self.finalKey = Slice.finalKey
        self.enhanceKey = Slice.enhanceKey
        self.err = spacemap.find.default()
        self.verbose = 100
        self.finalErr = 0.01
        self.enhanceErr = 0.01
        
    def show_err(self, imgI, imgJ2, imgJ3, tag):
        e1 = self.err.computeI(imgI, imgJ2, False)
        e2 = self.err.computeI(imgI, imgJ3, False)
        spacemap.Info("Err LDMPairMulti %s: %.5f->%.5f" % (tag, e1, e2))

        
    def start_affine(self, show=False):
        spacemap.Info("LDMMgrMulti: Start Affine")
        for i in range(len(self.slices) - 1):
            TARGET_S = self.slices[i]
            NEW_S = self.slices[i+1]
            spacemap.Info("LDMMgrMulti: Start Affine %d/%d %s->%s" % (i+1, len(self.slices), NEW_S.index, TARGET_S.index))
            flow = spacemap.SpaceMapAutoFlow2(TARGET_S.to_points(self.initJKey), NEW_S.to_points(self.initJKey))
            flow.run()
            H = flow.bestH()
            NEW_S.saveH(H, TARGET_S.index)
        
        spacemap.Info("LDMMgrMulti: Start Merge Affine")
        INIT_S = self.slices[0]
        for i in range(0, len(self.slices) - 1):
            TARGET_S = self.slices[i]
            NEW_S = self.slices[i+1]
            newH = NEW_S.loadH(TARGET_S.index)
            if i > 0:
                initH = TARGET_S.loadH(INIT_S.index)
                newH = np.dot(initH, newH)
            NEW_S.saveH(newH, INIT_S.index)
            NEW_S.applyH(self.initJKey, newH, self.alignKey)
            if show:
                Slice.show_align(TARGET_S, NEW_S, self.alignKey, self.alignKey)
        spacemap.Info("LDMMgrMulti: Affine Finished")
            
    def start_ldm(self, show=False):
        spacemap.Info("LDMMgrMulti: Start LDM")
        for i in range(len(self.slices)-1):
            sI = self.slices[i]
            sJ = self.slices[i+1]
            spacemap.Info("LDMMgrMulti: Start LDM %d/%d %s->%s" % (i+1, len(self.slices), sJ.index, sI.index))
            imgI1 = sI.create_img2(self.alignKey)
            imgJ2 = sJ.create_img2(self.alignKey)
            ldm = spacemap.get_init2D(imgI1, imgJ2, gpu=self.gpu, verbose=self.verbose)
            spacemap.lddmm_main(ldm, err=self.finalErr)
            spacemap.Info("LDMMgrMulti: Finish LDM %d/%d %s->%s" % (i+1, len(self.slices), sJ.index, sI.index))
            grid = ldm.generateTransFromGrid()
            sJ.saveGrid(grid, sI.index)
        
        spacemap.Info("LDMMgrMulti: Start LDM Merge")
        INIT_S = self.slices[0]
        for i in range(len(self.slices) - 1):
            sI = self.slices[i]
            sJ = self.slices[i+1]
            newG = sJ.loadGrid(sI.index)
            if i > 0:
                lastG = sI.loadGrid(INIT_S.index)
                newG = spacemap.mergeGrid(newG, lastG)
                sJ.saveGrid(newG, INIT_S.index)
            points = sJ.to_points(self.alignKey)
            points2 = spacemap.applyPointsByGrid(newG, points)
            sJ.save_value_points(points2, self.finalKey)
            if show:
                Slice.show_align(sI, sJ, self.finalKey, self.finalKey)
            imgJ3 = sJ.create_img2(self.finalKey)
            self.show_err(imgI1, imgJ2, imgJ3, "Final")
        
        spacemap.Info("LDMMgrMulti: Finish LDM Merge")
        
    def start_enhance(self, show=False):
        xyd = spacemap.XYD
        if self.enhance is None:
            return
        spacemap.storage_variables()
        spacemap.XYD = int(xyd // self.enhance)
        kernel = self.enhanceKernel
        if kernel == -1:
            kernel = int(self.enhance // 2) - 1
        spacemap.IMGCONF["kernel"] = kernel          
        
        spacemap.Info("LDMMgrMulti: Start LDM Enhance")
        for i in range(len(self.slices) - 1):
            sI = self.slices[i]
            sJ = self.slices[i+1]
            imgI1 = sI.create_img2(self.enhanceKey)
            imgJ2 = sJ.create_img2(self.finalKey)
            ldm = spacemap.get_init2D(imgI1, imgJ2, gpu=self.gpu, verbose=self.verbose)
            spacemap.Info("Start LDMPairMulti Enhance: %s/%s -> %s/%s xyd=%d" % (sJ.index, self.finalKey, sI.index, self.enhanceKey, spacemap.XYD))
            spacemap.lddmm_main(ldm, err=self.enhanceErr)
            spacemap.Info("Finish LDMPairMulti Enhance: %s/%s -> %s/%s" % (sJ.index, self.finalKey, sI.index, self.enhanceKey))
            points2 = ldm.applyThisTransformPoints2D(sJ.to_points(self.finalKey))
            points2 = np.array(points2)
            points2 -= spacemap.XYD // 2
            sJ.save_value_points(points2, self.enhanceKey)
            if show:
                spacemap.Slice.show_align(sI, sJ, self.enhanceKey, self.enhanceKey)
            imgJ3 = sJ.create_img2(self.finalKey)
            self.show_err(imgI1, imgJ2, imgJ3, "Enhance")
            
        spacemap.revert_variables()
        spacemap.Info("LDMMgrMulti: Finish LDM Enhance")
        