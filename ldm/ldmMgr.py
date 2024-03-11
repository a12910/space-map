import spacemap
import numpy as np

class LDMMgrPair:
    def __init__(self, 
                 sliceI: spacemap.Slice, 
                 sliceJ: spacemap.Slice, 
                 initJKey = spacemap.Slice.align1Key, 
                 targetIKey=spacemap.Slice.enhanceKey,
                 gpu=None):
        self.sI = sliceI
        self.sJ = sliceJ
        self.initJKey = initJKey
        self.targetIKey = targetIKey
        self.finalKey = spacemap.Slice.finalKey
        self.finalErr = 1
        self.enhanceKey = spacemap.Slice.enhanceKey
        self.enhanceErr = 0.1
        self.verbose = 100
        self.gpu = gpu
        
    def start(self, show=True, saveGrid=False):
        sI, sJ = self.sI, self.sJ
        imgI1 = self.sI.create_img2(self.targetIKey)
        imgJ2 = self.sJ.create_img2(self.initJKey)
        ldm = spacemap.get_init2D(imgI1, imgJ2, gpu=self.gpu, verbose=self.verbose)
        spacemap.Info("Start LDMPair: %s/%s -> %s/%s" % (sJ.index, self.finalKey, sI.index, self.targetIKey))
        spacemap.lddmm_main(ldm, err=self.finalErr)
        spacemap.Info("Finish LDMPair: %s/%s -> %s/%s" % (sJ.index, self.finalKey, sI.index, self.targetIKey))
        points2 = ldm.applyThisTransformPoints2D(self.sJ.to_points(self.initJKey))
        points2 = np.array(points2)
        points2 -= spacemap.XYD // 2
        self.sJ.save_value_points(points2, self.finalKey)
        if show:
            spacemap.Slice.show_align(self.sI, self.sJ, self.targetIKey, self.finalKey)
        if saveGrid:
            grid = ldm.generateTransFromGrid()
            self.sJ.saveGrid(grid, self.sI.index)
    
    def start_enhance(self, enhance=2, show=True):
        xyd = spacemap.XYD
        sI, sJ = self.sI, self.sJ
        spacemap.XYD = int(xyd * enhance)
        imgI1 = sI.create_img2(self.targetIKey)
        imgJ2 = sJ.create_img2(self.finalKey)
        ldm = spacemap.get_init2D(imgI1, imgJ2, gpu=self.gpu, verbose=self.verbose)
        spacemap.Info("Start LDMPair Enhance: %s/%s -> %s/%s xyd=%d" % (sJ.index, self.finalKey, sI.index, self.targetIKey, xyd))
        spacemap.lddmm_main(ldm, err=self.enhanceErr)
        spacemap.Info("Finish LDMPair Enhance: %s/%s -> %s/%s" % (sJ.index, self.finalKey, sI.index, self.targetIKey))
        points2 = ldm.applyThisTransformPoints2D(sJ.to_points(self.finalKey))
        points2 = np.array(points2)
        points2 -= spacemap.XYD // 2
        self.sJ.save_value_points(points2, self.enhanceKey)
        if show:
            spacemap.Slice.show_align(sI, sJ, self.targetIKey, self.enhanceKey)
        spacemap.XYD = xyd
        
