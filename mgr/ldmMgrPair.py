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
        self.finalErr = 0.01
        self.enhanceKey = spacemap.Slice.enhanceKey
        self.enhanceErr = 0.01
        self.verbose = 100
        self.gpu = gpu
        self.err = spacemap.AffineFinderMultiDice(10)
        self.ldm = None
        self.emode = False
        
    def start(self, show=True, saveGrid=False):
        self.emode = False
        sI, sJ = self.sI, self.sJ
        imgI1 = self.sI.create_img2(self.targetIKey)
        imgJ2 = self.sJ.create_img2(self.initJKey)
        imgI1, imgJ2 = spacemap.img_norm(imgI1, imgJ2)
        self.ldm = spacemap.get_init2D(imgI1, imgJ2, gpu=self.gpu, verbose=self.verbose)
        spacemap.Info("Start LDMPair: %s/%s -> %s/%s" % (sJ.index, self.finalKey, sI.index, self.targetIKey))
        spacemap.lddmm_main(self.ldm, err=self.finalErr)
        spacemap.Info("Finish LDMPair: %s/%s -> %s/%s" % (sJ.index, self.finalKey, sI.index, self.targetIKey))
        points2 = self.ldm.applyThisTransformPoints2D(self.sJ.to_points(self.initJKey))
        points2 = np.array(points2)
        self.sJ.save_value_points(points2, self.finalKey)
        if show:
            spacemap.Slice.show_align(self.sI, self.sJ, self.targetIKey, self.finalKey)
        if saveGrid:
            grid = self.ldm.generateTransFromGrid()
            self.sJ.saveGrid(grid, self.sI.index)
        imgJ3 = sJ.create_img2(self.finalKey)
        self.show_err(imgI1, imgJ2, imgJ3, "")
    
    def start_enhance(self, enhance=2, kernel=-1, show=True):
        self.emode = True
        spacemap.storage_variables()
        xyd = spacemap.XYD
        sI, sJ = self.sI, self.sJ
        k = spacemap.IMGCONF.get("kernel", 0)
        if kernel == -1:
            kernel = int(enhance // 2) - 1
        spacemap.IMGCONF["kernel"] = kernel            
        spacemap.XYD = int(xyd // enhance)
        imgI1 = sI.create_img2(self.targetIKey)
        imgJ2 = sJ.create_img2(self.finalKey)
        imgI1, imgJ2 = spacemap.img_norm(imgI1, imgJ2)
        self.ldm = spacemap.get_init2D(imgI1, imgJ2, gpu=self.gpu, verbose=self.verbose)
        spacemap.Info("Start LDMPair Enhance: %s/%s -> %s/%s xyd=%d k=%d" % (sJ.index, self.finalKey, sI.index, self.targetIKey, spacemap.XYD, kernel))
        spacemap.lddmm_main(self.ldm, err=self.enhanceErr)
        spacemap.Info("Finish LDMPair Enhance: %s/%s -> %s/%s" % (sJ.index, self.finalKey, sI.index, self.targetIKey))
        points2 = self.ldm.applyThisTransformPoints2D(sJ.to_points(self.finalKey))
        points2 = np.array(points2)
        self.sJ.save_value_points(points2, self.enhanceKey)
        if show:
            spacemap.Slice.show_align(sI, sJ, self.targetIKey, self.enhanceKey)
        imgJ3 = sJ.create_img2(self.enhanceKey)
        self.show_err(imgI1, imgJ2, imgJ3, "Enhance")
        spacemap.revert_variables()
        
    def show_err(self, imgI, imgJ2, imgJ3, tag):
        e1 = self.err.computeI(imgI, imgJ2, False)
        e2 = self.err.computeI(imgI, imgJ3, False)
        spacemap.Info("Err LDMPair %s: %.5f->%.5f" % (tag, e1, e2))
        
