import spacemap
from spacemap import Slice2, SliceImg
import numpy as np
from .afFlow2Multi import AutoFlowMultiCenter2

class AutoFlowMultiCenter3(AutoFlowMultiCenter2):
    def __init__(self, slices: list[Slice2],
                 initJKey=Slice2.rawKey,
                 alignMethod=None,
                 gpu=None):
        super().__init__(slices, initJKey, alignMethod, gpu)
        self.rawXYD = spacemap.XYD
        self.XYDs = [self.rawXYD*2, 
                     int(self.rawXYD), 
                     int(self.rawXYD/2)]
        self.xydSteps = 3

    def ldm_pair(self,
                 fromKey, toKey,
                 finalErr=0.01,
                 show=False):
        """ customFunc: ([Slice2], index, dfKey) -> img """
        def _ldm_pair(indexI, indexJ, slices, err, show=False, ldm=None):
            sI = slices[indexI]
            sJ = slices[indexJ]
            useKey = SliceImg.DF
            imgI1 = sI.create_img(useKey, toKey,
                                  mchannel=False, scale=True, fixHe=True)
            imgJ2 = sJ.create_img(useKey, toKey,
                                  mchannel=False, scale=True, fixHe=True)
            ldm = spacemap.registration.LDDMMRegistration()
            ldm.gpu = self.gpu
            ldm.err = err
            N = imgI1.shape[1]
            ldm.load_img(imgJ2, imgI1)
            ldm.run()
            grid = ldm.generate_img_grid()
            imgI2 = ldm.apply_img(imgI1)
            self.show_err(imgJ2, imgI1, imgI2, sJ.index)
            grid = grid.reshape((N, N, 2))
            df = sJ.imgs["DF"]
            ps = df.ps(toKey)
            ps2, _ = spacemap.points.apply_points_by_grid(grid, ps, grid)
            df.save_points(ps2, toKey)
            if not show:
                return
            self.show_align(sI, sJ, useKey, toKey, toKey)

        spacemap.Info("LDMMgrMulti: Start LDM Pair")

        for s in self.slices:
            s.applyH(fromKey, None, toKey)
        
        for i in range(len(self.slices1) - 1):
            for ste in range(self.xydSteps):
                spacemap.XYD = self.XYDs[ste]
                _ldm_pair(i, i+1, self.slices1, 0.1**ste, False)
            self.show_align(self.slices1[i], self.slices1[i+1], 
                            SliceImg.DF, toKey, toKey)
        for i in range(len(self.slices2) - 1):
            for ste in range(self.xydSteps):
                spacemap.XYD = self.XYDs[ste]
                _ldm_pair(i, i+1, self.slices2, 0.1**ste, False)
            self.show_align(self.slices2[i], self.slices2[i+1], 
                            SliceImg.DF, toKey, toKey)
        spacemap.XYD = self.rawXYD
        spacemap.Info("LDMMgrMulti: Finish LDM Pair")
