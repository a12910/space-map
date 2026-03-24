from __future__ import annotations
import space_map
from space_map import Slice, SliceImg
import numpy as np
from .afFlow2Multi import AutoFlowMultiCenter2

class AutoFlowMultiCenter3(AutoFlowMultiCenter2):
    def __init__(self, slices: list[Slice],
                 initJKey=Slice.rawKey,
                 alignMethod=None,
                 gpu=None):
        super().__init__(slices, initJKey, alignMethod, gpu)
        space_map.Info("AutoFlowMultiCenter3: Init")
        self.rawXYD = space_map.XYD
        self.XYDs = [self.rawXYD*2, 
                     int(self.rawXYD), 
                     int(self.rawXYD/2)]
        self.xydSteps = 3

    def ldm_pair(self,
                 fromKey, toKey,
                 finalErr=0.01,
                 show=False):
        """ customFunc: ([Slice], index, dfKey) -> img """
        def _ldm_pair(indexI, indexJ, slices, err, show=False, ldm=None):
            sI = slices[indexI]
            sJ = slices[indexJ]
            useKey = SliceImg.DF
            imgI1 = sI.create_img(useKey, toKey,
                                  mchannel=False, scale=True, fixHe=True)
            imgJ2 = sJ.create_img(useKey, toKey,
                                  mchannel=False, scale=True, fixHe=True)
            ldm = space_map.registration.LDDMMRegistration()
            ldm.gpu = space_map.DEVICE
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
            ps2, _ = space_map.points.apply_points_by_grid(grid, ps, grid)
            ps2 = space_map.points.fix_points(imgI1, ps2)
            df.save_points(ps2, toKey)
            if not show:
                return
            self.show_align(sI, sJ, useKey, toKey, toKey)

        space_map.Info("LDMMgrMulti: Start LDM Pair")

        xys = [s.ps(fromKey) for s in self.slices]
        xys, xyrange = space_map.utils.grid_points.fix_center_points(xys)
        space_map.XYRANGE = xyrange
        for i, s in enumerate(self.slices):
            s.save_value_points(xys[i], toKey)
        
        for i in range(len(self.slices1) - 1):
            for ste in range(self.xydSteps):
                space_map.XYD = self.XYDs[ste]
                _ldm_pair(i, i+1, self.slices1, 0.1**ste, False)
            if show:
                self.show_align(self.slices1[i], self.slices1[i+1], 
                            SliceImg.DF, toKey, toKey)
        for i in range(len(self.slices2) - 1):
            for ste in range(self.xydSteps):
                space_map.XYD = self.XYDs[ste]
                _ldm_pair(i, i+1, self.slices2, 0.1**ste, False)
            if show:
                self.show_align(self.slices2[i], self.slices2[i+1], 
                            SliceImg.DF, toKey, toKey)
        space_map.XYD = self.rawXYD
        space_map.Info("LDMMgrMulti: Finish LDM Pair")
