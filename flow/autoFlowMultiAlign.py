import spacemap
from spacemap import Slice
import numpy as np
from spacemap.flow import AutoFlowBasic

class AutoFlowMultiAlign(AutoFlowBasic):
    def __init__(self, slices: list[Slice],
                 initJKey=Slice.rawKey,
                 alignMethod=None,
                 gpu=None):
        super().__init__(slices, initJKey, alignMethod, gpu)

    def ldm_fix_align(self, show=False, err=0.1):
        spacemap.Info("LDMMgrMulti: Start LDM Fix")
        mgr = spacemap.registration.LDDMMRegistration()
        mgr.gpu = self.gpu
        mgr.err = err
        initI = self.slices[0].index

        for i in range(len(self.slices) - 1):
            sI = self.slices[i]
            sJ = self.slices[i+1]
            imgI = sI.create_img(self.finalKey, he=self.heImg, useDF=self.dfMode)
            imgJ = sJ.create_img(self.alignKey, he=self.heImg, useDF=self.dfMode)
            mgr.load_img(imgI, imgJ)
            mgr.run()
            grid = mgr.generate_img_grid()
            self._apply_grid(sJ, self.alignKey, self.finalKey, grid)
            sJ.data.saveGrid(grid, initI, self.finalKey)
            if show:
                self.show_align(sI, sJ, self.finalKey, self.finalKey)
        spacemap.Info("LDMMgrMulti: Finish final Fix")

