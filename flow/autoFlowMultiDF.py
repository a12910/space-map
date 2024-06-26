import spacemap
from spacemap import Slice
import numpy as np
from spacemap.flow import AutoFlowBasic

class AutoFlowMultiDF(AutoFlowBasic):
    def __init__(self, slices: list[Slice],
                 initJKey=Slice.rawKey,
                 alignMethod=None,
                 gpu=None):
        super().__init__(slices, initJKey, alignMethod, gpu)
        self.dfMode = True
        
    def _apply_grid(self, S: Slice, fromKey, toKey, grid):
        ps = S.to_points(fromKey)
        ps2, _ = spacemap.points.apply_points_by_grid(grid, ps, grid[0])
        S.save_value_points(ps2, toKey)

    def ldm_fix_align(self, show=False, err=0.1):
        spacemap.Info("LDMMgrMulti: Start LDM Fix")
        mgr = spacemap.registration.LDDMMRegistration()
        mgr.gpu = self.gpu
        mgr.err = err
        initI = self.slices[0].index

        for i in range(len(self.slices) - 1):
            spacemap.Info("LDMMgrMulti: Start LDM Fix %d/%d" % (i+1, len(self.slices)))
            sI = self.slices[i]
            sJ = self.slices[i+1]
            imgI = sI.create_img(self.finalKey, he=self.heImg, useDF=self.dfMode)
            imgJ = sJ.create_img(self.alignKey, he=self.heImg, useDF=self.dfMode)
            mgr.load_img(imgJ, imgI)
            mgr.run()
            grid = mgr.generate_img_grid()
            self._apply_grid(sJ, self.alignKey, self.finalKey, grid)
            sJ.data.saveGrid(grid, initI, self.finalKey)
            if show:
                self.show_align(sI, sJ, self.finalKey, self.finalKey)
        spacemap.Info("LDMMgrMulti: Finish final Fix")


    def ldm_pair(self, show=False):
        spacemap.Info("LDMMgrMulti: Start LDM Pair")
        for i in range(len(self.slices)-1):
            sI = self.slices[i]
            sJ = self.slices[i+1]
            spacemap.Info("LDMMgrMulti: Start LDM %d/%d %s->%s" % (i+1, len(self.slices), sJ.index, sI.index))
            imgI1 = sI.create_img(self.alignKey, he=self.heImg, useDF=self.dfMode)
            imgJ2 = sJ.create_img(self.alignKey, he=self.heImg, useDF=self.dfMode)
            
            ldm = spacemap.registration.LDDMMRegistration()
            ldm.gpu = self.gpu
            ldm.err = self.finalErr
            ldm.load_img(imgJ2, imgI1)
            ldm.run()
            spacemap.Info("LDMMgrMulti: Finish LDM %d/%d %s->%s" % (i+1, len(self.slices), sJ.index, sI.index))
            grid = ldm.generate_img_grid()
            sJ.data.saveGrid(grid, sI.index, self.pairGridKey)
            imgI3 = ldm.apply_img(imgI1)
            self.show_err(imgJ2, imgI1, imgI3, sJ.index)
            if show:
                self._apply_grid(sJ, self.alignKey, self.ldmKey, grid)
                self.show_align(sI, sJ, self.alignKey, self.ldmKey)
        spacemap.Info("LDMMgrMulti: Finish LDM Pair")
    

    def ldm_merge(self, show=False):
        spacemap.Info("LDMMgrMulti: Start LDM Merge")
        key = self.pairGridKey
        INIT_S = self.slices[0]
        for i in range(len(self.slices) - 1):
            sI = self.slices[i]
            sJ = self.slices[i+1]
            newG = sJ.data.loadGrid(sI.index, key)
            if i > 0:
                lastG = sI.data.loadGrid(INIT_S.index, key)
                newG = spacemap.mergeImgGrid(lastG, newG)
                sJ.data.saveGrid(newG, INIT_S.index, key)
            self._apply_grid(sJ, self.alignKey, self.ldmKey, newG)
            if show:
                self.show_align(sI, sJ, self.ldmKey, self.ldmKey)
        spacemap.Info("LDMMgrMulti: Finish LDM Merge")
        
    def ldm_fix(self, show=False, err=0.1):
        spacemap.Info("LDMMgrMulti: Start LDM Fix")
        mgr = spacemap.registration.LDDMMRegistration()
        mgr.gpu = self.gpu
        mgr.err = err
        initI = self.slices[0].index
        
        for i in range(len(self.slices) - 1):
            sI = self.slices[i]
            sJ = self.slices[i+1]
            imgI = sI.create_img(self.finalKey, he=self.heImg, useDF=self.dfMode)
            imgJ = sJ.create_img(self.ldmKey, he=self.heImg, useDF=self.dfMode)
            mgr.load_img(imgJ, imgI)
            mgr.run()
            grid = mgr.generate_img_grid()
            self._apply_grid(sJ, self.ldmKey, self.finalKey, grid)
            sJ.data.saveGrid(grid, initI, self.fixGridKey)
            if show:
                self.show_align(sI, sJ, self.finalKey, self.finalKey)
        spacemap.Info("LDMMgrMulti: Finish LDM Fix")
        
    def ldm_final(self):
        spacemap.Info("LDMMgrMulti: Start LDM Final")
        initI = self.slices[0].index        
        for i in range(len(self.slices) - 1):
            sI = self.slices[i]
            sJ = self.slices[i+1]
            grid = sJ.data.loadGrid(initI, self.fixGridKey)
            grid1 = sJ.data.loadGrid(initI, self.pairGridKey)
            gridi = spacemap.mergeImgGrid(grid1, grid)
            sJ.data.saveGrid(gridi, initI, self.finalGridKey)
        spacemap.Info("LDMMgrMulti: Finish LDM Final")
       