import spacemap
from spacemap import Slice
import numpy as np
from spacemap.flow import AutoFlowBasic


class AutoFlowMultiCenter(AutoFlowBasic):
    def __init__(self, slices: list[Slice], 
                 initJKey=Slice.rawKey,
                 alignMethod=None,
                 gpu=None):
        super().__init__(slices, initJKey, alignMethod, gpu)
        self.cIndex = len(slices) // 2
        self.slices1 = self.slices[self.cIndex:]
        self.slices2 = self.slices[:self.cIndex+1]
        self.slices2.reverse()
        self.centerSlice = self.slices[self.cIndex]
        self.historyGain = 0.75
        
    def _apply_grid(self, S: Slice, fromKey, toKey, grid):
        imgJ2_ = S.create_img(fromKey, mchannel=True, useDF=self.dfMode)
        imgJ3_ = spacemap.img.apply_img_by_grid(imgJ2_, grid)
        meanJ2 = np.mean(imgJ2_)
        meanJ3 = np.mean(imgJ3_)
        imgJ3_ = imgJ3_ * meanJ2 / meanJ3
        
        S.save_value_img(imgJ3_, toKey)
    
    def ldm_pair(self, show=False):
        def _ldm_pair(indexI, indexJ, slices, show=False):
            sI = slices[indexI]
            sJ = slices[indexJ]
            imgI1 = self._create_imgI(indexI, slices, self.alignKey)
            imgJ2 = sJ.create_img(self.alignKey, he=self.heImg, useDF=self.dfMode)
            ldm = spacemap.registration.LDDMMRegistration()
            ldm.gpu = self.gpu
            ldm.err = self.finalErr
            ldm.load_img(imgI1, imgJ2)
            ldm.run()
            grid = ldm.generate_img_grid()
            sJ.data.saveGrid(grid, sI.index, self.pairGridKey)
            imgJ3 = ldm.apply_img(imgJ2)
            self.show_err(imgI1, imgJ2, imgJ3, sJ.index)
            if show:
                self._apply_grid(sJ, self.alignKey, self.ldmKey, grid)
                self.show_align(sI, sJ, self.alignKey, self.ldmKey)
                
        spacemap.Info("LDMMgrMulti: Start LDM Pair")
        for i in range(len(self.slices1) - 1):
            _ldm_pair(i, i+1, self.slices1, show)
        for i in range(len(self.slices2) - 1):
            _ldm_pair(i, i+1, self.slices2, show)
        spacemap.Info("LDMMgrMulti: Finish LDM Pair")
            
    def _create_imgI(self, indexI, slices, key, gg=None):
        if gg is None:
            gg = self.historyGain
        if gg == 0:
            return slices[indexI].create_img(key, he=self.heImg, useDF=self.dfMode)
        img0 = slices[indexI].create_img(key, he=self.heImg, useDF=self.dfMode)
        img0 = img0.astype(np.float32)
        gain = 1
        for i in range(indexI):
            g = gg ** (indexI - i - 1)
            img1 = slices[i].create_img(key, he=self.heImg, useDF=self.dfMode)
            img0 += img1 * g
            gain += g
        img0 /= gain
        return img0
    
    def ldm_merge(self, show=False):
        spacemap.Info("LDMMgrMulti: Start LDM Merge")
        key = self.pairGridKey
        def _ldm_merge(indexI, indexJ, slices):
            sI = slices[indexI]
            sJ = slices[indexJ]
            initI = slices[0].index
            grid = sJ.data.loadGrid(sI.index, key)
            if indexI > 0:
                lastGrid = sI.data.loadGrid(initI, key)
                # grid = add_grid_train(grid, lastGrid)
                grid = spacemap.mergeImgGrid(grid, lastGrid)
                sJ.data.saveGrid(grid, initI, key)
            self._apply_grid(sJ, self.alignKey, self.ldmKey, grid)
            if show:
                self.show_align(sI, sJ, self.ldmKey, self.ldmKey)
        for i in range(len(self.slices1) - 1):
            _ldm_merge(i, i+1, self.slices1)
        for i in range(len(self.slices2) - 1):
            _ldm_merge(i, i+1, self.slices2)
        spacemap.Info("LDMMgrMulti: Finish LDM Merge")
    
    def ldm_merge2(self, show=False):
        spacemap.Info("LDMMgrMulti: Start LDM Merge")
        key = self.pairGridKey
        grids1 = []
        grids2 = []
        def _ldm_merge(indexI, indexJ, slices):
            sJ0 = slices[indexJ]
            sI0 = slices[indexI]
            img = sJ0.create_img(self.alignKey, he=self.heImg, useDF=self.dfMode)

            initI = slices[0].index
            for index in range(indexJ-1, -1, -1):
                sI = slices[index]
                sJ = slices[index+1]
                grid = sJ.data.loadGrid(sI.index, key)
                img = spacemap.img.apply_img_by_grid(img, grid)
            sJ0.save_value_img(img, self.ldmKey)
            if show:
                self.show_align(sI0, sJ0, self.ldmKey, self.ldmKey)
                
        for i in range(len(self.slices1) - 1):
            _ldm_merge(i, i+1, self.slices1)
        for i in range(len(self.slices2) - 1):
            _ldm_merge(i, i+1, self.slices2)
        spacemap.Info("LDMMgrMulti: Finish LDM Merge")
        
    def ldm_fix(self, show=False, err=0.1):
        spacemap.Info("LDMMgrMulti: Start LDM Fix")
        def _ldm_fix(indexI, indexJ, slices, mgr):
            sI = slices[indexI]
            sJ = slices[indexJ]
            imgI = self._create_imgI(indexI, slices, self.finalKey, 0.0)
            imgJ = sJ.create_img(self.ldmKey, he=self.heImg, useDF=self.dfMode)
            mgr.load_img(imgI, imgJ)
            mgr.run()

            imgI = self._create_imgI(indexI, slices, self.finalKey)
            mgr.load_img(imgI, imgJ)
            mgr.run()

            grid = mgr.generate_img_grid()
            self._apply_grid(sJ, self.ldmKey, self.finalKey, grid)
            sJ.data.saveGrid(grid, self.centerSlice.index, self.fixGridKey)
            if show:
                self.show_align(sI, sJ, self.finalKey, self.finalKey)
        mgr = spacemap.registration.LDDMMRegistration()
        mgr.gpu = self.gpu
        mgr.err = err
        for i in range(len(self.slices1) - 1):
            _ldm_fix(i, i+1, self.slices1, mgr)
            
        mgr = spacemap.registration.LDDMMRegistration()
        mgr.gpu = self.gpu
        mgr.err = err
        for i in range(len(self.slices2) - 1):
            _ldm_fix(i, i+1, self.slices2, mgr)
            
        spacemap.Info("LDMMgrMulti: Finish LDM Fix")
        
    def ldm_final(self, show=False):
        spacemap.Info("LDMMgrMulti: Start LDM Final")
        initI = self.centerSlice.index
        def _ldm_final(indexI, indexJ, slices):
            sI = slices[indexI]
            sJ = slices[indexJ]
            grid = sJ.data.loadGrid(initI, self.fixGridKey)
            grid1 = sJ.data.loadGrid(initI, self.pairGridKey)
            gridi = spacemap.mergeImgGrid(grid1, grid)
            sJ.data.saveGrid(gridi, initI, self.finalGridKey)
            self._apply_grid(sJ, self.alignKey, self.finalKey, gridi)
            if show:
                self.show_align(sI, sJ, self.finalKey, self.finalKey)
        
        for i in range(len(self.slices1) - 1):
            _ldm_final(i, i+1, self.slices1)
        for i in range(len(self.slices2) - 1):
            _ldm_final(i, i+1, self.slices2)
        spacemap.Info("LDMMgrMulti: Finish LDM Final")

    def ldm_fix2(self, show=False, err=0.1):
        spacemap.Info("LDMMgrMulti: Start LDM Fix")
        def _ldm_fix(indexI, indexJ, slices, mgr):
            sI = slices[indexI]
            sJ = slices[indexJ]
            imgI = self._create_imgI(indexI, slices, self.finalKey, 0.0)
            imgJ = sJ.create_img(self.ldmKey, he=self.heImg, useDF=self.dfMode)
            mgr.load_img(imgI, imgJ)
            mgr.run()
            
            imgI = self._create_imgI(indexI, slices, self.finalKey)
            mgr.load_img(imgI, imgJ)
            mgr.run()
            grid = mgr.generate_img_grid()
            self._apply_grid(sJ, self.alignKey, self.finalKey, grid)
            sJ.data.saveGrid(grid, self.centerSlice.index, self.fixGridKey)
            if show:
                self.show_align(sI, sJ, self.finalKey, self.finalKey)
        centerSlice = self.centerSlice
        
        img = centerSlice.create_img(Slice.alignKey)
        centerSlice.save_value_img(img, self.finalKey)

        mgr = spacemap.registration.LDDMMRegistration()
        mgr.gpu = self.gpu
        mgr.err = err
        for i in range(len(self.slices1) - 1):
            _ldm_fix(i, i+1, self.slices1, mgr)
            
        mgr = spacemap.registration.LDDMMRegistration()
        mgr.gpu = self.gpu
        mgr.err = err
        for i in range(len(self.slices2) - 1):
            _ldm_fix(i, i+1, self.slices2, mgr)
        