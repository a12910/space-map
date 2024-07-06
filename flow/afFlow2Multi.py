import spacemap
from spacemap import Slice2, SliceImg
import numpy as np
from .afFlow2Basic import AutoFlowBasic2


class AutoFlowMultiCenter2(AutoFlowBasic2):
    def __init__(self, slices: list[Slice2], 
                 initJKey=Slice2.rawKey,
                 alignMethod=None,
                 gpu=None):
        super().__init__(slices, initJKey, alignMethod, gpu)
        self.cIndex = len(slices) // 2
        self.slices1 = self.slices[self.cIndex:]
        self.slices2 = self.slices[:self.cIndex+1]
        self.slices2.reverse()
        self.centerSlice = self.slices[self.cIndex]
        self.historyGain = 0.0
        
    def _create_imgI(self, indexI, slices, useKey, dfkey, gainStep=None):
        if gainStep is None:
            gainStep = self.historyGain
        
        def get_img(slice):
            return slice.get_img(useKey, dfkey, 
                                 mchannel=False, scale=True, fixHe=True)
        
        img0 = get_img(slices[indexI]) 
        if gainStep == 0:
            return img0
        allGain = 1
        img0 = img0.astype(np.float32)
        for i in range(indexI):
            g = gainStep ** (indexI - i - 1)
            img1 = get_img(slices[i])
            img0 += img1 * g
            allGain += g
        img0 /= allGain
        return img0
    
    def ldm_pair(self, useKey, 
                 fromKey=None, toKey=None, saveGridKey=None,
                 fixTrain=False, finalErr=None,
                 show=False, customImgFunc=None):
        """ customFunc: ([Slice2], index, dfKey) -> img """
        if fromKey is None:
            fromKey = self.alignKey
        if toKey is None:
            toKey = self.ldmKey
        if saveGridKey is None:
            saveGridKey = self.pairGridKey
        if finalErr is None:
            finalErr = self.finalErr
            
        def _ldm_pair(indexI, indexJ, slices, show=False, ldm=None):
            sI = slices[indexI]
            sJ = slices[indexJ]
            fromKeyI = fromKey
            if ldm is not None:
                fromKeyI = toKey
            if customImgFunc is not None:
                imgI1 = customImgFunc(slices, indexI, fromKeyI)
                imgJ2 = customImgFunc(slices, indexJ, toKey)
            else:
                imgI1 = sI.create_img(useKey, fromKeyI, 
                                      mchannel=False, scale=True, fixHe=True)
                imgJ2 = sJ.create_img(useKey, fromKey, 
                                      mchannel=False, scale=True, fixHe=True)
            if ldm is None:
                ldm = spacemap.registration.LDDMMRegistration()
                ldm.gpu = self.gpu
                ldm.err = finalErr
            N = imgI1.shape[1]
            grid0 = np.zeros((N, N, 4), dtype=np.float32)
            if useKey == SliceImg.DF:
                ldm.load_img(imgJ2, imgI1)
                ldm.run()
                grid = ldm.generate_img_grid()
                imgI2 = ldm.apply_img(imgI1)
                self.show_err(imgJ2, imgI1, imgI2, sJ.index)
                grid = grid.reshape((N, N, 2))
                grid0[:, :, 2:] = grid
            else:
                ldm.load_img(imgI1, imgJ2)
                ldm.run()
                grid = ldm.generate_img_grid()
                imgJ3 = ldm.apply_img(imgJ2)
                self.show_err(imgI1, imgJ2, imgJ3, sJ.index)
                grid = grid.reshape((N, N, 2))
                grid0[:, :, :2] = grid
            sJ.data.saveGrid(grid0, sI.index, saveGridKey)
            if not show:
                return
            sJ.apply_grid(fromKey, toKey, grid0)
            self.show_align(sI, sJ, useKey, fromKeyI, toKey)
                
        spacemap.Info("LDMMgrMulti: Start LDM Pair")
        self.centerSlice.applyH(fromKey, None, toKey)
        if fixTrain:
            ldm = spacemap.registration.LDDMMRegistration()
            ldm.gpu = self.gpu
            ldm.err = finalErr
        else:
            ldm = None
        for i in range(len(self.slices1) - 1):
            _ldm_pair(i, i+1, self.slices1, show, ldm)
        for i in range(len(self.slices2) - 1):
            _ldm_pair(i, i+1, self.slices2, show, ldm)
        spacemap.Info("LDMMgrMulti: Finish LDM Pair")
    
    def ldm_merge_pair(self, useKey, 
                  fromKey=None, toKey=None, 
                  pairGridKey=None, 
                  show=False):
        spacemap.Info("LDMMgrMulti: Start LDM Merge Pair")
        if fromKey is None:
            fromKey = self.alignKey
        if toKey is None:
            toKey = self.ldmKey
        if pairGridKey is None:
            pairGridKey = self.pairGridKey
            
        self.centerSlice.applyH(fromKey, None, toKey)
        
        def _ldm_merge_pair(indexI, indexJ, slices):
            sI = slices[indexI]
            sJ = slices[indexJ]
            initI = slices[0].index
            grid = sJ.data.loadGrid(sI.index, pairGridKey)
            if indexI > 0:
                lastGrid = sI.data.loadGrid(initI, pairGridKey)
                grid = self._merge_grid(grid, lastGrid)
            self._apply_grid(sJ, fromKey, toKey, grid)
            sJ.data.saveGrid(grid, initI, pairGridKey)
            if show:
                self.show_align(sI, sJ, useKey, toKey, toKey)
        for i in range(len(self.slices1) - 1):
            _ldm_merge_pair(i, i+1, self.slices1)
        for i in range(len(self.slices2) - 1):
            _ldm_merge_pair(i, i+1, self.slices2)
        spacemap.Info("LDMMgrMulti: Finish LDM Merge")
        
    def ldm_merge(self, useKey, fromKey, toKey, dfKey1, dfKey2, dfKeyOut):
        spacemap.Info("LDMMgrMulti: Start LDM Merge Pair")
        self.centerSlice.applyH(fromKey, None, toKey)
        for slice in self.slices1 + self.slices2:
            if slice.index == self.centerSlice.index:
                continue
            initI = self.centerSlice.index
            grid1 = slice.data.loadGrid(initI, dfKey1)
            grid2 = slice.data.loadGrid(initI, dfKey2)
            gridOut = self._merge_grid(grid1, grid2)
            self._apply_grid(slice, fromKey, toKey, gridOut)
            slice.data.saveGrid(gridOut, initI, dfKeyOut)
        spacemap.Info("LDMMgrMulti: Finish LDM Merge")
        
    
        