import spacemap
from spacemap import Slice, SliceImg
import numpy as np
from .afFlow2Basic import AutoFlowBasic2

def _show_align(S1: Slice, S2: Slice, useKey, key1, key2):
    img1 = S1.create_img(useKey, key1, scale=True, fixHe=True)
    img2 = S2.create_img(useKey, key2, scale=True, fixHe=True)
    err = spacemap.find.default()
    e = err.err(img1, img2)
    spacemap.Info("Show AlignErr %s/%s %s/%s %f" % (S1.index, key1, S2.index, key2, e))
    Slice.show_align(S1, S2, key1, key2, useKey)


def _show_err(imgI, imgJ2, imgJ3, tag):
    err = spacemap.find.default()
    e1 = err.computeI(imgI, imgJ2, False)
    e2 = err.computeI(imgI, imgJ3, False)
    spacemap.Info("Err LDMPairMulti %s: %.5f->%.5f" % (tag, e1, e2))

def _ldm_pair(pack):
    indexI = pack["indexI"]
    indexJ = pack["indexJ"]
    slices = pack["slices"]
    show = pack["show"]
    centerTrain = pack["centerTrain"]
    fromKey = pack["fromKey"]
    toKey = pack["toKey"]
    gpu = pack["gpu"]
    saveGridKey = pack["saveGridKey"]
    finalErr = pack["finalErr"]
    ldm = None
    
    sI = slices[indexI]
    sJ = slices[indexJ]
    fromKeyI = fromKey
    
    if ldm is not None or centerTrain:
        fromKeyI = toKey
    useKey = SliceImg.DF
    imgI1 = sI.create_img(useKey, fromKeyI, 
                            mchannel=False, scale=True, fixHe=True)
    imgJ2 = sJ.create_img(useKey, fromKey, 
                            mchannel=False, scale=True, fixHe=True)
    if ldm is None:
        ldm = spacemap.registration.LDDMMRegistration()
        ldm.gpu = gpu
        ldm.err = finalErr
    N = imgI1.shape[1]
    ldm.load_img(imgJ2, imgI1)
    ldm.run()
    grid = ldm.generate_img_grid()
    imgI2 = ldm.apply_img(imgI1)
    _show_err(imgJ2, imgI1, imgI2, sJ.index)
    grid = grid.reshape((N, N, 2))
    sJ.data.saveGrid(grid, sI.index, saveGridKey)
    if not show:
        return
    sJ.apply_grid(fromKey, toKey, grid, grid)
    _show_align(sI, sJ, useKey, fromKeyI, toKey)

class AutoFlowMultiCenter2DF(AutoFlowBasic2):
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
    
    def ldm_pair(self, saveGridKey=None,
                 fromKey=None, toKey=None,
                 centerTrain=False, finalErr=None,
                 show=False):
        """ customFunc: ([Slice], index, dfKey) -> img """
        if fromKey is None:
            fromKey = self.alignKey
        if toKey is None:
            toKey = self.ldmKey
        if saveGridKey is None:
            saveGridKey = self.pairGridKey
        if finalErr is None:
            finalErr = self.finalErr
                
        spacemap.Info("LDMMgrMulti: Start LDM Pair")
        self.centerSlice.applyH(fromKey, None, toKey)
        packs = []
        for i in range(len(self.slices1) - 1):
            packs.append({
                "indexI": i,
                "indexJ": i+1,
                "slices": self.slices1,
                "show": show,
                "centerTrain": centerTrain,
                "fromKey": fromKey,
                "toKey": toKey,
                "gpu": self.gpu,
                "saveGridKey": saveGridKey,
                "finalErr": finalErr
            })
        for i in range(len(self.slices2) - 1):
            packs.append({
                "indexI": i,
                "indexJ": i+1,
                "slices": self.slices2,
                "show": show,
                "centerTrain": centerTrain,
                "fromKey": fromKey,
                "toKey": toKey,
                "gpu": self.gpu,
                "saveGridKey": saveGridKey,
                "finalErr": finalErr
            })
        if centerTrain:
            for p in packs:
                _ldm_pair(p)
        else:
            from multiprocessing import Pool
            import os
            with Pool(os.cpu_count()) as pool:
                pool.map(_ldm_pair, packs)
            
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
        
    
        