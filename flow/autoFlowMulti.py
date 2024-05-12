import spacemap
from spacemap import Slice
import numpy as np

"""
1. affine_pair: initJKey -> alignKey = H/cell 
    :affine to last 相邻affine
2. affine_merge: initJKey -> alignKey = H/cell 
    :affine to first 合并affine到第一张
3. ldm_pair: alignKey -> ldmKey = grid/img 
    :ldm to last 相邻ldm
4. ldm_merge: alignKey -> ldmKey = grid/img 
    :ldm to first 合并ldm到第一张
5. ldm_fix: ldmKey -> finalKey = grid/fix 
    :ldm to first 修正合并后的ldm
6. ldm_final: alignKey -> finalKey = grid/final_ldm 
    :ldm to first 生成最终grid
"""

class AutoFlowMulti:
    def __init__(self, slices: list[Slice], 
                 initJKey=Slice.rawKey,
                 alignMethod=None,
                 gpu=None):
        self.slices = slices
        self.initJKey = initJKey
        self.alignKey = Slice.align1Key
        self.ldmKey = Slice.align2Key
        self.gpu=gpu
        self.finalKey = Slice.finalKey
        self.enhanceKey = Slice.enhanceKey
        self.err = spacemap.find.default()
        self.verbose = 100
        self.finalErr = 0.1
        self.enhanceErr = 0.01
        self.heImg = False
        self.alignMethod = alignMethod
        
        self.applyDF = False
        self.applyImg = False
        
    def show_err(self, imgI, imgJ2, imgJ3, tag):
        e1 = self.err.computeI(imgI, imgJ2, False)
        e2 = self.err.computeI(imgI, imgJ3, False)
        spacemap.Info("Err LDMPairMulti %s: %.5f->%.5f" % (tag, e1, e2))
        
    def affine(self, show=False, affine=True, merge=True):
        spacemap.Info("LDMMgrMulti: Start Affine Pair&Merge")
        method = self.alignMethod
        if method is None:
            method = "sift_vgg"
            
        initS = self.slices[0]
        key = "cell"
            
        for i in range(len(self.slices) - 1):
            S1 = self.slices[i]
            S2 = self.slices[i+1]
            spacemap.Info("LDMMgrMulti: Start Affine %d/%d %s->%s" % (i+1, len(self.slices), S1.index, S2.index))
            if affine:
                img1 = S1.create_img(self.initJKey, useDF=False, 
                                     he=self.heImg)
                img2 = S2.create_img(self.initJKey, useDF=False, 
                                     he=self.heImg)
                mgr = spacemap.affine_block.AutoAffineImgKey(img1, img2, show=show, method=self.alignMethod)
                mgr.run()
                H21 = mgr.resultH_img()
                S2.data.saveH(H21, S1.index, key)
            if merge:
                H21 = S2.data.loadH(S1.index, key)
                if i == 0:
                    H1i = np.eye(3)
                else:
                    H1i = S1.data.loadH(initS.index, key)
                H2i = np.dot(H1i, H21)
                S2.data.saveH(H2i, initS.index, key)
                S2.applyH(self.initJKey, H2i, self.alignKey, forIMG=True)
                if self.applyDF:
                    H2i_np = spacemap.img.to_npH(H2i)
                    S2.applyH(self.initJKey, H2i_np, self.alignKey, forIMG=False)
            if show:
                self.show_align(S1, S2, self.alignKey, self.alignKey)
            
        spacemap.Info("LDMMgrMulti: Finish Affine Pair&Merge")
        
    def affine_pair(self, show=False):
        self.affine(affine=True, merge=False, show=show)
            
    def affine_merge(self, show=False):
        self.affine(merge=True, affine=False, show=True)
        
    def show_align(self, S1, S2, key1, key2):
        Slice.show_align(S1, S2, key1, key2, forIMG=True, imgHE=self.heImg)
        if self.applyDF:
            Slice.show_align(S1, S2, key1, key2, forIMG=False, imgHE=self.heImg)
            
    def ldm_pair(self, show=False):
        spacemap.Info("LDMMgrMulti: Start LDM Pair")
        for i in range(len(self.slices)-1):
            sI = self.slices[i]
            sJ = self.slices[i+1]
            spacemap.Info("LDMMgrMulti: Start LDM %d/%d %s->%s" % (i+1, len(self.slices), sJ.index, sI.index))
            imgI1 = sI.get_img(self.alignKey, he=self.heImg)
            imgJ2 = sJ.get_img(self.alignKey, he=self.heImg)
            
            ldm = spacemap.registration.LDDMMRegistration()
            ldm.gpu = self.gpu
            ldm.err = self.finalErr
            ldm.load_img(imgI1, imgJ2)
            ldm.run()
            spacemap.Info("LDMMgrMulti: Finish LDM %d/%d %s->%s" % (i+1, len(self.slices), sJ.index, sI.index))
            grid = ldm.generate_img_grid()
            sJ.data.saveGrid(grid, sI.index, "img")
            imgJ3 = ldm.apply_img(imgJ2)
            self.show_err(imgI1, imgJ2, imgJ3, sJ.index)
            if show:
                self._apply_grid(sJ, self.alignKey, self.ldmKey, grid)
                self.show_align(sI, sJ, self.alignKey, self.ldmKey)
        spacemap.Info("LDMMgrMulti: Finish LDM Pair")
        
    def _apply_grid(self, S: Slice, fromKey, toKey, grid):
        imgJ2_ = S.get_img(fromKey, mchannel=True, scale=False)
        imgJ3_ = spacemap.img.apply_img_by_grid(imgJ2_, grid)
        S.save_value_img(imgJ3_, toKey)
    
    def ldm_merge(self, show=False):
        spacemap.Info("LDMMgrMulti: Start LDM Merge")
        key = "img"
        INIT_S = self.slices[0]
        for i in range(len(self.slices) - 1):
            sI = self.slices[i]
            sJ = self.slices[i+1]
            newG = sJ.data.loadGrid(sI.index, key)
            if i > 0:
                lastG = sI.data.loadGrid(INIT_S.index, key)
                newG = spacemap.mergeImgGrid(newG, lastG)
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
            imgI = sI.get_img(self.finalKey, he=self.heImg)
            imgJ = sJ.get_img(self.ldmKey, he=self.heImg)
            mgr.load_img(imgI, imgJ)
            mgr.run()
            grid = mgr.generate_img_grid()
            self._apply_grid(sJ, self.ldmKey, self.finalKey, grid)
            sJ.data.saveGrid(grid, initI, "fix")
            if show:
                self.show_align(sI, sJ, self.finalKey, self.finalKey)
        spacemap.Info("LDMMgrMulti: Finish LDM Fix")
        
    def ldm_final(self):
        spacemap.Info("LDMMgrMulti: Start LDM Final")
        initI = self.slices[0].index        
        for i in range(len(self.slices) - 1):
            sI = self.slices[i]
            sJ = self.slices[i+1]
            grid = sJ.data.loadGrid(initI, "fix")
            grid1 = sJ.data.loadGrid(initI, "img")
            gridi = spacemap.mergeImgGrid(grid1, grid)
            sJ.data.saveGrid(gridi, initI, "final_ldm")
        spacemap.Info("LDMMgrMulti: Finish LDM Final")
        
    def ldm_continue(self, fromKey, toKey, he=False,
                     affineFirst=False, fromGridKey="final_ldm", toGridKey="continue", affineFirstKey=None, show=False, err=0.01):
        spacemap.Info("LDMMgrMulti: Start LDM Continue: %s->%s use: %s->%s" % (fromKey, toKey, fromGridKey, toGridKey))
        
        mgr = spacemap.registration.LDDMMRegistration()
        mgr.gpu = self.gpu
        mgr.err = 0.01
        initI = self.slices[0].index
        slices = self.slices
        
        if affineFirst:
            spacemap.Info("LDMMgrMulti: Start Affine First")
            if affineFirstKey is None:
                affineFirstKey = "affineFirstKey"
            for i in range(len(self.slices) - 1):
                sI = self.slices[i]
                imgI = sI.get_img_raw(fromKey)
                if i == 0:
                    sI.save_value_img(imgI, affineFirstKey)
                else:
                    imgI2 = spacemap.img.apply_transform(sI, imgI, 
                                        affineShape=None, 
                                        initIndex=initI, gridKey=fromGridKey)
                    sI.save_value_img(imgI2, affineFirstKey)
            fromKey = affineFirstKey
        
        for i in range(len(slices) - 1):
            spacemap.Info("LDMMgrMulti: Continue %d/%d" % (i+1, len(slices)))
            sI = self.slices[i]
            sJ = slices[i+1]
            if i == 0:
                imgI = sI.get_img_raw(fromKey)
                sI.save_value_img(imgI, toKey)
            imgI = sI.get_img(toKey, he=he)
            imgJ = sJ.get_img(fromKey, he=he)
            mgr.load_img(imgI, imgJ)
            mgr.run()
            
            grid = mgr.generate_img_grid()
            imgJ2_ = sJ.get_img_raw(fromKey)
            imgJ3_ = spacemap.img.apply_img_by_grid(imgJ2_, grid)
            sJ.save_value_img(imgJ3_, toKey)
            
            grid = mgr.generate_img_grid()
            gridOld = sJ.data.loadGrid(initI, fromGridKey)
            gridNew = spacemap.mergeImgGrid(gridOld, grid)
            sJ.data.saveGrid(gridNew, initI, toGridKey)
            if show:
                self.show_align(sI, sJ, toKey, toKey)
        
        spacemap.Info("LDMMgrMulti: Finish LDM Continue: %s->%s use: %s->%s" % (fromKey, toKey, fromGridKey, toGridKey))
        