import spacemap
from spacemap import Slice
import numpy as np

class AutoFlowBasic2:
    def __init__(self, slices: list[Slice], 
                 initJKey=Slice.rawKey,
                 alignMethod=None,
                 gpu=None):
        self.slices: list[Slice] = slices
        self.initJKey = initJKey
        self.alignKey = Slice.align1Key
        self.affineKey  ="cell"
        self.pairGridKey = "img"
        self.fixGridKey = "fix"
        self.finalGridKey = "final_ldm"
        self.ldmKey = Slice.align2Key
        self.continueStart = 0
        self.gpu=spacemap.DEVICE
        if spacemap.DEVICE == "cpu":
            self.gpu = None
        self.finalKey = Slice.finalKey
        self.enhanceKey = Slice.enhanceKey
        self.err = spacemap.find.default()
        self.verbose = 100
        self.finalErr = 0.1
        self.enhanceErr = 0.01
        self.alignMethod = alignMethod
        
    def show_err(self, imgI, imgJ2, imgJ3, tag):
        e1 = self.err.computeI(imgI, imgJ2, False)
        e2 = self.err.computeI(imgI, imgJ3, False)
        spacemap.Info("Err LDMPairMulti %s: %.5f->%.5f" % (tag, e1, e2))
        
    def try_raw(self, useKey):
        S1 = self.slices[0]
        S2 = self.slices[1]
        spacemap.Info("TryRaw: raw=0")
        spacemap.IMGCONF = {"raw": 0}
        err = spacemap.find.default()
        imgA1 = S1.create_img(useKey, self.initJKey, fixHe=True)
        imgA2 = S2.create_img(useKey, self.initJKey, fixHe=True)
        mgr = spacemap.affine_block.AutoAffineImgKey(imgA1, imgA2, show=False, method=self.alignMethod)
        mgr.run()
        H = mgr.resultH_img()
        imgA3 = spacemap.he_img.rotate_imgH(imgA2, H)
        eA2 = err.err(imgA1, imgA3)
        eA1 = err.err(imgA1, imgA2)
        e1 = eA2 / eA1
                
        spacemap.Info("TryRaw: raw=1")
        spacemap.IMGCONF = {"raw": 1}
        imgB1 = S1.create_img(useKey, self.initJKey, fixHe=True)
        imgB2 = S2.create_img(useKey, self.initJKey, fixHe=True)
        mgr = spacemap.affine_block.AutoAffineImgKey(imgB1, imgB2, show=False, method=self.alignMethod)
        mgr.run()
        H = mgr.resultH_img()
        imgB3 = spacemap.he_img.rotate_imgH(imgB2, H)
        eB1 = err.err(imgB1, imgB2)
        eB2 = err.err(imgB1, imgB3)
        e2 = eB2 / eB1
        spacemap.IMGCONF = {"raw": 0 if e1 > e2 else 1}
        spacemap.Info("TryRaw choose-%d 0:%f 1:%f" % (spacemap.IMGCONF["raw"], e1, e2))
        
    def affine(self, useKey, show=False, affine=True, 
               merge=True, customImgFunc=None):
        """ customFunc: (Slice) -> img"""
        spacemap.Info("LDMMgrMulti: Start Affine Pair&Merge")
        method = self.alignMethod
        if method is None:
            method = "sift_vgg"
        spacemap.affine_block.AutoAffineImgKey.restart()
        initS = self.slices[0]
        key = self.affineKey
        
        for i in range(self.continueStart, len(self.slices) - 1):
            S1 = self.slices[i]
            S2 = self.slices[i+1]
            spacemap.Info("LDMMgrMulti: Start Affine %d/%d %s->%s" % (i+1, len(self.slices), S1.index, S2.index))
            if affine:
                if customImgFunc is not None:
                    img1 = customImgFunc(S1)
                    img2 = customImgFunc(S2)
                else: 
                    img1 = S1.create_img(useKey, self.initJKey, fixHe=True)
                    img2 = S2.create_img(useKey, self.initJKey, fixHe=True)
                lastH = S1.data.loadH(initS.index, key) if i > 0 else np.eye(3)
                mgr = spacemap.affine_block.AutoAffineImgKey(img1, img2, show=show, method=self.alignMethod)
                mgr.each.lastImgs.lastH = lastH
                mgr.run()
                H21 = mgr.resultH_img()
                S2.data.saveH(H21, S1.index, key)
                S2.applyH(self.initJKey, H21, self.alignKey)
            if merge:
                H21 = S2.data.loadH(S1.index, key)
                if i == 0:
                    H1i = np.eye(3)
                    S1.applyH(self.initJKey, H1i, self.alignKey)
                else:
                    H1i = S1.data.loadH(initS.index, key)
                H2i = np.dot(H1i, H21)
                S2.data.saveH(H2i, initS.index, key)
                S2.applyH(self.initJKey, H2i, self.alignKey)
                imgJ_new = S2.create_img(useKey, self.alignKey, fixHe=True)
                mgr.each.lastImgs.add_img(imgJ_new)
            if show:
                if merge:
                    self.show_align(S1, S2, useKey, self.alignKey, self.alignKey)
                else:
                    self.show_align(S1, S2, useKey, self.initJKey, self.alignKey)
        if merge and useKey == "DF":
            # final merge
            self.final_fix(useKey, show)            
        spacemap.Info("LDMMgrMulti: Finish Affine Pair&Merge")
        
    def affine2(self, useKey, show=False):
        """ customFunc: (Slice) -> img"""
        spacemap.Info("LDMMgrMulti: Start Affine2 Pair&Merge")
        method = self.alignMethod
        if method is None:
            method = "sift_vgg"
        spacemap.affine_block.AutoAffineImgKey.restart()
        initS = self.slices[0]
        key = self.affineKey
        
        lastImgJ = None
        for i in range(self.continueStart, len(self.slices) - 1):
            S1 = self.slices[i]
            S2 = self.slices[i+1]
            spacemap.Info("LDMMgrMulti: Start Affine %d/%d %s->%s" % (i+1, len(self.slices), S1.index, S2.index))
            if lastImgJ is None:
                lastH = np.eye(3)
                img1 = S1.create_img(useKey, self.initJKey, fixHe=True)
            else:
                lastH = S1.data.loadH(initS.index, key)
                img1 = lastImgJ
            S2.applyH(self.initJKey, lastH, self.alignKey)
            img2 = S2.create_img(useKey, self.alignKey, fixHe=True)
            mgr = spacemap.affine_block.AutoAffineImgKey(img1, img2, show=show, method=self.alignMethod)
            mgr.run()
            H21 = mgr.resultH_img()
            S2.data.saveH(H21, S1.index, key)
            H2i = np.dot(lastH, H21)
            S2.data.saveH(H2i, initS.index, key)
            S2.applyH(self.initJKey, H2i, self.alignKey)
            imgJ_new = S2.create_img(useKey, self.alignKey, fixHe=True)
            mgr.each.lastImgs.add_img(imgJ_new)
            lastImgJ = imgJ_new
            if show:
                self.show_align(S1, S2, useKey, self.alignKey, self.alignKey)     
        if useKey == "DF":
            # final merge
            self.final_fix(useKey, show)            
        spacemap.Info("LDMMgrMulti: Finish Affine Pair&Merge")
        
    def final_fix(self, useKey, show=False):
        spacemap.Info("LDMMgrMulti: Fix Affine Merge DF")
        dfs = None
        # initS = self.slices[0]
        # Hs = []
        # for s in self.slices[1:]:
        #     H = s.data.loadH(initS.index, self.affineKey)
        #     Hs.append(H)
        # H1 = np.array(Hs)[:, :2, :2].mean(axis=0)
        # H2 = np.linalg.inv(H1)
        # H3 = np.eye(3)
        # H3[:2, :2] = H2
        for s in self.slices:
            df = s.ps(self.alignKey)
            if dfs is None:
                dfs = df
            else:
                dfs = np.concatenate((dfs, df), axis=0)
        mid = dfs.mean(axis=0)
        mid0 = np.array([spacemap.XYRANGE/2, spacemap.XYRANGE/2])
        spacemap.Info("LDMMgrMulti: Fix Center DF %d %d -> %d %d" % (mid[0], mid[1], mid0[0], mid0[1]))
        for s in self.slices:
            df = s.ps(self.alignKey)
            df1 = df - mid
            # df1 = spacemap.points.applyH_np(df1, H3)
            df1 += mid0
            s.save_value_points(df1, self.alignKey)
                
        
    def affine_pair(self, show=False):
        self.affine(affine=True, merge=False, show=show)
            
    def affine_merge(self, show=False):
        self.affine(merge=True, affine=False, show=show)
    
    @staticmethod
    def _is_zero(grid):
        return np.sum(np.abs(grid)) < 0.1
        
    def show_align(self, S1: Slice, S2: Slice, useKey, key1, key2):
        img1 = S1.create_img(useKey, key1, scale=True, fixHe=True)
        img2 = S2.create_img(useKey, key2, scale=True, fixHe=True)
        e = self.err.err(img1, img2)
        spacemap.Info("Show AlignErr %s/%s %s/%s %f" % (S1.index, key1, S2.index, key2, e))
        Slice.show_align(S1, S2, key1, key2, useKey)
        
    def _apply_grid(self, S: Slice, fromKey, toKey, grid):
        grid1 = grid[:, :, :2]
        inv_grid1 = grid[:, :, 2:]
        if self._is_zero(grid1):
            grid1 = None
        if self._is_zero(inv_grid1):
            inv_grid1 = None
        S.apply_grid(fromKey, toKey, grid1, inv_grid1)
        
    def _merge_grid(self, grid, lastGrid):
        N = grid.shape[1]
        if self._is_zero(grid[:, :, :2]):
            grid1 = grid[:, :, 2:]
            lastGrid1 = lastGrid[:, :, 2:]
            grid1 = spacemap.mergeImgGrid(lastGrid1, grid1)
            grid[:, :, 2:] = grid1.reshape((N, N, 2))
        else:
            grid1 = grid[:, :, :2]
            lastGrid1 = lastGrid[:, :, :2]
            grid1 = spacemap.mergeImgGrid(grid1, lastGrid1)
            grid[:, :, :2] = grid1.reshape((N, N, 2))
        return grid
        
    def show(self, useKey, dfKey):
        count = len(self.slices)
        for i in range(count - 1):
            self.show_align(self.slices[i], self.slices[i+1], useKey, dfKey, dfKey)