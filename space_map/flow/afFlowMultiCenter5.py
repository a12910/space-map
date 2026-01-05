import space_map
from space_map import Slice, SliceImg
import numpy as np
import os
from .afFlowMultiCenter4 import AutoFlowMultiCenter4


class AutoFlowMultiCenter5(AutoFlowMultiCenter4):
    def __init__(self, slices: list[Slice],
                 initJKey=Slice.rawKey,
                 alignMethod=None,
                 gpu=None):
        super().__init__(slices, initJKey, alignMethod, gpu)
        self.affine_skip=True
        self.keep_1 = False

    def _affine_run(self, S1, S2, useKey, i, show, initS, key):
        img1 = S1.create_img(useKey, self.initJKey, fixHe=True)
        img2 = S2.create_img(useKey, self.initJKey, fixHe=True)
        space_map.Info("LDMMgrMulti: Start Affine %d/%d %s->%s" % (i+1, len(self.slices), S1.index, S2.index))
        # lastH = S1.data.loadH(initS.index, key) if i > 0 else np.eye(3)
        df = None
        if useKey == "DF":
            df = (S1.ps(self.initJKey), S2.ps(self.initJKey))
        mgr = space_map.affine_block.AutoAffineImgKey(img1, img2, show=False, method=self.alignMethod)
        # mgr.each.lastImgs.lastH = lastH
        mgr.run(df)
        H21 = mgr.resultH_img()
        if self.keep_1:
            nonzero_indices = np.argwhere(img1 > 0)
            center = nonzero_indices.mean(axis=0)
            H21 = self.fix_single_matrix(H21, center)
        return H21

    def fix_single_matrix(self, H, center_point):
        H_old = H.astype(float)
        cx, cy = center_point
        P_center = np.array([cx, cy])
        A_old = H_old[:2, :2]
        T_old = H_old[:2, 2]
        P_target = A_old @ P_center + T_old
        U, S, Vt = np.linalg.svd(A_old)
        R_new = U @ Vt
        if np.linalg.det(R_new) < 0:
            Vt[1, :] *= -1  # 反转 V 的第二行
            R_new = U @ Vt
        T_new = P_target - (R_new @ P_center)
        H_new = H_old.copy()
        H_new[:2, :2] = R_new
        H_new[:2, 2] = T_new
        if H_new.shape[0] == 3:
            H_new[2, :] = [0, 0, 1]
        return H_new

    def _applyH(self, S, H, key):
        img = S.create_img("DF", key, fixHe=True)
        img2 = SliceImg.applyH_img(img, H)
        # ps = S.ps(key)
        # ps2 = SliceImg.applyH_ps(ps, H)
        # img2 = spacemap.show_img(ps2)
        return img2

    def affine(self, useKey, show=False):
        """ customFunc: (Slice) -> img"""
        space_map.Info("LDMMgrMulti: Start Affine Pair&Merge")
        method = self.alignMethod
        if method is None:
            method = "sift_vgg"
        space_map.affine_block.AutoAffineImgKey.restart()
        initS = self.slices[0]
        key = self.affineKey
        
        for i in range(self.continueStart + 1, len(self.slices)):
            S1 = self.slices[i-1]
            S2 = self.slices[i]
            H21 = self._affine_run(S1, S2, useKey, i, show, initS, key)          
            if i == 1:
                H1i = np.eye(3)
                S1.applyH(self.initJKey, H1i, self.alignKey)
                H2i = np.dot(H1i, H21)
                S2.data.saveH(H2i, initS.index, key)
                S2.applyH(self.initJKey, H2i, self.alignKey)
                if show:
                    self.show_align(S1, S2, useKey, self.initJKey, self.alignKey)  
            elif not self.affine_skip:
                H1i = S1.data.loadH(initS.index, key)
                H2i = np.dot(H1i, H21)
                S2.data.saveH(H2i, initS.index, key)
                S2.applyH(self.initJKey, H2i, self.alignKey)
                if show:
                    self.show_align(S1, S2, useKey, self.initJKey, self.alignKey)  
            else:
                err = space_map.find.default()
                S0 = self.slices[i-2]
                H20 = self._affine_run(S0, S2, useKey, i, show, initS, key)
                imgS1 = S1.create_img(useKey, self.initJKey, fixHe=True)
                imgS0 = S0.create_img(useKey, self.initJKey, fixHe=True)
                imgS21 = self._applyH(S2, H21, self.initJKey)
                imgS20 = self._applyH(S2, H20, self.initJKey)
                e1 = err.err(imgS21, imgS1)
                e2 = err.err(imgS20, imgS0)
                if e1 < e2:
                    H1i = S1.data.loadH(initS.index, key)
                    H2i = np.dot(H1i, H21)
                    space_map.Info("LDMMgrMulti: Affine %d/%d %s->%s by S1 %.4f>%.4f" % (i+1, len(self.slices), S1.index, S2.index, e1, e2))
                    S2.data.saveH(H2i, initS.index, key)
                    S2.applyH(self.initJKey, H2i, self.alignKey)
                    if show:
                        self.show_align(S1, S2, useKey, self.alignKey, self.alignKey)  
                else:
                    H0i = S0.data.loadH(initS.index, key)
                    H2i = np.dot(H0i, H20)
                    space_map.Info("LDMMgrMulti: Affine %d/%d %s->%s by S0 %.4f>%.4f" % (i+1, len(self.slices), S0.index, S2.index, e2, e1))
                    S2.data.saveH(H2i, initS.index, key)
                    S2.applyH(self.initJKey, H2i, self.alignKey)
                    if show:
                        self.show_align(S0, S2, useKey, self.alignKey, self.alignKey) 
        space_map.Info("LDMMgrMulti: Finish Affine Pair&Merge")

    def affine_fix(self, useKey, show=False):
        Hs = []
        for i in range(len(self.slices) - 1):
            S1 = self.slices[i]
            S2 = self.slices[i+1]
            H21 = S2.data.loadH(S1.index, self.affineKey)
            Hs.append(H21)
        Hs = self._affine_fix(useKey, Hs)
        for i in range(len(self.slices) - 1):
            S1 = self.slices[i]
            S2 = self.slices[i+1]
            S2.data.saveH(Hs[i], S1.index, self.affineKey)
            S2.applyH(self.initJKey, Hs[i], self.alignKey)
            if show:
                self.show_align(S1, S2, useKey, self.alignKey, self.alignKey)    

    def create_last_merge(self, newI):
        if len(self.last_imgs) == 0 or self.last_img_ratio == 0.0:
            return newI
        lasts = np.array(self.last_imgs)
        newI = self.last_img_ratio * np.mean(lasts, axis=0) + (1 - self.last_img_ratio) * newI
        return newI

    def _ldm_pair(self, indexI, indexJ, slices, toKey, show=False):
        sI = slices[indexI]
        sJ = slices[indexJ]
        useKey = SliceImg.DF
        imgI1 = sI.create_img(useKey, toKey,
                                mchannel=False, scale=True, fixHe=True)
        imgJ2 = sJ.create_img(useKey, toKey,
                                mchannel=False, scale=True, fixHe=True)
        ldm = space_map.registration.SVFLDDMM()
        ldm.device = space_map.DEVICE
        ldm.grid_size = (8, 8)
        imgI1 = self.create_last_merge(imgI1)
        ldm.load_img(imgI1, imgJ2)
        imgI2 = ldm.run()
        self.show_err(imgJ2, imgI1, imgI2, sJ.index)
        space_map.show_compare_channel(imgJ2, imgI2)
        ps = sJ.imgs["DF"].ps(toKey)
        ps2 = ldm.apply_points2d(ps, space_map.XYD)
        # ps2 = spacemap.points.fix_points(imgI1, ps2)
        sJ.imgs["DF"].save_points(ps2, toKey)
        imgJ1 = space_map.show_img(ps2)
        self.last_imgs.append(imgJ1)
        # spacemap.utils.show_flow.analyze_distortion(ldm.mgr.flow, title=f"High Res Loss + Low Res Grid Result")
        if not show:
            return
        self.show_align(sI, sJ, useKey, toKey, toKey)
        return imgJ1

    def ldm_pair(self,
                 fromKey, toKey,
                 show=False):
        """ customFunc: ([Slice], index, dfKey) -> img """
        space_map.Info("LDMMgrMulti: Start LDM Pair")

        for s in self.slices:
            s.applyH(fromKey, None, toKey)
        
        self.last_imgs = []
        for i in range(len(self.slices1) - 1):
            space_map.XYD = self.rawXYD
            self._ldm_pair(i, i+1, self.slices1, toKey, False)
            if show:
                self.show_align(self.slices1[i], self.slices1[i+1], SliceImg.DF, toKey, toKey)
                
        self.last_imgs = []
        for i in range(len(self.slices2) - 1):
            space_map.XYD = self.rawXYD
            self._ldm_pair(i, i+1, self.slices2, toKey, False)
            if show:
                self.show_align(self.slices2[i], self.slices2[i+1], SliceImg.DF, toKey, toKey)
        space_map.XYD = self.rawXYD
        space_map.Info("LDMMgrMulti: Finish LDM Pair")

    def ldm_global(self, fromKey, toKey):
        imgs = []
        for s in self.slices:
            img = s.create_img(SliceImg.DF, fromKey,
                                mchannel=False, scale=True, fixHe=True)
            imgs.append(img)
        m = space_map.registration.SVFLDDMM()
        imgs_, flow_ = m.run_global(imgs)
        imgs2 = []
        for i, s in enumerate(self.slices):
            ps = s.imgs["DF"].ps(fromKey)
            flow = flow_[i]
            ps2 = m.apply_points2d(ps, space_map.XYD, flow)
            # ps2 = spacemap.points.fix_points(imgs_[i], ps2)
            s.imgs["DF"].save_points(ps2, toKey)
            imgs2.append(space_map.show_img(ps2))

        for i in range(len(imgs_) - 1):
            e1 = self.err.computeI(imgs[i], imgs[i+1], False)
            e2 = self.err.computeI(imgs2[i], imgs2[i+1], False)
            space_map.Info("Err LDMGlobal %d: %.5f->%.5f" % (i, e1, e2))
    
    def cpd_affine(self, show=True):
        from ..affine_block import CPDAffine
        dfs = [s.ps(Slice.rawKey) for s in self.slices]
        Hs = CPDAffine.compute_all(dfs)
        Hlast = Hs[0]
        Hs2 = [Hlast]
        for i in range(1, len(Hs)):
            Hlast = Hs[i] @ Hlast
            Hs2.append(Hlast)
        for i, H in enumerate(Hs2):
            self.slices[i].applyH(Slice.rawKey, H, Slice.affineKey)
            if show:
                self.show_align(self.slices[i], self.slices[1], SliceImg.DF, Slice.affineKey, Slice.affineKey)
