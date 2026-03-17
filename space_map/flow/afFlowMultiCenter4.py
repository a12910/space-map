from __future__ import annotations
import space_map
from space_map import Slice, SliceImg
import numpy as np
import os


class AutoFlowMultiCenter4:
    def __init__(self, slices: list[Slice],
                 initJKey=Slice.rawKey,
                 alignMethod=None,
                 gpu=None):
        space_map.Info("AutoFlowMultiCenter4: Init")
        self.slices = slices
        self.initJKey = initJKey
        self.alignKey = Slice.align1Key
        self.affineKey  ="cell"
        self.ldmKey = Slice.align2Key
        self.continueStart = 0
        self.gpu=space_map.DEVICE
        if space_map.DEVICE == "cpu":
            self.gpu = None
        self.finalKey = Slice.finalKey
        self.enhanceKey = Slice.enhanceKey
        self.err = space_map.find.default()
        self.alignMethod = alignMethod
        self.rawXYD = space_map.XYD
        self.last_imgs = []
        self.last_img_ratio = 0.5

        self.cIndex = len(slices) // 2
        self.slices1 = self.slices[self.cIndex:]
        self.slices2 = self.slices[:self.cIndex+1]
        self.slices2.reverse()

    
    def show_err(self, imgI, imgJ2, imgJ3, tag):
        e1 = self.err.computeI(imgI, imgJ2, False)
        e2 = self.err.computeI(imgI, imgJ3, False)
        space_map.Info("Err LDMPairMulti %s: %.5f->%.5f" % (tag, e1, e2))

    def show(self, useKey, dfKey):
        count = len(self.slices)
        for i in range(count - 1):
            self.show_align(self.slices[i], self.slices[i+1], useKey, dfKey, dfKey)

    def show_align(self, S1: Slice, S2: Slice, useKey, key1, key2):
        img1 = S1.create_img(useKey, key1, scale=True, fixHe=True)
        img2 = S2.create_img(useKey, key2, scale=True, fixHe=True)
        e = self.err.err(img1, img2)
        space_map.Info("Show AlignErr %s/%s %s/%s %f" % (S1.index, key1, S2.index, key2, e))
        Slice.show_align(S1, S2, key1, key2, useKey)

    def _affine_fix(self, imgs, Hs):
        return Hs

    def _affine_run(self, S1, S2, useKey, i, show, initS, key):
        img1 = S1.create_img(useKey, self.initJKey, fixHe=True)
        img2 = S2.create_img(useKey, self.initJKey, fixHe=True)
        space_map.Info("LDMMgrMulti: Start Affine %d/%d %s->%s" % (i+1, len(self.slices), S1.index, S2.index))
        # lastH = S1.data.loadH(initS.index, key) if i > 0 else np.eye(3)
        df = None
        if useKey == "DF":
            df = (S1.ps(self.initJKey), S2.ps(self.initJKey))
        mgr = space_map.affine_block.AutoAffineImgKey(img1, img2, show=show, method=self.alignMethod)
        # mgr.each.lastImgs.lastH = lastH
        mgr.run(df)
        H21 = mgr.resultH_img()
        S2.data.saveH(H21, S1.index, key)
        S2.applyH(self.initJKey, H21, self.alignKey)
        return H21


    def affine(self, useKey, show=False):
        """ customFunc: (Slice) -> img"""
        space_map.Info("LDMMgrMulti: Start Affine Pair&Merge")
        method = self.alignMethod
        if method is None:
            method = "sift_vgg"
        space_map.affine_block.AutoAffineImgKey.restart()
        initS = self.slices[0]
        key = self.affineKey
        
        for i in range(self.continueStart, len(self.slices) - 1):
            S1 = self.slices[i]
            S2 = self.slices[i+1]
            H21 = self._affine_run(S1, S2, useKey, i, show, initS, key)
            if show:
                self.show_align(S1, S2, useKey, self.initJKey, self.alignKey)                    
            if i == 0:
                H1i = np.eye(3)
                S1.applyH(self.initJKey, H1i, self.alignKey)
            else:
                H1i = S1.data.loadH(initS.index, key)
            H2i = np.dot(H1i, H21)
            S2.data.saveH(H2i, initS.index, key)
            S2.applyH(self.initJKey, H2i, self.alignKey)
            # imgJ_new = S2.create_img(useKey, self.alignKey, fixHe=True)
            # mgr.each.lastImgs.add_img(imgJ_new)
            
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
