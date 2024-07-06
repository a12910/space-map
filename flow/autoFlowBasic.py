import spacemap
from spacemap import Slice
import numpy as np

class AutoFlowBasic:
    def __init__(self, slices: list[Slice], 
                 initJKey=Slice.rawKey,
                 alignMethod=None,
                 gpu=None):
        self.slices = slices
        self.initJKey = initJKey
        self.alignKey = Slice.align1Key
        self.pairGridKey = "img"
        self.fixGridKey = "fix"
        self.finalGridKey = "final_ldm"
        self.ldmKey = Slice.align2Key
        self.continueStart = 0
        self.gpu=gpu
        self.finalKey = Slice.finalKey
        self.enhanceKey = Slice.enhanceKey
        self.err = spacemap.find.default()
        self.verbose = 100
        self.finalErr = 0.1
        self.enhanceErr = 0.01
        self.heImg = False
        self.alignMethod = alignMethod
        self.dfMode = False
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
            
        for i in range(self.continueStart, len(self.slices) - 1):
            S1 = self.slices[i]
            S2 = self.slices[i+1]
            spacemap.Info("LDMMgrMulti: Start Affine %d/%d %s->%s" % (i+1, len(self.slices), S2.index, S1.index))
            if affine:
                img1 = S1.create_img(self.initJKey, useDF=self.dfMode, 
                                     he=self.heImg)
                img2 = S2.create_img(self.initJKey, useDF=self.dfMode, 
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
                if self.applyDF or self.dfMode:
                    H2i_np = spacemap.img.to_npH(H2i)
                    S2.applyH(self.initJKey, H2i_np, self.alignKey, forIMG=False)
            if show:
                self.show_align(S1, S2, self.alignKey, self.alignKey)
            
        spacemap.Info("LDMMgrMulti: Finish Affine Pair&Merge")
        
    def affine_pair(self, show=False):
        self.affine(affine=True, merge=False, show=show)
            
    def affine_merge(self, show=False):
        self.affine(merge=True, affine=False, show=show)
        
    def show_align(self, S1, S2, key1, key2):
        img1 = S1.create_img(key1, he=self.heImg, useDF=self.dfMode)
        img2 = S2.create_img(key2, he=self.heImg, useDF=self.dfMode)
        e = self.err.err(img1, img2)
        spacemap.Info("Show AlignErr %s/%s %s/%s %f" % (S1.index, key1, S2.index, key2, e))
        
        if self.applyDF or self.dfMode:
            Slice.show_align(S1, S2, key1, key2, forIMG=False, imgHE=self.heImg)
        else:
            Slice.show_align(S1, S2, key1, key2, forIMG=True, imgHE=self.heImg)
        
    def _apply_grid(self, S: Slice, fromKey, toKey, grid):
        imgJ2_ = S.create_img(fromKey, mchannel=True, useDF=self.dfMode)
        imgJ3_ = spacemap.img.apply_img_by_grid(imgJ2_, grid)
        meanJ2 = np.mean(imgJ2_)
        meanJ3 = np.mean(imgJ3_)
        imgJ3_ = imgJ3_ * meanJ2 / meanJ3
        
        S.save_value_img(imgJ3_, toKey)
        if self.applyDF or self.dfMode:
            ps = S.to_points(fromKey)
            ps2, _ = spacemap.points.apply_points_by_grid(grid, ps)
            # ps2 = spacemap.applyPointsByGrid(grid, ps)
            S.save_value_points(ps2, toKey)
   
        