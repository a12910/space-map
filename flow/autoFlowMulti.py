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
        
        self.applyDF = False
        self.applyImg = False
        
    def show_err(self, imgI, imgJ2, imgJ3, tag):
        e1 = self.err.computeI(imgI, imgJ2, False)
        e2 = self.err.computeI(imgI, imgJ3, False)
        spacemap.Info("Err LDMPairMulti %s: %.5f->%.5f" % (tag, e1, e2))
        
    def affine_pair(self, show=False):
        spacemap.Info("LDMMgrMulti: Start Affine Pair")
        for i in range(len(self.slices) - 1):
            S1 = self.slices[i]
            S2 = self.slices[i+1]
            spacemap.Info("LDMMgrMulti: Start Affine %d/%d %s->%s" % (i+1, len(self.slices), S1.index, S2.index))
            img1 = S1.create_img(self.initJKey, useDF=False, he=self.heImg)
            img2 = S2.create_img(self.initJKey, useDF=False, he=self.heImg)
            mgr = spacemap.affine_block.AutoAffineImgGrad(img1, img2, show=show)
            mgr.run()
            H = mgr.resultH_img()
            S2.data.saveH(H, S1.index)
        spacemap.Info("LDMMgrMulti: Finish Affine Pair")
            
    def affine_merge(self, show=False):
        spacemap.Info("LDMMgrMulti: Start Merge Affine")
        initS = self.slices[0]
        key = "cell"
        for i in range(0, len(self.slices) - 1):
            S1 = self.slices[i]
            S2 = self.slices[i+1]
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
                self.show_align(initS, S2, self.initJKey, self.alignKey)
        spacemap.Info("LDMMgrMulti: Affine Finished")
        
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
            imgI1 = sI.get_img(self.alignKey)
            imgJ2 = sJ.get_img(self.alignKey)
            
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
            sJ.save_value_img(imgJ3, self.finalKey)
            if show:
                self.show_align(sI, sJ, self.alignKey, self.finalKey)
        spacemap.Info("LDMMgrMulti: Finish LDM Pair")
    
    def ldm_merge(self, show=False):
        spacemap.Info("LDMMgrMulti: Start LDM Merge")
        INIT_S = self.slices[0]
        for i in range(len(self.slices) - 1):
            sI = self.slices[i]
            sJ = self.slices[i+1]
            newG = sJ.data.loadGrid(sI.index, "img")
            if i > 0:
                lastG = sI.data.loadGrid(INIT_S.index, "img")
                newG = spacemap.mergeImgGrid(newG, lastG)
                sJ.data.saveGrid(newG, INIT_S.index, "img")
            imgJ1 = sJ.get_img(self.alignKey)
            imgJ2 = spacemap.applyPointsByGrid(newG, imgJ1)
            sJ.save_value_img(imgJ2, self.ldmKey)
            if show:
                Slice.show_align(sI, sJ, self.finalKey, self.finalKey)
        spacemap.Info("LDMMgrMulti: Finish LDM Merge")
        
    def ldm_fix(self, show=False):
        spacemap.Info("LDMMgrMulti: Start LDM Fix")
        mgr = spacemap.registration.LDDMMRegistration()
        mgr.gpu = self.gpu
        initI = self.slices[0].index
        
        for i in range(1, len(self.slices) - 1):
            sI = self.slices[i]
            sJ = self.slices[i+1]
            if i == 0:
                imgJ2 = sI.get_img(self.ldmKey)
                sI.save_value_img(imgJ2, self.finalKey)
                continue
            imgI = sI.get_img(self.finalKey)
            imgJ = sJ.get_img(self.ldmKey)
            mgr.load_img(imgI, imgJ)
            mgr.run()
            imgJ2 = mgr.apply_img(imgJ)
            sJ.save_value_img(imgJ2, self.finalKey)
            grid = mgr.generate_img_grid()
            sJ.data.saveGrid(grid, initI, "fix")
            if show:
                Slice.show_align(sI, sJ, self.finalKey, self.finalKey)
        spacemap.Info("LDMMgrMulti: Finish LDM Fix")
        
    def ldm_final(self):
        spacemap.Info("LDMMgrMulti: Start LDM Final")
        initI = self.slices[0].index        
        for i in range(1, len(self.slices) - 1):
            sI = self.slices[i]
            sJ = self.slices[i+1]
            if i == 0:
                grid1 = sI.data.loadGrid(initI, "img")
                sJ.data.saveGrid(grid1, initI, "final_ldm")
                continue
            grid = sJ.data.loadGrid(initI, "fix")
            grid1 = sJ.data.loadGrid(initI, "img")
            gridi = spacemap.mergeImgGrid(grid1, grid)
            sJ.data.saveGrid(gridi, initI, "final_ldm")
        spacemap.Info("LDMMgrMulti: Finish LDM Final")
