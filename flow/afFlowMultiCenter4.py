import spacemap
from spacemap import Slice, SliceImg
import numpy as np
from .afFlow2Multi import AutoFlowMultiCenter2

class AutoFlowMultiCenter4(AutoFlowMultiCenter2):
    def __init__(self, slices: list[Slice],
                 initJKey=Slice.rawKey,
                 alignMethod=None,
                 gpu=None):
        super().__init__(slices, initJKey, alignMethod, gpu)
        spacemap.Info("AutoFlowMultiCenter4: Init")
        self.rawXYD = spacemap.XYD
        self.last_imgs = []
        self.last_img_ratio = 0.9

    def create_last_merge(self, newI):
        if len(self.last_imgs) == 0 or self.last_img_ratio == 0.0:
            return newI
        lasts = np.array(self.last_imgs)
        # newI = self.last_img_ratio * np.median(lasts, axis=0) + (1 - self.last_img_ratio) * newI
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
        ldm = spacemap.registration.SVFLDDMM()
        ldm.device = spacemap.DEVICE
        ldm.grid_size = (8, 8)
        imgI1 = self.create_last_merge(imgI1)
        ldm.load_img(imgI1, imgJ2)
        imgI2 = ldm.run()
        self.show_err(imgJ2, imgI1, imgI2, sJ.index)
        ps = sJ.imgs["DF"].ps(toKey)
        ps2 = ldm.apply_points2d(ps, spacemap.XYD)
        ps2 = spacemap.points.fix_points(imgI1, ps2)
        sJ.imgs["DF"].save_points(ps2, toKey)
        imgJ1 = spacemap.show_img(ps2)
        self.last_imgs.append(imgJ1)
        spacemap.utils.show_flow.analyze_distortion(ldm.mgr.flow, title=f"High Res Loss + Low Res Grid Result")
        if not show:
            return
        self.show_align(sI, sJ, useKey, toKey, toKey)
        return imgJ1

    def ldm_pair(self,
                 fromKey, toKey,
                 show=False):
        """ customFunc: ([Slice], index, dfKey) -> img """
        spacemap.Info("LDMMgrMulti: Start LDM Pair")

        for s in self.slices:
            s.applyH(fromKey, None, toKey)
        
        self.last_imgs = []
        for i in range(len(self.slices1) - 1):
            spacemap.XYD = self.rawXYD
            self._ldm_pair(i, i+1, self.slices1, toKey, False)
            if show:
                self.show_align(self.slices1[i], self.slices1[i+1], SliceImg.DF, toKey, toKey)
                
        self.last_imgs = []
        for i in range(len(self.slices2) - 1):
            spacemap.XYD = self.rawXYD
            self._ldm_pair(i, i+1, self.slices2, toKey, False)
            if show:
                self.show_align(self.slices2[i], self.slices2[i+1], SliceImg.DF, toKey, toKey)
        spacemap.XYD = self.rawXYD
        spacemap.Info("LDMMgrMulti: Finish LDM Pair")

    def ldm_global(self, fromKey, toKey):
        imgs = []
        for s in self.slices:
            img = s.create_img(SliceImg.DF, fromKey,
                                mchannel=False, scale=True, fixHe=True)
            imgs.append(img)
        m = spacemap.registration.SVFLDDMM()
        imgs_, flow_ = m.run_global(imgs)
        imgs2 = []
        for i, s in enumerate(self.slices):
            ps = s.imgs["DF"].ps(fromKey)
            flow = flow_[i]
            ps2 = m.apply_points2d(ps, spacemap.XYD, flow)
            # ps2 = spacemap.points.fix_points(imgs_[i], ps2)
            s.imgs["DF"].save_points(ps2, toKey)
            imgs2.append(spacemap.show_img(ps2))

        for i in range(len(imgs_) - 1):
            e1 = self.err.computeI(imgs[i], imgs[i+1], False)
            e2 = self.err.computeI(imgs2[i], imgs2[i+1], False)
            spacemap.Info("Err LDMGlobal %d: %.5f->%.5f" % (i, e1, e2))
    