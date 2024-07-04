import spacemap
from .base import Registration
import numpy as np
from spacemap import LDDMM2D

class LDDMMRegistration(Registration):
    nt=5
    verbose=100
    def __init__(self):
        super().__init__("LDDMM")
        self.imgI = None
        self.imgJ = None
        self.ldm: spacemap.LDDMM2D = None
        self.gpu = None
        self.err = 0.1

    def run(self):
        if self.imgI is None or self.imgJ is None:
            raise Exception("LDDMMRegistration: imgI or imgJ is None")
        if self.ldm is None:
            self._get_init(restart=True)
            # restart
            spacemap.lddmm_main(self.ldm, err=self.err)
        else:
            # continue
            spacemap.lddmm_main2(self.ldm, err=self.err)
            
    def _get_init(self, restart=False):
        do_l = 0 if restart else 1
        self.ldm = LDDMM2D(template=self.imgJ, 
                           target=self.imgI, do_affine=1,do_lddmm=do_l, 
                           nt=LDDMMRegistration.nt,optimizer='adam', sigma=20.0,sigmaR=40.0, gpu_number=self.gpu, target_err=0.1,verbose=LDDMMRegistration.verbose, target_step=20000, show_init=False)

    def load_params_path(self, path):
        path = path + ".npz"
        params = np.load(path)
        self.load_params(params)

    def load_params(self, params):
        self._get_init(restart=False)
        vt0 = list(params["vt0"])
        vt1 = list(params["vt1"])
        affineA = params["affineA"]        
        self.ldm.loadTransforms(vt0, vt1, affineA)

    def output_params(self):
        if self.ldm is None:
            raise Exception("LDDMMRegistration: ldm is None")
        vt0, vt1, affine = self.ldm.outputTransforms()
        vt0 = np.array(vt0)
        vt1 = np.array(vt1)
        affine = np.array(affine)
        params = {
            "vt0": vt0,
            "vt1": vt1,
            "affineA": affine
        }
        return params

    def output_params_path(self, path):
        params = self.output_params()
        path = path + ".npz"
        np.savez_compressed(path, **params)

    def generate_img_grid(self):
        if self.ldm is None:
            raise Exception("LDDMMRegistration: ldm is None")
        grid = self.ldm.generateTransFormGridImg()
        return grid

    def apply_points2d(self, points):
        if self.ldm is None:
            raise Exception("LDDMMRegistration: ldm is None")
        pass

    def apply_img(self, img):
        if self.ldm is None:
            raise Exception("LDDMMRegistration: ldm is None")
        return self.ldm.applyThisTransformNT2d(img)

    def load_img(self, imgI, imgJ):
        self.imgI = imgI
        self.imgJ = imgJ
        if self.ldm is not None:
            params = self.output_params()
            # J -> I
            self.load_params(params)
    