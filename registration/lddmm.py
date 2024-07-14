import spacemap
from .base import Registration
import numpy as np
from spacemap import LDDMM2D


def lddmm_main(ldm, err=0.1):
    """ J -> I """
    ldm.setParams('do_lddmm', 0)
    ldm.setParams('do_affine', 1)
    ldm.setParams('v_scale', 8.0)
    ldm.setParams('target_err_skip', err)
    ldm.setParams('epsilon', 10000)
    ldm.setParams('niter', 300)
    ldm.run()
    ldm.setParams('epsilon', 1000)
    ldm.setParams('v_scale', 4.0)
    ldm.setParams('target_err_skip', err)
    ldm.setParams('niter', 1000)
    ldm.run()
    ldm.setParams('epsilon', 50)
    ldm.setParams('v_scale', 1.0)
    ldm.setParams('target_err_skip',err)
    ldm.setParams('niter', 6000)
    ldm.run()
    ldm.setParams('target_err_skip', err)
    ldm.setParams('epsilon', 1000)
    ldm.setParams('niter', 20000)
    ldm.setParams('do_lddmm', 1)
    ldm.setParams('do_affine', 0)
    ldm.run()
    ldm.setParams('epsilon', 1)
    ldm.setParams('niter', 20000)
    ldm.run()
    return ldm

def lddmm_main2(ldm, err=0.1):
    """ J -> I """
    ldm.setParams('target_err_skip', err)
    ldm.setParams('epsilon', 1000)
    ldm.setParams('niter', 20000)
    ldm.setParams('do_lddmm', 1)
    ldm.setParams('do_affine', 0)
    ldm.run()
    ldm.setParams('epsilon', 1)
    ldm.setParams('niter', 20000)
    ldm.run()
    return ldm


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
            lddmm_main(self.ldm, err=self.err)
        else:
            # continue
            lddmm_main2(self.ldm, err=self.err)
            
    def _get_init(self, restart=False):
        do_l = 0 if restart else 1
        self.ldm = LDDMM2D(template=self.imgJ, 
                           target=self.imgI, do_affine=1,do_lddmm=do_l, 
                           nt=LDDMMRegistration.nt,optimizer='adam', sigma=20.0,sigmaR=20.0, gpu_number=self.gpu, target_err=0.1,verbose=LDDMMRegistration.verbose, target_step=20000, show_init=False)

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
        return self.ldm.applyThisTransformNT2d(img).cpu().numpy()

    def load_img(self, imgI, imgJ):
        self.imgI = imgI
        self.imgJ = imgJ
        if self.ldm is not None:
            params = self.output_params()
            # J -> I
            self.load_params(params)
    