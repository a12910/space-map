import numpy as np
import pandas as pd
import spacemap

def applyH_np(df: np.array, H) -> np.array:
    df2 = df.copy()
    H = np.array(H)
    df2[:, 0] = (df[:, 0] * H[0, 0] + df[:, 1] * H[0, 1]) + H[0, 2]
    df2[:, 1] = (df[:, 0] * H[1, 0] + df[:, 1] * H[1, 1]) + H[1, 2]
    return df2

def lddmm_init(imgI, imgJ, verbose=100, gpu_number=None) -> spacemap.LDDMM:
    if len(imgI.shape) == 2:
        imgI = np.stack([imgI, imgI], axis=2)
        imgJ = np.stack([imgJ, imgJ], axis=2)
        
    ldm = spacemap.LDDMM(template=imgJ,target=imgI,
                          do_affine=1,do_lddmm=0,
                          a=7,
                          optimizer='adam',
                          sigma=20.0,sigmaR=40.0,
                          gpu_number=gpu_number,
                          target_err=30,
                          verbose=verbose,
                          target_step=20000,
                          show_init=False)
    return ldm

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
    ldm.setParams('do_affine', 1)
    ldm.run()
    ldm.setParams('epsilon', 1)
    ldm.setParams('niter', 20000)
    ldm.run()
    return ldm
