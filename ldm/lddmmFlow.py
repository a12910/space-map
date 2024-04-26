import numpy as np
import pandas as pd
import spacemap
from spacemap import Slice
import matplotlib.pyplot as plt

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

def lddmm_init2D(imgI, imgJ, gpu=None, verbose=100):
    if len(imgI.shape) > 2:
        imgI = np.mean(imgI[:, :, :3], axis=2)
    if len(imgJ.shape) > 2:
        imgJ = np.mean(imgJ[:, :, :3], axis=2)
    ldm = spacemap.LDDMM2D(template=imgI,target=imgJ,
                              do_affine=1,do_lddmm=0,
                              a=7,
                              optimizer='adam',
                              sigma=20.0,sigmaR=40.0,
                              gpu_number=gpu,
                              target_err=0.1,
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

def lddmm_main2(ldm, err=0.1):
    """ J -> I """
    ldm.setParams('target_err_skip', err)
    ldm.setParams('epsilon', 1000)
    ldm.setParams('niter', 20000)
    ldm.setParams('do_lddmm', 1)
    ldm.run()
    ldm.setParams('epsilon', 1)
    ldm.setParams('niter', 20000)
    ldm.run()
    return ldm

from spacemap import Slice
def lddmm_run(index, index2, projectF, 
              baseFromKey, baseToKey, dfMode=True,
              targetFromKey=None,
              gridSave=None, imgMChannel=False,
              outputs=None, gpu=None,
              verbose=100, err=0.1, show=False, saveGrid=False):
    spacemap.Info("Start Index %d to %d" % (index2, index))
    if targetFromKey is None:
        targetFromKey = baseToKey
    TARGET_S = Slice(index, projectF, dfMode=dfMode)
    imgI1 = TARGET_S.create_img(baseToKey, useDF=dfMode)
    NEW_S = Slice(index2, projectF, dfMode=dfMode)
    imgJ2 = NEW_S.create_img(baseFromKey, useDF=dfMode)
    
    ldm = spacemap.LDDMM2D(template=imgI1,target=imgJ2,
                              do_affine=1,do_lddmm=0,
                              a=7,
                              optimizer='adam',
                              sigma=20.0,sigmaR=40.0,
                              gpu_number=gpu,
                              target_err=err,
                              verbose=verbose,
                              target_step=20000,
                              show_init=False)
    if outputs is not None:
        ldm.loadTransforms(*outputs)
        ldm.setParams('target_err_skip', err)
        ldm.setParams('epsilon', 1000)
        ldm.setParams('niter', 20000)
        ldm.setParams('do_lddmm', 1)
        ldm.run()
        ldm.setParams('epsilon', 1)
        ldm.setParams('niter', 20000)
        ldm.run()
    else:
        spacemap.lddmm_main(ldm, err)
        
    outputs = ldm.outputTransforms()
    if gridSave is None:
        gridSave = [(baseFromKey, baseToKey, dfMode)]
    for fromKey, toKey, dfm in gridSave:
        if dfm:
            points2 = ldm.applyThisTransformPoints2D(NEW_S.to_points(fromKey))
            NEW_S.save_value_points(points2, dfKey=toKey)
        else:
            if imgMChannel:
                imgJ2_ = NEW_S.create_img(fromKey, useDF=dfm,
                                          mchannel=imgMChannel)
                imgJ3 = np.zeros_like(imgJ2_)
                for i in range(imgJ2_.shape[2]):
                    imgJ3[:,:,i] = ldm.applyThisTransform2d(imgJ2_[:,:,i])[0][-1].cpu()
            else:
                imgJ3 = ldm.applyThisTransform2d(imgJ2)[0][-1].cpu()
            NEW_S.save_value_img(imgJ3, toKey)
    if saveGrid:
        grid = ldm.generateTransFromGrid()
        NEW_S.data.saveGrid(grid, index, tag=toKey)
    if show:
        Slice.show_align(TARGET_S, NEW_S, keyI=toKey, keyJ=toKey, forIMG= not dfMode)
        plt.show()
    return outputs

