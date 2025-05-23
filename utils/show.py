import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.signal import convolve2d
import numpy as np
import spacemap
import cv2

def show_img_labels(xy: np.array, labels: np.array):
    xyr = spacemap.XYRANGE
    xyd = spacemap.XYD
    sx, sy = xyr // xyd, xyr // xyd
    mapp = np.zeros((sx, sy,
                        len(labels[0])))
    mapCount = np.zeros((sx, sy))
    for i in range(xy.shape[0]):
        x, y = xy[i]
        x_, y_ = int(x // xyd), int(y // xyd)
        if x_ >= sx or y_ >= sy:
            continue
        mapp[x_, y_] += labels[i]
        mapCount[x_, y_] += 1
    return mapp, mapCount    

def show_xy_np(nps: [np.array], labels: [str], 
               xylim=None, s=1, alpha=0.1, 
               legend=True, transparent=False, 
               outTag="", outSave=True):
    fig,ax = plt.subplots()
    xyr = spacemap.XYRANGE
    if xylim is None:
        plt.xlim((0, xyr))
        plt.ylim((0, xyr))
    else:
        plt.xlim((0, xylim))
        plt.ylim((0, xylim))
    for i in range(len(nps)):
        label = labels[i]
        xI = np.array(nps[i][:, 0])
        yI = np.array(nps[i][:, 1])
        ax.scatter(xI,yI,s=s,alpha=alpha, label=label)
    ax.legend(markerscale = 10)
    if not legend:
        ax.get_legend().remove()
    if outSave:
        import time
        
        path = "%s/imgs/showxy%s_%s_%s.png" % (spacemap.BASE, outTag, "-".join(labels), 
                                            str(int(time.time())))
        fig.savefig(path, transparent=transparent)
    plt.show()

def show_xy(dfs: list[pd.DataFrame], labels: list[str], 
            keyx: str="x", keyy: str="y", 
            xylim=None, s=1, alpha=0.2):
    nps = []
    for i in range(len(dfs)):
        df = dfs[i]
        kx = keyx
        ky = keyy
        if not isinstance(keyx, str):
            kx = keyx[i]
        if not isinstance(keyy, str):
            ky = keyy[i]
        npp = np.array(df[[kx, ky]].values)
        nps.append(npp)
    show_xy_np(nps, labels, xylim, s, alpha)
    
def show_align_np(npI, npJ, titleI, titleJ):
    show_xy_np([npI, npJ], [titleI, titleJ])
    
def show_img4(values: np.array, kernel=0):
    # n*3
    xyrange = spacemap.XYRANGE
    xyd = spacemap.XYD
    img = np.zeros((int(xyrange/xyd), int(xyrange/xyd)), dtype=np.float64)
    values1 = values.copy()
    values1[:, 0] = (values1[:, 0]) // xyd
    values1[:, 1] = (values1[:, 1]) // xyd
    values_ = [(int(x), int(y), z) for x, y, z in values1]
    for ix, iy, iz in values_:
        if iz == 0:
            continue
        if ix < 0 or ix >= img.shape[0] or iy < 0 or iy >= img.shape[1]:
            continue
        img[ix: ix+kernel+1, iy-kernel:iy+kernel+1] += iz
    return img
    
def show_layer_img(points, labels):
    count, channels = labels.shape
    img0 = spacemap.show_img(points, {"raw": 1})
    shape = img0.shape
    rawI = np.zeros((channels + 1, *shape))
    rawI[0, :, :] = img0
    for i in range(channels):
        ps = np.zeros((count, 3))
        ps[:, :2] = points
        ps[:, 2] = labels[:, i]
        img = spacemap.show_img4(ps)
        rawI[i+1, :, :] = img
    return rawI
    
def show_img(values: np.array, imgConf=None, multi=1):
    if imgConf is None:
        imgConf = spacemap.IMGCONF
    kernel = imgConf.get("kernel", 0)
    mid = imgConf.get("mid", 0)
    raw = imgConf.get("raw", 0)
    
    xyrange = spacemap.XYRANGE
    xyd = spacemap.XYD
    imgsize = int(xyrange/xyd)
    img = np.zeros((imgsize, imgsize), dtype=int)
    values1 = (values // xyd).astype(int)
    valid_mask = (values1[:, 0] >= 0) & (values1[:, 0] < imgsize) & (values1[:, 1] >= 0) & (values1[:, 1] < imgsize)
    values2 = values1[valid_mask].astype(int)
    ix = values2[:, 0]
    iy = values2[:, 1]
    np.add.at(img, (ix, iy), 1)
    if raw == 0:
        img[img > 0] = 1
    if kernel > 0:
        k = np.ones((kernel+1, kernel+1))
        img = convolve2d(img, k, mode="same", boundary="fill", fillvalue=0)
    density = imgConf.get("density", 0)
    gauss = imgConf.get("gauss", 0)
    if gauss > 0:
        k = int(img.shape[0] * gauss) if gauss < 1 else int(gauss)
        img = cv2.GaussianBlur(img, (k, k), 0)
    if density > 0:
        d = 2 * int(density) + 1
        density_limit = imgConf.get("density_limit", 0)
        density_limit_ = int(d * d * density_limit)
        ones = np.ones((d, d))
        img_ = img.copy()
        img_[img_ > 0] = 1
        img1 = ss.convolve2d(img_, ones, mode="same")
        img[img1 < density_limit_] = 0    
    softKernel = imgConf.get("soft_kernel", 0)
    if softKernel > 0:
        img1 = spacemap.interest.compute_interest_area(img, softKernel)
        img += img1
    if mid > 0:
        img = cv2.medianBlur(img.astype(np.float32), mid)
    if multi > 1:
        img = img * multi
        img[img > 255] = 255
    img = img.astype(np.uint8)
    return img

def img_norm(imgI, imgJ):
    imgConf = spacemap.IMGCONF
    clahe = imgConf.get("clahe", 0)
    clahe_limit = imgConf.get("clahe_limit", 20)
    maxx, minn = imgI.max(), imgI.min()
    imgI = (imgI - minn) / (maxx - minn)
    imgJ = (imgJ - minn) / (maxx - minn)
    if clahe > 0:
        imgI_ = np.array(imgI, dtype=np.uint8)
        imgJ_ = np.array(imgJ, dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clahe_limit, 
                                tileGridSize=(8,8))
        imgI = clahe.apply(imgI_)
        imgJ = clahe.apply(imgJ_)
    return imgI, imgJ

def show_images_form(imgs, shape, titles, size=12):
    sx, sy = shape
    fig, axes = plt.subplots(shape[0], shape[1], 
                             figsize=(size*shape[1], size*shape[0]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            ii = i*shape[1] + j
            if ii >= len(imgs):
                continue
            if sx > 1 and sy > 1:
                axes[i, j].imshow(imgs[ii])
                axes[i, j].set_title(titles[ii])
            else:
                axes[ii].imshow(imgs[ii])
                axes[ii].set_title(titles[ii])
    plt.show()
    
def imshow(img, **kwargs):
    plt.imshow(img, **kwargs)
    plt.show()
    
def imsave(path, img: np.array):
    if np.mean(img) < 1.0:
        img[img >= 1.0] = 1.0
        img[img <= 0.0] = 0.0
    else:
        img = img.astype(np.uint8)
    img = img.copy(order='C')
    plt.imsave(path, img)
            
    
def show_images_form2(imgs, shape, titles):
    sx, sy = shape
    plt.figure(figsize=(8*sx, 8*sy))    
    for i in range(sx):
        for j in range(sy):
            ii = i*shape[1] + j
            if ii >= len(imgs):
                continue
            plt.subplot(sx, sy, ii+1)
            plt.imshow(imgs[ii])
            plt.title(titles[ii])
    plt.show()
    
def show_compare_img(imgI, imgJ, size=6, titleI="I", titleJ="J"):
    def __process(imgI):
        if len(imgI.shape) > 2 and imgI.shape[2] > 3:
            imgI = imgI[:, :, 3]
        if len(imgI.shape) != 3:
            imgI = np.stack([imgI, imgI, imgI], axis=2)
        minn, maxx = np.min(imgI), np.max(imgI)
        if minn == maxx:
            return imgI
        imgI = (imgI - minn) / (maxx - minn)
        imgI = np.array(imgI * 255)
        imgI = np.array(imgI, dtype=np.uint8)
        return imgI
    imgI = __process(imgI)
    imgJ = __process(imgJ)            
    if imgJ.shape != imgI.shape:
        imgJ = cv2.resize(imgJ, imgI.shape)
        
    diff = imgI - imgJ
    show_images_form([imgI, imgJ, diff], (1, 3), [titleI, titleJ, "Diff"], size=size)
    return imgI, imgJ

def show_compare_channel(imgI, imgJ, size=6, titleI="I", titleJ="J"):
    def __process(imgI):
        if len(imgI.shape) > 2:
            if imgI.shape[2] > 3:
                imgI = imgI[:, :, :3]
            imgI = imgI.mean(axis=2)
        minn, maxx = np.min(imgI), np.max(imgI)
        if minn == maxx:
            return imgI
        imgI = (imgI - minn) / (maxx - minn)
        imgI = np.array(imgI * 255)
        imgI = np.array(imgI, dtype=np.uint8)
        return imgI
    imgI = __process(imgI)
    imgJ = __process(imgJ)            
    if imgJ.shape != imgI.shape:
        imgJ = cv2.resize(imgJ, imgI.shape)
    imgIJ = np.zeros((imgI.shape[0], imgI.shape[1], 3), dtype=np.uint8)
    # imgIJ[:, :, 2] = imgI
    # imgJ[imgJ > 0] = 255
    # imgI[imgI > 0] = 255
    imgIJ[:, :, 1] = imgJ 
    imgIJ[:, :, 0] = imgI
    plt.imshow(imgIJ)
    plt.show()
    return imgI, imgJ
    