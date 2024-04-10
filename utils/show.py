import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np
import spacemap
import cv2

def conv_2d(img, kernel):
    img1 = ss.convolve2d(img, kernel, mode="same")
    return img1

def show_img_labels(xy: np.array, labels: np.array):
    xyr = spacemap.XYRANGE
    xyd = spacemap.XYD
    sx, sy = xyr[1] // xyd, xyr[3] // xyd
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
               xylim=None, s=1, alpha=0.2, 
               legend=True, transparent=False, 
               outTag="", outSave=True):
    fig,ax = plt.subplots()
    xyr = spacemap.XYRANGE
    if xylim is None:
        plt.xlim((xyr[0], xyr[1]))
        plt.ylim((xyr[2], xyr[3]))
    else:
        plt.xlim((xylim[0], xylim[1]))
        plt.ylim((xylim[2], xylim[3]))
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

def show_xy(dfs: list[pd.DataFrame], labels: list[str], keyx: str or list ="x", keyy: str or list="y", xylim=None, s=1, alpha=0.2):
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
    img = np.zeros((int((xyrange[1]-xyrange[0])/xyd), int((xyrange[3]-xyrange[2])/xyd)), dtype=np.float64)
    values1 = values.copy()
    values1[:, 0] = (values1[:, 0] - xyrange[0]) // xyd
    values1[:, 1] = (values1[:, 1] - xyrange[2]) // xyd
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
    img0 = spacemap.show_img3(points, {"raw": 1})
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
    
def show_img3(values: np.array, imgConf=None):
    if imgConf is None:
        imgConf = spacemap.IMGCONF
    kernel = imgConf.get("kernel", 0)
    mid = imgConf.get("mid", 0)
    raw = imgConf.get("raw", 0)
    
    xyrange = spacemap.XYRANGE
    xyd = spacemap.XYD
    
    img = np.zeros((int((xyrange[1]-xyrange[0])/xyd), int((xyrange[3]-xyrange[2])/xyd)), dtype=np.float64)
    values1 = values.copy()
    values1[:, 0] = (values1[:, 0] - xyrange[0]) // xyd
    values1[:, 1] = (values1[:, 1] - xyrange[2]) // xyd
    if raw == 0:
        values_ = set([(int(x), int(y)) for x, y in values1])
    else:
        values_ = [(int(x), int(y)) for x, y in values1]
    for ix, iy in values_:
        if ix < 0 or ix >= img.shape[0] or iy < 0 or iy >= img.shape[1]:
            continue
        img[ix-kernel: ix+kernel+1, iy-kernel:iy+kernel+1] += 1
    
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
        img1 = spacemap.compute_interest_area(img, softKernel)
        img += img1
    if mid > 0:
        img = cv2.medianBlur(img.astype(np.float32), mid)
    return img

def plot_hist(img):
    img = np.array(img, dtype=np.uint8)
    grayHist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(range(256), grayHist, 'r', linewidth=1.5, c='red')
    y_maxValue = np.max(grayHist)
    plt.axis([0, 255, 0, y_maxValue]) # x和y的范围
    plt.xlabel("gray Level")
    plt.ylabel("Number Of Pixels")
    plt.show()
    
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

def show_images_form(imgs, shape, titles):
    sx, sy = shape
    fig, axes = plt.subplots(shape[0], shape[1], figsize=(12*shape[1], 12*shape[0]))
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