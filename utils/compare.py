import spacemap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def filter_part_df(dfI: np.array, xyrange):
    minx, maxx, miny, maxy = xyrange
    dfI = dfI[(dfI["x"] > minx) & (dfI["x"] < maxx) & (dfI["y"] > miny) & (dfI["y"] < maxy)]
    return dfI

def cmp_filter_part(dfI: np.array, dfJ: np.array, xyrange):
    minx, maxx, miny, maxy = xyrange
    dfI = dfI[(dfI[:, 0] > minx) & (dfI[:, 0] < maxx) & (dfI[:, 1] > miny) & (dfI[:, 1] < maxy)]
    dfJ = dfJ[(dfJ[:, 0] > minx) & (dfJ[:, 0] < maxx) & (dfJ[:, 1] > miny) & (dfJ[:, 1] < maxy)]
    return dfI, dfJ

def cmp_filter_part_show(dfI: np.array, dfJ: np.array, xy, size, labels, s=1, alpha=0.5, tag="", trans=False):
    x, y = xy
    minx, maxx, miny, maxy = x, x+size, y, y+size
    xyr = [minx, maxx, miny, maxy]
    dfI = dfI[(dfI[:, 0] > minx) & (dfI[:, 0] < maxx) & (dfI[:, 1] > miny) & (dfI[:, 1] < maxy)]
    dfJ = dfJ[(dfJ[:, 0] > minx) & (dfJ[:, 0] < maxx) & (dfJ[:, 1] > miny) & (dfJ[:, 1] < maxy)]
    spacemap.show_xy_np([dfI, dfJ], labels, legend=False, xylim=xyr, s=s, alpha=alpha, outTag=tag, transparent=trans)
    return dfI, dfJ

def cmp_imgs(imgs1, imgs2, err: spacemap.AffineFinder):
    result = np.zeros(len(imgs1))
    for i in range(len(imgs1)):
        result[i] = err.err(imgs1[i], imgs2[i])
    return np.mean(result), result
    
def cmp_adjacent_layers(df, start, end, err):
    """ start/end 从layer=start到layer=end """
    return cmp_layers(df, df, start, end, err)

def cmp_layers(df1, df2, start, end, err):
    """ start/end 从layer=start到layer=end """
    img1s = []
    img2s = []
    for layer in range(start, end):
        df1_ = df1[df1["layer"] == layer][["x", "y"]].copy()
        df2_ = df2[df2["layer"] == layer+1][["x", "y"]].copy()
        img1 = spacemap.show_img3(np.array(df1_.values))
        img2 = spacemap.show_img3(np.array(df2_.values))
        img1s.append(img1)
        img2s.append(img2)
    return cmp_imgs(img1s, img2s, err)

def convert_label(labels: list, conf=None):
    if conf is None:
        conf = {}
    maxx = len(conf.values()) 
    result = np.zeros(len(labels))
    for i in range(len(labels)):
        key = labels[i]
        if key in conf:
            result[i] = conf[key]
        else:
            result[i] = maxx
            conf[key] = maxx
            maxx += 1
    return result, conf

def cmp_layers_label(df1: pd.DataFrame, df2: pd.DataFrame, 
                     start: int, end: int, err: spacemap.AffineFinder):
    if df2 is None:
        df2 = df1.copy()
    labelI = df1["cell_type"].values.tolist()
    labelJ = df2["cell_type"].values.tolist()
    clasI, conf = convert_label(labelI)
    clasJ, conf = convert_label(labelJ, conf)
    clas = len(conf.values())
    
    err_detail = np.zeros((clas, end-start))
    errs = np.zeros(clas)
    count = np.zeros(clas)
    for i in range(clas):
        dfi = df1[clasI == i][["x", "y", "layer"]]
        dfj = df2[clasJ == i][["x", "y", "layer"]]
        count[i] = dfi.shape[0]
        err1 = cmp_layers(dfi, dfj, start, end, err)
        err_detail[i] = err1[1]
        errs[i] = err1[0]
    result = np.sum(errs * count) / np.sum(count)
    return result, err_detail

def merge_label(df, label, start, end):
    df1 = df.copy()
    df1["cell_type"] = ""
    for i in range(start, end+1):
        df1.loc[df1["layer"] == i, "cell_type"] = label[label["layer"] == i]["cell_type"]
    return df1
