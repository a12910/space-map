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

def cmp_imgs(imgs1, imgs2, err: spacemap.AffineFinder=None):
    result = np.zeros(len(imgs1))
    if err is None:
        err = spacemap.find.FinderBasic("dice")
    for i in range(len(imgs1)):
        result[i] = err.err(imgs1[i], imgs2[i])
    return np.mean(result), result

def cmp_imgs_metric(imgs1, imgs2, err: spacemap.AffineFinder):
    result = np.zeros((len(imgs1), len(imgs1)))
    for i in range(len(imgs1)):
        for j in range(i+1, len(imgs1)):
            e = err.err(imgs1[i], imgs2[j])
            result[i, j] = e
            result[j, i] = e
    return result

def cmp_imgs_LXYC(imgs: np.array, err=None):
    """ 按照layer*width*height*channel对img进行比较 
        然后按照channel进行加权 
    """
    if err is None:
        err = spacemap.find.default()
    maxx = imgs.shape[3]
    err_result = np.zeros((imgs.shape[0]-1, maxx), dtype=float)
    for i in range(maxx):
        imgs1 = imgs[:, :, :, i]
        _, err1 = cmp_imgs(imgs1[:-1], imgs1[1:], err)
        err_result[:, i] = err1
    err_count = np.sum(imgs, axis=(0, 1, 2))
    all_count = np.sum(err_count)
    err_count = err_count / all_count
    err_result1 = np.mean(err_result, axis=0)
    err_result1 = err_result1 * err_count
    return err_result1, np.sum(err_result1)

def __generate_imgs(df, start, end):
    img1s = []
    for layer in range(start, end):
        df_ = df[df["layer"] == layer][["x", "y"]].copy()
        img1 = spacemap.show_img(np.array(df_.values))
        img1s.append(img1)
    return img1s
    
def cmp_adjacent_layers(df, start, end, err):
    """ start/end 从layer=start到layer=end """
    return cmp_layers(df, df, start, end, err)

def cmp_layers(df1, df2, start, end, err):
    """ start/end 从layer=start到layer=end """
    img1s = __generate_imgs(df1, start, end)
    img2s = __generate_imgs(df2, start+1, end+1)
    return cmp_imgs(img1s, img2s, err)

def convert_label_to_id(labels: list, conf=None):
    """ 将label标签转化为序号 """
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
                     start: int, end: int, err: spacemap.AffineFinder=None, label: pd.DataFrame=None):
    """ 根据celltype对相邻层进行比较 """
    if df2 is None:
        df2 = df1.copy()
    if label is not None:
        df1 = merge_label(df1, label, start, end)
        df2 = merge_label(df2, label, start+1, end+1)
    if err is None:
        err = spacemap.find.default()
    labelI = df1["cell_type"].values.tolist()
    labelJ = df2["cell_type"].values.tolist()
    clasI, conf = convert_label_to_id(labelI)
    clasJ, conf = convert_label_to_id(labelJ, conf)
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

def cmp_layers_metric(df1: pd.DataFrame, df2: pd.DataFrame,
                     start: int, end: int, err: spacemap.AffineFinder):
    """ 比较多层之间的相似度 """
    if df2 is None:
        df2 = df1.copy()
    img1s = __generate_imgs(df1, start, end)
    img2s = __generate_imgs(df2, start+1, end+1)
    return cmp_imgs_metric(img1s, img2s, err)
    
def merge_label(df, label, start, end):
    """ 将来自label的cell_type合并到df中 """
    df1 = df.copy()
    df1["cell_type"] = ""
    for i in range(start, end+1):
        df1.loc[df1["layer"] == i, "cell_type"] = label[label["layer"] == i]["cell_type"]
    return df1

def compare_workflow(dfs: dict[str:pd.DataFrame], start, end, 
                     err: spacemap.AffineFinder=None):
    """ 比较不同df之间相邻层的评分 """
    if err is None:
        err = spacemap.find.default()
    result = np.zeros((len(dfs), end-start))
    keys = list(dfs.keys())
    target = dfs.get("target", None)
    for i, k in enumerate(keys):
        df = dfs[k]
        if target is not None:
            result[i] = cmp_layers(df, target, start, end, err)[1]
        else:
            result[i] = cmp_adjacent_layers(df, start, end, err)[1]
    result = result.T
    df = pd.DataFrame(result, columns=keys)
    return df
    
# 36, 37