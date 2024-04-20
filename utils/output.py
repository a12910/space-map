import pandas as pd
import numpy as np
import spacemap
import tqdm

def merge_layer_final2raw(rawDF, pointsDF, start, end, outKeys=None, convert=None):
    dfs = []
    if convert is not None:
        rawDF["layer"] = rawDF.apply(lambda x: convert(x))
        
    for i in range(start, end+1):
        raw = rawDF[rawDF['layer'] == i].copy()
        points = pointsDF[pointsDF['layer'] == i].copy()
        points = points[["x", "y"]].copy()
        points.columns = ["x_align", "y_align"]
        points.reset_index(drop=True, inplace=True)
        raw.reset_index(drop=True, inplace=True)
        df = pd.concat([raw, points], axis=1)
        dfs.append(df)
    df = pd.concat(dfs)
    if outKeys is not None:
        df = df[outKeys]
    return df

def merge_slices_df(start, end, columns, path):
    dfs = []
    for key in columns:
        for index in range(start, end+1):
            TARGET_S = spacemap.Slice(index, spacemap.BASE)
            dff = TARGET_S.get_df(key)
            dff["layer"] = index
            dff["step"] = key
            dfs.append(dff)
    df = pd.concat(dfs)
    if "/" not in path:
        path = "%s/outputs/%s.csv.gz" % (spacemap.BASE, path)
    df.to_csv(path, index=False)

def merge_columns(df1, df2, start, end, keys1, keys2):
    dfs = []
    for i in range(start, end+1):
        df1_ = df1[df1["layer"] == i].copy()
        df2_ = df2[df2["layer"] == i].copy()
        df1_ = df1_[keys1]
        df2_ = df2_[keys2]
        df = pd.concat([df1_, df2_], axis=1)
        dfs.append(df)
    return pd.concat(dfs)

import nibabel as nib
 
def outputs_nii(data: np.array, path):
    affine = np.eye(4)
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, path)
    