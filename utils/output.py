import pandas as pd
import numpy as np

def outputs_merge_layer(rawDF, pointsDF, start, end, outKeys=None, convert=None):
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

