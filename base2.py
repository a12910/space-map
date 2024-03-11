import os, sys, logging, time, json
import numpy as np
import spacemap

def auto_init_xyd(xy: np.array):
    xy = np.array(xy)[:, :2]
    minn = (np.min(xy, axis=0)) - 500
    minn = (minn // 100) * 100
    maxx = np.max(xy, axis=0) + 500
    maxx = (maxx // 100) * 100 + minn
    spacemap.APPEND = -minn
    spacemap.XYRANGE = [0, maxx[0], 0, maxx[1]]
    spacemap.XYD = int(max(maxx) / 400)
    
def init_xy(xyr, xyd):
    spacemap.XYRANGE = xyr
    spacemap.XYD = xyd

def init_path(path):
    spacemap.BASE = path
    for f in ["imgs", "outputs", "raw"]:
        spacemap.mkdir(path + "/" + f)

