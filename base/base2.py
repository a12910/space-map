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
    spacemap.XYRANGE = maxx[0]
    spacemap.XYD = int(max(maxx) / 400)
    
def init_xy(xyr, xyd):
    spacemap.XYRANGE = xyr
    spacemap.XYD = xyd

def init_path(path):
    spacemap.BASE = path
    for f in ["imgs", "outputs", "raw"]:
        spacemap.mkdir(path + "/" + f)

def storage_variables():
    spacemap.GLOBAL_STORAGE = {"XYD": spacemap.XYD, 
                               "IMGCONF": spacemap.IMGCONF}
    
def revert_variables():
    spacemap.XYD = spacemap.GLOBAL_STORAGE.get("XYD", spacemap.XYD)
    spacemap.IMGCONF = spacemap.GLOBAL_STORAGE.get("IMGCONF", spacemap.IMGCONF)
    