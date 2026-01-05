import os, sys, logging, time, json
import numpy as np
import space_map

def auto_init_xyd(xy: np.array):
    xy = np.array(xy)[:, :2]
    minn = (np.min(xy, axis=0)) - 500
    minn = (minn // 100) * 100
    maxx = np.max(xy, axis=0) + 500
    maxx = (maxx // 100) * 100 + minn
    space_map.APPEND = -minn
    space_map.XYRANGE = maxx[0]
    space_map.XYD = int(max(maxx) / 400)
    
def init_xy(xyr, xyd):
    space_map.XYRANGE = xyr
    space_map.XYD = xyd

def init_path(path):
    space_map.BASE = path
    for f in ["imgs", "outputs", "raw", "logs"]:
        space_map.mkdir(path + "/" + f)
    __reload_handler()

def storage_variables():
    space_map.GLOBAL_STORAGE = {"XYD": space_map.XYD, 
                               "IMGCONF": space_map.IMGCONF}
    
def revert_variables():
    space_map.XYD = space_map.GLOBAL_STORAGE.get("XYD", space_map.XYD)
    space_map.IMGCONF = space_map.GLOBAL_STORAGE.get("IMGCONF", space_map.IMGCONF)
    
def __reload_handler():
    L = space_map.L
    for handler in L.handlers:
        if isinstance(handler, logging.FileHandler):
            L.removeHandler(handler)
    path = space_map.BASE + "/logs"
    logPath = path + "/%s.log" % time.strftime("%Y%m%d-%H%M%S")
    fileHandle = logging.FileHandler(logPath)
    formatter = logging.Formatter('[%(asctime)s]%(levelname)s: %(message)s')
    fileHandle.setFormatter(formatter)
    L.addHandler(fileHandle)
    