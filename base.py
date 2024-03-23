import os, sys, logging, time, json
import numpy as np

L = logging.Logger("3DCell")
L.setLevel(logging.INFO)

stdHandle = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter('[%(asctime)s]%(levelname)s: %(message)s')
stdHandle.setFormatter(formatter)
L.addHandler(stdHandle)

XYRANGE = [0, 4000, 0, 4000]
XYD = 10
IMGCONF = {"raw": 1}
BASE = "data/flow"
APPEND = np.array([0, 0])

DEVICE = "cpu"

GLOBAL_STORAGE = {}

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
# for item in ["data"]:
#     mkdir(item)

def get_logger():
    global L
    return L

def Error(content):
    global L
    L.error(content)

def Info(content):
    global L
    L.info(content)

CMAP = "twilight"
