import os, sys, logging, time, json
import numpy as np

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
for item in ["data", "data/logs"]:
    mkdir(item)

L = logging.Logger("3DCell")
L.setLevel(logging.INFO)

stdHandle = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter('[%(asctime)s]%(levelname)s: %(message)s')
stdHandle.setFormatter(formatter)
L.addHandler(stdHandle)

# add handler to file
logPath = "data/logs/%s.log" % time.strftime("%Y%m%d-%H%M%S")
fileHandle = logging.FileHandler(logPath)
fileHandle.setFormatter(formatter)
L.addHandler(fileHandle)

XYRANGE = 4000
XYD = 10
IMGCONF = {"raw": 1}
BASE = "data/flow"
APPEND = np.array([0, 0])

DEVICE = "cpu"

import torch
if torch.cuda.is_available():
    DEVICE = "cuda:0"

LAYER_START = 0
LAYER_END = 0

GLOBAL_STORAGE = {}

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

