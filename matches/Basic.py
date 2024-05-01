
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacemap
import cv2

class Basic:
    def __init__(self, method="sift"):
        """ sift surf brisk orb akaze vgg"""
        self.method = method
        self.multiChannel = False
        
    def compute_multi(self, imgI, imgJ, matchr=None):
        if len(imgI.shape) == 2:
            return self.compute(imgI, imgJ, matchr=matchr)
        matches = []
        for i in range(imgI.shape[2]):
            mat = self.compute(imgI[:, :, i], imgJ[:, :, i], matchr=matchr)
            matches += mat.tolist()
        mat = self.compute(imgI.mean(axis=2), imgJ.mean(axis=2), matchr=matchr)
        matches += mat.tolist()
        matches.sort(key=lambda x: x[-1])
        
        return np.array(matches)
    
    def compute(self, imgI, imgJ, matchr=None):
        if matchr is None:
            matchr = 0.75
        methods = self.method
        if isinstance(self.method, str):
            methods = [self.method]
        matches = []
        for method in methods:
            mat = spacemap.matches.siftImageAlignment(imgI, imgJ, matchr=matchr, method=method, scale=False)
            matches += mat.tolist()
        matches.sort(key=lambda x: x[-1], reverse=True)
        matches = np.array(matches) 
        return matches
            