import spacemap
from spacemap import Slice
import numpy as np

class LDMMgrMulti:
    def __init__(self, slices: list[Slice], 
                 initJKey=Slice.rawKey,
                 enhance=None, gpu=None, 
                 finalKey=Slice.finalKey,
                 enhanceKey=Slice.enhanceKey):
        self.slices = slices
        self.initJKey = initJKey
        self.enhance=enhance
        self.gpu=gpu
        self.finalKey = finalKey
        self.enhanceKey = enhanceKey
        self.err = spacemap.AffineFinderMultiDice(10)
        
    def start(self):
        pass
    