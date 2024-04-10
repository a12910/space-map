import numpy as np
import spacemap

class MoveCenter(spacemap.AffineBlock):
    def __init__(self) -> None:
        super().__init__("MoveCenter")
    
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        meanIX, meanIY = np.mean(dfI["x"]), np.mean(dfI["y"])
        meanJX, meanJY = np.mean(dfJ["x"]), np.mean(dfJ["y"])
        H11 = np.array([[1, 0, -meanJX], [0, 1, -meanJY], [0, 0, 1]])
        H13 = np.array([[1, 0, meanIX], [0, 1, meanIY], [0, 0, 1]])
        H1 = np.dot(H13, H11)
        return H1
    