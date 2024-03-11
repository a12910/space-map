import numpy as np
import cv2
import spacemap

class AffineBlockMoveCenter(spacemap.AffineBlock):
    def __init__(self) -> None:
        super().__init__("MoveCenter")
    
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        meanIX, meanIY = np.mean(dfI["x"]), np.mean(dfI["y"])
        meanJX, meanJY = np.mean(dfJ["x"]), np.mean(dfJ["y"])
        H11 = np.array([[1, 0, -meanJX], [0, 1, -meanJY], [0, 0, 1]])
        H13 = np.array([[1, 0, meanIX], [0, 1, meanIY], [0, 0, 1]])
        H1 = np.dot(H13, H11)
        return H1
    
class AffineBlockBestRotateSift(spacemap.AffineBlockBestRotate):
    def __init__(self) -> None:
        super().__init__("BestRotateSift")
        self.rotate = 0
        
    def find_rotate(self, imgI, imgJ, finder: spacemap.AffineFinder):
        finder = spacemap.AffineBlockBestRotateSift()
        w, h = imgI.shape[:2]
        imgI_ = np.zeros((w * 2, h * 2))
        imgJ_ = np.zeros((w * 2, h * 2))
        imgI_[int(w // 2): int(w // 2 + w), int(h // 2): int(h // 2 + h)] = imgI
        imgJ_[int(w // 2): int(w // 2 + w), int(h // 2): int(h // 2 + h)] = imgJ
        for rotate in range(0, 360, self.step1):
            imgJ1 = spacemap.AffineBlockRotate.rotate_img(imgJ_, rotate)
            finder.add_result(rotate, None, imgI_, imgJ1)
        min_rotate = finder.bestH()
        finder.clear()
        for rotate in range(min_rotate - self.step2, 
                            min_rotate + self.step2):
            imgJ1 = spacemap.AffineBlockRotate.rotate_img(imgJ_, rotate)
            finder.add_result(rotate, None, imgI_, imgJ1)
        return finder.bestH()
    