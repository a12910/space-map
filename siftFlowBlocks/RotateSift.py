import numpy as np
import cv2
import spacemap

class BestRotate(spacemap.AffineBlockBestRotate):
    def __init__(self) -> None:
        super().__init__("BestRotate")
        self.rotate = 0
        
    def find_rotate(self, imgI, imgJ, finder: spacemap.AffineFinder):
        w, h = imgI.shape[:2]
        imgI_ = np.zeros((w * 2, h * 2))
        imgJ_ = np.zeros((w * 2, h * 2))
        imgI_[int(w // 2): int(w // 2 + w), int(h // 2): int(h // 2 + h)] = imgI
        imgJ_[int(w // 2): int(w // 2 + w), int(h // 2): int(h // 2 + h)] = imgJ
        for rotate in range(0, 360, self.step1):
            imgJ1 = spacemap.compute.rotate_img(imgJ_, rotate)
            finder.add_result(rotate, None, imgI_, imgJ1)
        min_rotate = finder.bestH()
        finder.clear()
        for rotate in range(min_rotate - self.step2, 
                            min_rotate + self.step2):
            imgJ1 = spacemap.compute.rotate_img(imgJ_, rotate)
            finder.add_result(rotate, None, imgI_, imgJ1)
        min_rotate = finder.bestH()
        return finder.bestH()
    