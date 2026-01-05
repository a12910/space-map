import numpy as np
from numpy.core.multiarray import array as array
import space_map

class FinalRotate(space_map.AffineBlock):
    def __init__(self) -> None:
        super().__init__("FinalRotate")
        self.kernel = 9
        self.step1 = 5
        self.step2 = 0.1
    
    def compute(self, dfI: np.array, dfJ: np.array, finder: space_map.AffineFinder):
        imgI = space_map.show_img(dfI)
        imgJ = space_map.show_img(dfJ)
        diceImg = space_map.err_conv_min(imgI, imgJ, kernel=self.kernel)
        xx, yy = np.argmax(diceImg)
        x_ = int(xx * space_map.XYD)
        y_ = int(yy * space_map.XYD)
        
        center_x, center_y = space_map.XYRANGE // 2, space_map.XYRANGE // 2
        H1 = np.array([[1, 0, center_x-x_], [0, 1, center_y-y_], [0, 0, 1]])
        dfI_ = space_map.points.applyH_np(dfI, H1, fromImgH=False)
        dfJ_ = space_map.points.applyH_np(dfJ, H1, fromImgH=False)
        imgI2 = space_map.show_img(dfI_)
        imgJ2 = space_map.show_img(dfJ_)
        
        finder.clear()
        for rotate_ in range(-int(self.kernel / self.step2), int(self.kernel / self.step2)):
            rotate = rotate_ * self.step2
            imgJ3 = space_map.compute.rotate_img(imgJ2, rotate)
            finder.add_result(rotate, None, imgI2, imgJ3)
        H2 = finder.bestH()
        H3 = np.array([[1, 0, x_-center_x], [0, 1, y_-center_y], [0, 0, 1]])
        H = np.dot(H3, np.dot(H2, H1))
        return H
        