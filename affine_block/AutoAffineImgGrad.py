import numpy as np
import spacemap
import pandas as pd

class AutoAffineImgGrad(spacemap.AffineFlowMgrImg):
    def __init__(self, imgI: np.array, imgJ: np.array, finder=None, show=False):
        super().__init__("AutoAffineImgGrad", imgI, imgJ, finder)
        self.show=show
        
    def run(self):
        grad = spacemap.affine_block.AutoGradImg()
        _ = self.run_flow(grad)
        if self.show:
            _ = self.run_flow(spacemap.affine_block.ImgDiffShow())
        H = self.resultH_img()
        return H
    