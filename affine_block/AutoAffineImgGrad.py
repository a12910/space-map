import numpy as np
import spacemap
import pandas as pd

class AutoAffineImgGrad(spacemap.AffineFlowMgrImg):
    def __init__(self, imgI: np.array, imgJ: np.array, finder=None, show=False):
        super().__init__("AutoAffineImgGrad", imgI, imgJ, finder)
        self.show=show
        self.step1Err = 0.1
        
    def run(self):
        grad1 = spacemap.affine_block.AutoGradImg()
        grad1.finalErr = self.step1Err
        _ = self.run_flow(grad1)
        grad2 = spacemap.affine_block.AutoGradImg2()
        _ = self.run_flow(grad2)
        if self.show:
            _ = self.run_flow(spacemap.affine_block.ImgDiffShow())
        H = self.resultH_img()
        return H
    