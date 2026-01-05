import numpy as np
import space_map
import pandas as pd
from .flowMgrImg import AffineFlowMgrImg

class AutoAffineImgGrad(AffineFlowMgrImg):
    def __init__(self, imgI: np.array, imgJ: np.array, finder=None, show=False):
        super().__init__("AutoAffineImgGrad", imgI, imgJ, finder)
        self.show=show
        self.step1Err = 0.1
        
    def run(self):
        grad1 = space_map.affine_block.AutoGradImg()
        grad1.finalErr = self.step1Err
        _ = self.run_flow(grad1)
        grad2 = space_map.affine_block.AutoGradImg2()
        _ = self.run_flow(grad2)
        if self.show:
            _ = self.run_flow(space_map.affine_block.ImgDiffShow())
        H = self.resultH_img()
        return H
    