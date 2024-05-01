import numpy as np
import spacemap
import pandas as pd

class AutoAffineImgKey(spacemap.AffineFlowMgrImg):
    def __init__(self, imgI: np.array, imgJ: np.array, finder=None, show=False, method="sift_vgg"):
        super().__init__("AutoAffineImgKey", imgI, imgJ, finder)
        self.show=show
        self.method = method
        
    def run(self):
        self.run_flow(spacemap.affine_block.MatchInitImg(matchr=0.75, method=self.method))
        if len(self.matches) > 5:
            self.run_flow(spacemap.affine_block.FilterGraphImg(std=2))
            self.run_flow(spacemap.affine_block.FilterGlobalImg())
            self.run_flow(spacemap.affine_block.MatchEachImg())
        else:
            grad = spacemap.affine_block.AutoGradImg()
            self.run_flow(grad)
        if self.show:
            self.run_flow(spacemap.affine_block.ImgDiffShow())
        H = self.resultH_img()
        return H
    