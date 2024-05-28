import numpy as np
import spacemap
import pandas as pd

class AutoAffineImgKey(spacemap.AffineFlowMgrImg):
    def __init__(self, imgI: np.array, imgJ: np.array, finder=None, show=False, method="sift_vgg"):
        super().__init__("AutoAffineImgKey", imgI, imgJ, finder)
        self.show=show
        self.method = method
        self.step1Err = 0.1
        
    def run(self):
        if self.method is None:
            self.method = "sift_vgg"
        if self.method != "" and self.method != "only_grad":
            self.run_flow(spacemap.affine_block.MatchInitImg(matchr=0.75, method=self.method))
        if len(self.matches) > 5 and self.method != "only_grad":
            self.run_flow(spacemap.affine_block.FilterGraphImg(std=2))
            self.run_flow(spacemap.affine_block.FilterGlobalImg())
            self.run_flow(spacemap.affine_block.MatchEachImg())
        else:
            grad1 = spacemap.affine_block.AutoGradImg()
            grad1.finalErr = self.step1Err
            _ = self.run_flow(grad1)
        grad2 = spacemap.affine_block.AutoGradImg2()
        _ = self.run_flow(grad2)
        if self.show:
            self.run_flow(spacemap.affine_block.ImgDiffShow(channel=True))
        H = self.resultH_img()
        return H
    