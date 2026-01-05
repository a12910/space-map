import numpy as np
import space_map
import pandas as pd
from .flowMgrImg import AffineFlowMgrImg


class AlignMethod:
    auto = "auto"
    sift_only = "sift_only"
    auto_old = "auto_old"
    cpd = "cpd"

class AutoAffineImgKey(AffineFlowMgrImg):
    useLDM = False
    useDetail = True
    useGrad = False
    
    @staticmethod
    def restart():
        space_map.affine_block.MATCH_EACH_IMG.init = True
        space_map.affine_block.MATCH_EACH_IMG.clear()
    
    def __init__(self, imgI: np.array, imgJ: np.array, finder=None, show=False, method="auto"):
        super().__init__("AutoAffineImgKey", imgI, imgJ, finder)
        self.show=show
        self.method = method
        self.step1Err = 0.1
        self.processinit = False
        self.each = space_map.affine_block.MATCH_EACH_IMG
        
    def run(self, df=None):
        if self.processinit:
            self.imgI, self.imgJ = AutoAffineImgKey.process_init(self.imgI, self.imgJ)
        if self.method is None:
            self.method = "auto"
        if self.method == "auto":
            match = space_map.affine_block.MatchInitAuto(df)
            _ = self.run_flow(match)
        elif self.method == "sift_only":
            _ = self.run_flow(space_map.affine_block.MatchInitImg(matchr=0.75, method="sift_vgg"))
        elif self.method == "auto_old":
            _ = self.run_flow(space_map.affine_block.MatchInitImg(matchr=0.75, method="sift_vgg"))
            _ = self.run_flow(space_map.affine_block.FilterLPMImg())
            if len(self.matches) < 50:
                _ = self.run_flow(space_map.affine_block.MatchInitImg(matchr=0.75, method="loftr"))
                if len(self.matches) < 10:
                    self.method = ""
        elif self.method == "cpd":
            f = space_map.affine_block.CPDAffine()
            _ = self.run_flow(f)
            if self.useDetail:
                grad2 = space_map.affine_block.AutoGradImg2()
                _ = self.run_flow(grad2)
            if self.show:
                self.run_flow(space_map.affine_block.ImgDiffShow(channel=True))
            H = self.resultH_img()
            return H
        elif self.method != "" and self.method != "only_grad":
            self.run_flow(space_map.affine_block.MatchInitImg(matchr=0.75, method=self.method))
        if len(self.matches) > 5 and self.method != "only_grad":
            # self.run_flow(spacemap.affine_block.FilterGraphImg(std=2))
            self.run_flow(space_map.affine_block.FilterGlobalImg())
            self.run_flow(space_map.affine_block.FilterLPMImg())
            self.run_flow(self.each)
        if self.useGrad:
            grad1 = space_map.affine_block.AutoGradImg()
            grad1.finalErr = self.step1Err
            _ = self.run_flow(grad1)
        if self.useLDM == True:
            self.useLDM = 3
        oldXYD = space_map.XYD
        for i in range(self.useLDM):
            space_map.XYD = oldXYD // (2**i)
            space_map.Info("Affine LDM %d xyd=%d" % (i+1, space_map.XYD))
            ldm = space_map.affine_block.LDMAffine()
            ldm.err = self.step1Err
            _ = self.run_flow(ldm)
        space_map.XYD = oldXYD
        # detail = spacemap.affine_block.DetailAffine()
        # _ = self.run_flow(detail)
        if self.useDetail:
            grad2 = space_map.affine_block.AutoGradImg2()
            _ = self.run_flow(grad2)
        if self.show:
            self.run_flow(space_map.affine_block.ImgDiffShow(channel=True))
        H = self.resultH_img()
        return H
    
    @staticmethod
    def process_init(arr1, arr2):
        def downscale(arr):
            N = arr.shape[0]
            if N % 2 != 0:
                N -= 1
            arr = arr[:N, :N]
            return arr.reshape(N//2, 2, N//2, 2).mean(axis=(1, 3))

        def non_zero_mean(arr):
            non_zero_elements = arr[arr != 0]
            if len(non_zero_elements) == 0:
                return 0
            return non_zero_elements.mean()
        
        small_arr1 = downscale(arr1)
        small_arr2 = downscale(arr2)
        
        mean1 = non_zero_mean(small_arr1)
        mean2 = non_zero_mean(small_arr2)
        arr1 = arr1.astype(np.float32)
        arr2 = arr2.astype(np.float32)
        
        # 调整平均值
        if mean1 < mean2 and mean1 != 0:
            adjustment_factor = mean2 / mean1
            arr1 *= adjustment_factor
        elif mean2 < mean1 and mean2 != 0:
            adjustment_factor = mean1 / mean2
            arr2 *= adjustment_factor

        arr1[arr1 < 0] = 0
        arr2[arr2 < 0] = 0
        arr1 = arr1.astype(np.uint8)
        arr2 = arr2.astype(np.uint8)
        
        return arr1, arr2
    