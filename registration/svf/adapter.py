from ..base import Registration
from .svf_lddmm import GlobalLDDMM_Register
from .hull_alignment import align_stack_perfect_shape
import torch

class SVFLDDMM(Registration):
    def __init__(self, device=None):
        super().__init__("SVFLDDMM")
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.imgFix = None
        self.imgMov = None
        self.mgr = GlobalLDDMM_Register(device=self.device)
        self.grid_size =(16, 16)

    def load_img(self, imgI, imgJ):
        self.imgFix = imgJ
        self.imgMov = imgI

    def run(self):
        # img, flow = self.mgr.run(self.imgMov, self.imgFix)
        img, _ = self.mgr.run_high_res_loss_low_res_grid(self.imgMov, self.imgFix, grid_size=self.grid_size)
        return img

    def apply_points2d(self, points, xyd, flow=None):
        if flow is None:
            flow = self.mgr.flow
        return self.mgr.map_fix_points_to_moving_space(points, flow, xyd)

    def run_global(self, imgs):
        return align_stack_perfect_shape(imgs)

