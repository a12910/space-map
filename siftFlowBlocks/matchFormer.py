"""
import matplotlib.pyplot as plt
import torch
from config.defaultmf import get_cfg_defaults

imgI = plt.imread("data/raw_22.png")[:, :, 0]
imgJ = plt.imread("data/raw_23.png")[:, :, 0]
imgI_ = imgI.reshape((1, 1, *imgI.shape))
imgJ_ = imgJ.reshape((1, 1, *imgJ.shape))

batch = {
    "image0": torch.tensor(imgI_), 
    "image1": torch.tensor(imgJ_)
}

from model.lightning_loftr import PL_LoFTR
from model.matchformer import Matchformer
from model.utils.misc import lower_config, flattenList
config = get_cfg_defaults()
# model = PL_LoFTR(config, pretrained_ckpt="model/weights/outdoor-large-LA.ckpt")
config_ = lower_config(config)
model = Matchformer(config_["matchformer"])

ckpt = "model/weights/outdoor-large-LA.ckpt"

state = torch.load(ckpt, map_location='cpu')
state = {k.replace('matcher.',''):v for k,v in state.items()}
model.load_state_dict(state)
model.eval()
model(batch)
# points: batch["mkpts0_c"] / mkpts1_c / f
# confi: batch["mconf"]

"""