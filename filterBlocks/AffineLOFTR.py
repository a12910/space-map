
from kornia.feature import LoFTR
import torch
import spacemap
import numpy as np

class AffineAlignmentLOFTR(spacemap.AffineAlignment):
    def __init__(self):
        super().__init__()
        
    def compute(self, imgI, imgJ, matchr):
        return loftr_compute_matches(imgI, imgJ, matchr)

def loftr_compute_matches(imgI, imgJ, matchr, device=None):
    if device is None:
        device = spacemap.DEVICE
    imgI = np.array(imgI)
    imgJ = np.array(imgJ)
    minn, maxx = imgI.min(), imgI.max()
    imgI = (imgI - minn) / (maxx - minn)
    imgJ = (imgJ - minn) / (maxx - minn)
    imgI = np.array(imgI, dtype=np.float32)
    imgJ = np.array(imgJ, dtype=np.float32)
    
    if len(imgI.shape) == 2:
        imgI = imgI.reshape((1, *imgI.shape))
        imgJ = imgJ.reshape((1, *imgJ.shape))
    imgI = imgI.reshape((1, *imgI.shape))
    imgJ = imgJ.reshape((1, *imgJ.shape))

    batch = {
        "image0": torch.tensor(imgI, device=device), 
        "image1": torch.tensor(imgJ, device=device)
    }

    matcher = LoFTR("indoor").to(device)
    out = matcher(batch)
    pts1 = out["keypoints0"].cpu().numpy()
    pts2 = out["keypoints1"].cpu().numpy()
    
    pts1_ = np.zeros_like(pts1)
    pts1_[:, 0] = pts1[:, 1]
    pts1_[:, 1] = pts1[:, 0]
    pts2_ = np.zeros_like(pts2)
    pts2_[:, 0] = pts2[:, 1]
    pts2_[:, 1] = pts2[:, 0]
    
    confidence = out["confidence"].cpu().numpy()
    result = np.concatenate((pts1_, pts2_, confidence.reshape(-1, 1)), axis=1)
    lis = list(result)
    lis.sort(key=lambda x: x[-1], reverse=True)
    result2 = []
    if matchr > 1:
        result2 = lis[:int(matchr)]
    else:
        for i in lis:
            if i[-1] > matchr:
                result2.append(i)
    return np.array(result2)

