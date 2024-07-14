import numpy as np
import pandas as pd
import spacemap
from spacemap import Slice
import matplotlib.pyplot as plt

def applyH_np(df: np.array, H) -> np.array:
    df2 = df.copy()
    H = np.array(H)
    df2[:, 0] = (df[:, 0] * H[0, 0] + df[:, 1] * H[0, 1]) + H[0, 2]
    df2[:, 1] = (df[:, 0] * H[1, 0] + df[:, 1] * H[1, 1]) + H[1, 2]
    return df2
