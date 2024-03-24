import spacemap
import numpy as np
from PIL import Image

def model3d_imaris_imgs1(df, start, end, outFolder, prefix="s"):
    spacemap.mkdir(outFolder)
    for i in range(start, end+1):
        xy = df[df["layer"] == i][["x", "y"]].values
        img = spacemap.show_img3(np.array(xy))
        path = "%s/%s_%d.tiff" % (outFolder, prefix, i)
        ii = Image.fromarray(img)
        ii.save(path)
