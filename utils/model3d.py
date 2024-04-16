import spacemap
import numpy as np
from PIL import Image
import tqdm

def generate_imgs_basic(df, start, end, outFolder, prefix="s"):
    spacemap.mkdir(outFolder)
    for i in tqdm.trange(start, end+1):
        xy = df[df["layer"] == i][["x", "y"]].values
        img = spacemap.show_img3(np.array(xy))
        path = "%s/%s_%d.tiff" % (outFolder, prefix, i)
        ii = Image.fromarray(img)
        ii.save(path)

def generate_grid(rawDF, alignDF, start, end, outFolder, 
                          prefix="grid"):
    """ 为每一层生成一个grid """
    spacemap.mkdir(outFolder)
    shape = spacemap.XYRANGE[1], spacemap.XYRANGE[3]
    xyd = spacemap.XYD
    gridShape = (int(shape[0]/xyd), int(shape[1]/xyd))
    for i in range(start, end+1):
        spacemap.Info("Generate grid for layer %d" % i)
        raw = rawDF[rawDF["layer"] == i][["x", "y"]].values
        align = alignDF[alignDF["layer"] == i][["x", "y"]].values
        grid = spacemap.grid.GridGenerate(gridShape, xyd, 1)
        grid.init_db(raw, align)
        grid.generate()
        grid.fix()
        path = "%s/%s_%.2d.npy" % (outFolder, prefix, i)
        np.save(path, grid)
