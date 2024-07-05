import spacemap
import numpy as np
from PIL import Image
import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os

def generate_imgs_basic(df, start, end, outFolder, prefix="s"):
    spacemap.mkdir(outFolder)
    for i in tqdm.trange(start, end+1):
        xy = df[df["layer"] == i][["x", "y"]].values
        img = spacemap.show_img3(np.array(xy))
        path = "%s/%s_%d.tiff" % (outFolder, prefix, i)
        ii = Image.fromarray(img)
        ii.save(path)

def generate_grid(rawDF, alignDF, start, end, outFolder, 
                          prefix="grid", multiprocess=True):
    """ 为每一层生成一个grid """
    spacemap.mkdir(outFolder)
    spacemap.Info("Generate grid start %d~%d ->  %s/%s" % (start, end, outFolder, prefix))
    shape = spacemap.XYRANGE[1], spacemap.XYRANGE[3]
    xyd = spacemap.XYD
    gridShape = (int(shape[0]/xyd), int(shape[1]/xyd))
    
    data = []
    for i in range(start, end+1):
        raw = rawDF[rawDF["layer"] == i][["x", "y"]].values
        align = alignDF[alignDF["layer"] == i][["x", "y"]].values
        data.append((raw, align, xyd, gridShape, outFolder, "%s_%.2d" % (prefix, i)))
    cpu = os.cpu_count() if multiprocess else 1
    with Pool(cpu) as p:
        p.map(__generate_grid, data)
    spacemap.Info("Generate grid finish")
        
def __generate_grid(pack):
    raw, align, xyd, gridShape, outFolder, prefix = pack
    spacemap.Info("Generate grid for layer %s" % prefix)
    grid = spacemap.grid.GridGenerate(gridShape, xyd, 1)
    grid.useTQDM = False
    grid.init_db(raw, align)
    grid.generate()
    grid.fix()
    path = "%s/%s.npy" % (outFolder, prefix)
    np.save(path, grid.grid)

def compute_err_from_folder(baseF, start, end, filename, err=None):
    if err is None:
        err = spacemap.find.default()
    err_result = np.zeros(end-start)
    imgs = {}
    for i in range(start, end+1):
        file = filename.replace("NUM", str(i))
        img = plt.imread(baseF + "/" + file)
        imgs[i] = img
    
    for i in range(start, end):
        img1 = imgs[i]
        img2 = imgs[i+1]
        err_result[i-start] = err.computeI(img1, img2, show=False)
    return err_result, err_result.mean()


import torch
import torch.nn.functional as F

def grid_sample_img(img_, grid_, maxx=None, targetSize=None, exchange=True, move=None):
    # img: HxWx3, grid: HxWx2
    # use torch grid_sample to sample img with grid
    grid = grid_.copy()
    if maxx is not None:
        grid = grid / maxx * 2 - 1
    
    if exchange:
        grid2 = grid.copy()
        grid[:, :, 0] = grid2[:, :, 1]
        grid[:, :, 1] = grid2[:, :, 0]
    
    img = img_.copy()
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=2)
        
    img = img.astype(np.float64)
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    # grid 插值到与img相同尺寸
    if targetSize is None:
        targetSize = img_.shape[:2]
        
    grid = torch.tensor(grid).permute(2, 0, 1).unsqueeze(0)
    grid = F.interpolate(grid, targetSize, mode='bilinear', align_corners=True)
    grid = grid.permute(0, 2, 3, 1)
    
    img2 = torch.nn.functional.grid_sample(img, grid, align_corners=True)
    img2 = img2.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    if len(img_.shape) == 2:
        img2 = img2[:, :, 0]
    if move is not None:
        x, y = move
        img2_ = np.zeros_like(img2)
        img2_ = spacemap.he_img.cut_img(img2, x, y, 
                                        img2_.shape[1]+x, img2_.shape[0]+y)
        img2 = img2_
    return img2


