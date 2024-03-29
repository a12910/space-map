import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scanpy as sc
import paste as pst
import pandas as pd
import os, shutil
import spacemap

TEMP = "../data/flow/outputs/temp.csv"

def generate_slice(img, distance=10, kernel=5):
    """ img: shape*shape*channel """
    points = []
    shape = (400, 400)
    cells = 20
    for x in range(0, shape[0], distance):
        for y in range(0, shape[1], distance):
            base = [x+kernel, y+kernel]
            part = img[x:x+kernel*2+1, y:y+kernel*2+1, :]
            part_sum = np.sum(part, axis=(0, 1))
            part2 = img[x-kernel*2:x+kernel*4+1, y-kernel*2:y+kernel*4+1, :]
            part2_sum = np.sum(part2, axis=(0, 1))
            if part2_sum[0] < 1:
                continue            
            base += list(part_sum[1:cells+1])
            points.append(base)
    points = np.array(points)
    df = pd.DataFrame(points[:, 2:], columns=["c"+str(i+1) for i in range(cells)])
    df.to_csv(TEMP, index=False)
    slice = sc.read_csv(TEMP)
    coor = points[:, :2]
    slice.obsm["spatial"] = coor
    sc.pp.filter_genes(slice, min_counts = 0)
    sc.pp.filter_cells(slice, min_counts = 0)
    os.remove(TEMP)
    return slice, coor

def try_paste(img1, img2, show_spot=True):
    slice1, coor1 = generate_slice(img1)
    slice2, coor2 = generate_slice(img2)
    pi12 = pst.pairwise_align(slice1, slice2)

    slices, pis = [slice1, slice2], [pi12]
    new_slices = pst.stack_slices_pairwise(slices, pis)

    if show_spot:
        slice_colors = ['#e41a1c','#377eb8']
        plt.figure(figsize=(7,7))
        for i in range(len(new_slices)):
            pst.plot_slice(new_slices[i],slice_colors[i],s=40)
        plt.legend(handles=[mpatches.Patch(color=slice_colors[0], label='1'),mpatches.Patch(color=slice_colors[1], label='2')])
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.show()
    
    grid = np.zeros((2, 40, 40))
    coor2_ = np.dot(pi12.T, coor1)
    pi12_sum = pi12.sum(axis=0)
    for i in range(coor2_.shape[0]):
        coor2_[i] = coor2_[i] / pi12_sum[i]
        
    grid = generate_grid(pi12, coor1, coor2, distance)
    img1 = compute_img(index1, distance)
    img2 = compute_img(index2, distance, grid)
    plt.imshow(img1)
    plt.show()
    plt.imshow(img2)
    plt.show()
        
def generate_grid(pi12, coor1, coor2, distance):
    grid = np.zeros((2, int(400 // distance), int(400 // distance)))
    coor2_ = np.dot(pi12.T, coor1)
    pi12_sum = pi12.sum(axis=0)
    for i in range(coor2_.shape[0]):
        coor2_[i] = coor2_[i] / pi12_sum[i]

    for i in range(len(coor2)):
        xx, yy = coor2[i]
        xi, yi = int(xx // distance), int(yy // distance)
        nxx, nyy = coor2_[i]
        nxi, nyi = nxx / 200 - 1, nyy / 200 - 1
        grid[1, xi, yi] = nxi
        grid[0, xi, yi] = nyi
    return grid

def compute_img(points, distance, grid=None):
    spacemap.XYD = distance * 10
    if grid is not None:
        points = spacemap.applyPointsByGrid(grid, points)
    spacemap.XYD = 10
    img = spacemap.show_img3(points)
    return img
    
    

"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scanpy as sc
import paste as pst
import sys
sys.path.append("../")
import spacemap

import pandas as pd
data = np.load("../data/flow/raw/raw_cells400t10.npy")
df = pd.read_csv("../data/flow/raw/cells.csv.gz")
data.shape

def generate_slice(index, distance):
    points = []
    shape = (400, 400)
    cells = 20
    d = data[index-3]
    kernel = int(distance // 2)
    for x in range(0, shape[0], distance):
        for y in range(0, shape[1], distance):
            base = [x+kernel, y+kernel]
            part = d[x:x+kernel*2+1, y:y+kernel*2+1, :]
            part_sum = np.sum(part, axis=(0, 1))
            part2 = d[x-kernel*2:x+kernel*4+1, y-kernel*2:y+kernel*4+1, :]
            part2_sum = np.sum(part2, axis=(0, 1))
            if part2_sum[0] < 1:
                continue            
            base += list(part_sum[1:cells+1])
            points.append(base)
    points = np.array(points)
    df = pd.DataFrame(points[:, 2:], columns=["c"+str(i+1) for i in range(cells)])
    df.to_csv("../data/flow/outputs/temp.csv", index=False)
    slice = sc.read_csv("../data/flow/outputs/temp.csv")
    coor = points[:, :2]
    slice.obsm["spatial"] = coor
    sc.pp.filter_genes(slice, min_counts = 0)
    sc.pp.filter_cells(slice, min_counts = 0)
    return slice, coor

def generate_grid(pi12, coor1, coor2, distance):
    grid = np.zeros((2, int(400 // distance), int(400 // distance)))
    coor2_ = np.dot(pi12.T, coor1)
    pi12_sum = pi12.sum(axis=0)
    for i in range(coor2_.shape[0]):
        coor2_[i] = coor2_[i] / pi12_sum[i]

    for i in range(len(coor2)):
        xx, yy = coor2[i]
        xi, yi = int(xx // distance), int(yy // distance)
        nxx, nyy = coor2_[i]
        nxi, nyi = nxx / 200 - 1, nyy / 200 - 1
        grid[1, xi, yi] = nxi
        grid[0, xi, yi] = nyi
    return grid

def compute_img(index, distance, grid=None):
    spacemap.XYD = distance * 10
    points = np.array(df[df["layer"] == index][["x", "y"]].values)
    if grid is not None:
        points = spacemap.applyPointsByGrid(grid, points)
    spacemap.XYD = 10
    img = spacemap.show_img3(points)
    return img

distance = 10
index1 = 3
index2 = 4

result = np.zeros((17, 3))
err = spacemap.AffineFinderBasic("dice")
for index in range(3, 20):
    for d_ in range(3):
        distance = [5, 10, 20][d_]
        print(index, distance)
        index1 = index
        index2 = index+1
        slice1, coor1 = generate_slice(index1, distance)
        slice2, coor2 = generate_slice(index2, distance) 
        pi12 = pst.pairwise_align(slice1, slice2, use_gpu=True)
        grid = generate_grid(pi12, coor1, coor2, distance)
        img1 = compute_img(index1, distance)
        img2 = compute_img(index2, distance, grid)
        e = err.err(img1, img2)
        result[index-3, d_] = e
plt.imshow(result)
plt.show()
dff = pd.DataFrame(result, columns=["paste-5", "paste-10", "paste-20"])
dff.to_csv("../data/flow/values/paste_test.csv", index=False)

import pandas as pd
dff = pd.read_csv("../data/flow/values/paste_test.csv")
df2 = pd.read_csv("../data/flow/values/compare_dice.csv")
dff.shape

dff["enhance"] = df2["enhance"][:17]

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] ='sans-serif'
import seaborn as sns
import matplotlib.pyplot as plt
dff = -dff
data = dff[["enhance", "paste-5", "paste-10", "paste-20"]].values
fig, ax = plt.subplots()
sns.boxplot(data)
ax.set_xticklabels(["本文算法", "PASTE-80", "PASTE-40", "PASTE-20"])
ax.set_ylim(0.2, 0.8)
plt.show()
"""