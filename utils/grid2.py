from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import spacemap

from multiprocessing import Pool
import os

class GridRevert:
    def __init__(self, grid: np.array) -> None:
        self.shape = grid.shape[:2]
        grid = grid.copy()
        self.template = (grid + 1) * (self.shape[0] / 2)
        self.mesh = np.zeros_like(self.template)
        for i in range(self.shape[0]):
            self.mesh[:, i, 0] = i
            self.mesh[i, :, 1] = i
            # self.mesh[x, y] = [x, y]
        self.revert = np.zeros_like(self.mesh)
        self.db = {}
        self.useTQDM = True
        self.label = np.zeros(self.shape)
        
    def xy_to_tag(self, x, y):
        return "%d_%d" % (int(x), int(y))
        
    def init_db(self):
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                fx, fy = self.template[x, y]
                tx, ty = y, x
                tag = self.xy_to_tag(fx, fy)
                if tag not in self.db.keys():
                    self.db[tag] = []
                self.db[tag].append([fx, fy, tx, ty])
            
    def generate(self, kernel=1):
        for x in tqdm.trange(kernel, self.shape[0]-kernel, 
                             disable=not self.useTQDM):
            for y in range(kernel, self.shape[1]-kernel):
                lis = []
                for ix in range(-kernel, kernel+1):
                    for iy in range(-kernel, kernel+1):
                        tag = self.xy_to_tag((x+ix), (y+iy))
                        lis += self.db.get(tag, [])
                if len(lis) > 2:
                    ps = np.array(lis)
                    newp = self.fit_new_points(ps[:, :2], ps[:, 2:], [x, y], 1)
                    if np.min(newp) < 0 or np.max(newp) > self.shape[0]:
                        continue
                    self.revert[x, y, 0] = newp[0]
                    self.revert[x, y, 1] = newp[1]
                    self.label[x, y] = 1
                    
    def generate_thread(self, kernel=1):
        # kernel, xFrom, xTo, shape, db, grid, label = pack
        count = os.cpu_count()
        step = int(self.shape[0] // count + 1)
        
        datas = [(kernel, step*i, step*(i+1), self.shape[0], self.db, self.revert, self.label) for i in range(count)]
        result = []
        # for i in range(count):
        #     re = GridRevert.generate_thread_step(datas[i])
        #     result.append(re)
        with Pool(os.cpu_count()) as p:
            result = p.map(generate_thread_step, datas)
        for g, label in result:
            self.revert[label > 0] = g[label > 0]
            self.label += label
                    
    def export_grid(self):
        revert = self.revert.copy()
        revert = revert / self.shape[0] * 2 - 1
        revert = torch.Tensor(revert).permute(1, 0, 2).numpy()
        revert2 = revert.copy()
        revert2[:, :, 0] = revert[:, :, 1]
        revert2[:, :, 1] = revert[:, :, 0]
        return revert2
                    
    def fix(self, kernel2=2):
        label2 = np.zeros_like(self.label)
        for x in tqdm.trange(kernel2, self.shape[0]-kernel2,
                             disable=not self.useTQDM):
            for y in range(kernel2, self.shape[1]-kernel2):
                if self.label[x, y] > 0:
                    continue
                partl = self.label[x-kernel2:x+kernel2+1, y-kernel2:y+kernel2+1]
                if np.sum(partl) < 3:
                    continue
                part2 = self.revert[x-kernel2:x+kernel2+1, y-kernel2:y+kernel2+1, :]
                part_mesh = self.mesh[x-kernel2:x+kernel2+1, y-kernel2:y+kernel2+1, :]
                part3 = np.concatenate([part_mesh, part2], axis=2)
                ps = part3[partl > 0, :].reshape(-1, 4)
                newp = self.fit_new_points(ps[:, :2], ps[:, 2:], [x, y], 1)
                if np.min(newp) < 0 or np.max(newp) > self.shape[0]:
                    continue
                self.revert[x, y, 0] = newp[0]
                self.revert[x, y, 1] = newp[1]
                label2[x, y] = 1
        self.label = self.label + label2
        
    @staticmethod
    def fit_new_points(pFrom, pTo, target, poly_degree):
        poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
        original_points_poly = poly_features.fit_transform(pFrom)
        model = LinearRegression().fit(original_points_poly, pTo)

        new_point_poly = poly_features.transform(np.array(target).reshape(1, -1))
        mapped_new_point = model.predict(new_point_poly)
        return mapped_new_point[0]
        
def generate_thread_step(pack):
    kernel, xFrom, xTo, shape, db, grid, label = pack
    spacemap.Info("GridRevert Start %d / %d" % (xFrom, xTo))
    if xFrom < kernel:
        xFrom = kernel
    if xTo > shape - kernel:
        xTo = shape - kernel
    for x in range(xFrom, xTo):
        for y in range(kernel, shape-kernel):
            lis = []
            for ix in range(-kernel, kernel+1):
                for iy in range(-kernel, kernel+1):
                    tag = "%d_%d" % (int(x+ix), int(y+iy))
                    lis += db.get(tag, [])
            if len(lis) > 2:
                ps = np.array(lis)
                newp = GridRevert.fit_new_points(ps[:, :2], ps[:, 2:], [x, y], 1)
                if np.min(newp) < 0 or np.max(newp) > shape:
                    continue
                grid[x, y, 1] = newp[0]
                grid[x, y, 0] = newp[1]
                label[x, y] = 1
    spacemap.Info("GridRevert Finish %d / %d" % (xFrom, xTo))
    return [grid, label]

"""
def fix_revert_grid(grid, invGrid):
    # N*N*2 0-N
    
    ori_grid = torch.tensor(grid, dtype=torch.float32)
    inv_grid = torch.nn.Parameter(torch.tensor(invGrid, dtype=torch.float32))
    optimizer = torch.optim.Adam([inv_grid], lr=0.0001)
    shape = grid.shape[0]
    
    def clip(x):
            return max(min(x, shape-1), 0)
    
    def get_net(xi, yi):
        xi = clip(xi)
        yi = clip(yi)
        xy = inv_grid[xi, yi]
        return xy
    
    for iter in range(100):
        loss = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        for x in range(shape):
            for y in range(shape):
                px, py = ori_grid[x, y]
                xl, yt = int(px), int(py)
                xlratio, ytratio = px - xl, py-yt
                ptop = get_net(xl, yt) * (1 - xlratio) + \
                        get_net(xl + 1, yt) * xlratio
                pbottom = get_net(xl, yt + 1) * (1 - xlratio) + \
                        get_net(xl + 1, yt + 1) * xlratio
                p = ptop * (1 - ytratio) + pbottom * ytratio
                loss += abs(p[0] - x) + abs(p[1] - y)
        l = loss.detach().numpy()
        print(iter, l)
        loss.backward()
        optimizer.step()
    return invGrid.detach().numpy()
    
def fix_revert_grid(grid, invGrid):
    ori_grid = torch.tensor(grid, dtype=torch.float32) / (grid.shape[0] - 1) * 2 - 1
    inv_grid = torch.nn.Parameter(torch.tensor(invGrid, dtype=torch.float32) / (grid.shape[0] - 1) * 2 - 1)
    optimizer = torch.optim.Adam([inv_grid], lr=0.1)
    
    def clip(x, max_val):
        return max(min(x, max_val), 0)
    
    for iter in range(10000):
        optimizer.zero_grad()  # Clear gradients at each iteration
        loss = torch.tensor(0.0, dtype=torch.float32)  # Initialize loss
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                px, py = ori_grid[x, y]
                xl, yt = int(px), int(py)
                xlratio, ytratio = px - xl, py - yt
                xl = clip(xl, grid.shape[0] - 2)
                yt = clip(yt, grid.shape[1] - 2)
                ptop = inv_grid[xl, yt] * (1 - xlratio) + inv_grid[xl + 1, yt] * xlratio
                pbottom = inv_grid[xl, yt + 1] * (1 - xlratio) + inv_grid[xl + 1, yt + 1] * xlratio
                p = ptop * (1 - ytratio) + pbottom * ytratio
                loss += (p[0] - x) ** 2 + (p[1] - y) ** 2  # Use squared error for stability
        loss.backward()
        optimizer.step()
        print(iter, loss.item())  # Print loss at each iteration

    # Scale grid back to original coordinates
    return inv_grid.detach().numpy() * (grid.shape[0] - 1) / 2 + (grid.shape[0] - 1) / 2
"""