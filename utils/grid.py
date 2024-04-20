from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import tqdm

class GridGenerate:
    def __init__(self, shape, xyd, degree=1) -> None:
        """ 从运动的点生成转换网格
        Args:
            shape (int, int): 网格的size range//xyd
            xyd (int): xyd
            degree (int): 拟合次数 
        Methods:
            init_db: 初始化数据库
            generate: 生成网格
            fix: 修正网格
        """
        self.shape = shape
        self.xyd = xyd
        self.grid = np.zeros((shape[0], shape[1], 2))
        self.label = np.zeros(shape)
        self.degree = degree
        
        self.mesh = np.zeros_like(self.grid)
        for i in range(shape[0]):
            self.mesh[i, :, 0] = i * xyd
            self.mesh[:, i, 1] = i * xyd
        self.db = {}
        self.useTQDM = True
        
    @staticmethod
    def fit_new_points(pFrom, pTo, target, poly_degree):
        poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
        original_points_poly = poly_features.fit_transform(pFrom)
        model = LinearRegression().fit(original_points_poly, pTo)

        new_point_poly = poly_features.transform(np.array(target).reshape(1, -1))
        mapped_new_point = model.predict(new_point_poly)
        return mapped_new_point[0]
        
    def show_grid(self):
        maxx = self.shape[0] * self.xyd * 1.2
        img = np.zeros((*self.grid.shape[:2], 3))
        img[:, :, 0] = self.grid[:, :, 0]
        img[:, :, 1] = self.grid[:, :, 1]
        img = img / maxx
        plt.imshow(img)
        plt.show()
        
    def xy_to_tag(self, x, y):
        return "%d_%d" % (int(x // self.xyd), int(y // self.xyd))
        
    def init_db(self, pFrom, pTo):
        pFrom = np.array(pFrom)
        pTo = np.array(pTo)
        for i in range(len(pFrom)):
            fx, fy = pFrom[i]
            tx, ty = pTo[i]
            tag = self.xy_to_tag(fx, fy)
            if tag not in self.db.keys():
                self.db[tag] = []
            self.db[tag].append([fx, fy, tx, ty])
            
    def generate(self, kernel=1):
        xyd = self.xyd
        
        for x in tqdm.trange(kernel, self.shape[0]-kernel, 
                             disable=not self.useTQDM):
            for y in range(kernel, self.shape[1]-kernel):
                lis = []
                for ix in range(-kernel, kernel+1):
                    for iy in range(-kernel, kernel+1):
                        tag = self.xy_to_tag((x+ix)*xyd, (y+iy)*xyd)
                        lis += self.db.get(tag, [])
                if len(lis) > 2:
                    ps = np.array(lis)
                    newp = GridGenerate.fit_new_points(ps[:, :2], ps[:, 2:], [x*xyd, y*xyd], self.degree)
                    if np.min(newp) < 0 or np.max(newp) > self.xyd * self.shape[0]:
                        continue
                    self.grid[x, y, 0] = newp[0]
                    self.grid[x, y, 1] = newp[1]
                    self.label[x, y] = 1
        
    def fix(self, kernel2=10):
        label2 = np.zeros_like(self.label)
        for x in tqdm.trange(kernel2, self.shape[0]-kernel2,
                             disable=not self.useTQDM):
            for y in range(kernel2, self.shape[1]-kernel2):
                if self.label[x, y] > 0:
                    continue
                partl = self.label[x-kernel2:x+kernel2+1, y-kernel2:y+kernel2+1]
                if np.sum(partl) < 3:
                    continue
                part2 = self.grid[x-kernel2:x+kernel2+1, y-kernel2:y+kernel2+1, :]
                part_mesh = self.mesh[x-kernel2:x+kernel2+1, y-kernel2:y+kernel2+1, :]
                part3 = np.concatenate([part_mesh, part2], axis=2)
                ps = part3[partl > 0, :].reshape(-1, 4)
                newp = GridGenerate.fit_new_points(ps[:, :2], ps[:, 2:], [x*self.xyd, y*self.xyd], self.degree)
                if np.min(newp) < 0 or np.max(newp) > self.xyd * self.shape[0]:
                    continue
                self.grid[x, y, 0] = newp[0]
                self.grid[x, y, 1] = newp[1]
                label2[x, y] = 1
        self.label = self.label + label2
        
    def grid_sample_points(self, points):
        result = []
        phi = self.grid
        xyd = self.xyd
        imgMaxX = int(self.shape[0]) - 1
        imgMaxY = int(self.shape[1]) - 1
        def get_net(xi, yi):
            if xi > imgMaxX:
                xi = imgMaxX
            if yi > imgMaxY:
                yi = imgMaxY
            if xi < 0:
                xi = 0
            if yi < 0:
                yi = 0
            x = phi[xi, yi, 0]
            y = phi[xi, yi, 1]
            return np.array([x, y])
        
        for x, y in points:
            xl = int(x // self.xyd)
            yt = int(y // self.xyd)
            xlratio = (x % xyd) / xyd
            ytratio = (y % xyd) / xyd
            ptop = get_net(xl, yt) * (1 - xlratio) + get_net(xl + 1, yt) * xlratio
            pbottom = get_net(xl, yt + 1) * (1 - xlratio) + get_net(xl + 1, yt + 1) * xlratio
            p = ptop * (1 - ytratio) + pbottom * ytratio
            tx, ty = p[0], p[1]
            result.append([tx, ty])
        result = np.array(result)
        return result
        
    