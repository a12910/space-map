import numpy as np

class NearPointErr:
    def __init__(self, xyd) -> None:
        self.xyd = xyd
        self.db = {}
        
    def init_db(self, points):
        points = np.array(points)
        for i in range(points.shape[0]):
            px, py = points[i]
            tag = self.xy_to_tag(px, py)
            if tag not in self.db.keys():
                self.db[tag] = []
            self.db[tag].append([i, px, py])
        
    def search(self, points, maxDis=3):
        count = len(points)
        points = np.array(points)
        result = np.zeros((count, 2))
        result[:, 0] = -1
        for i in range(count):
            find = -1
            minDis = (maxDis) * self.xyd * 1.5
            px, py = points[i]
            px_, py_ = int(px), int(py)
            ps2 = []
            for dis in range(maxDis):
                for dx in range(px_-dis, px_+dis+1):
                    for dy in range(py_-dis, py_+dis+1):
                        tag = self.xy_to_tag(dx, dy)
                        ps2 += self.db.get(tag, [])
                if len(ps2) > 0:
                    break
            if len(ps2) > 0:
                ps2_ = np.array(ps2)[:, 1:] - np.array([px, py])
                dis = ps2_[:, 0] **2 + ps2_[:, 1] **2
                minI = np.argmin(dis, axis=0)
                find = ps2[minI][0]
                minDis = dis[minI]
            result[i, 0] = find
            result[i, 1] = minDis
        result[:, 1] = np.sqrt(result[:, 1])
        return result

    def compute_err_img(self,result, maxDis=3, step=0.1):
        maxDis = self.xyd * maxDis
        length = int(maxDis // step) + 1
        count = np.zeros(length + 1)
        for i, dis in result:
            index = int(dis // step)
            if index >= length:
                index = length - 1
            count[index] += 1
        img = np.zeros((length, length))
        for i in range(length):
            v = int(np.sum(count[:i]) / len(result) * length)
            img[:v, i] = 1
        
        return count, img
                
    def xy_to_tag(self, x, y):
        return (int(x // self.xyd), int(y // self.xyd))
    