from numpy.core.multiarray import array as array
import spacemap
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class AffineBlockSiftGraph(spacemap.AffineBlock):
    def __init__(self, matchr=0.75, std=1.5):
        super().__init__("AffineBlockSiftGrap")
        self.matchr = matchr
        self.show_match = False
        self.show_graph_match = False
        self.std = std
        self.minMatch = 4
        self.history = []
        self.use_matches = None
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        imgI = spacemap.show_img3(dfI)
        imgJ = spacemap.show_img3(dfJ)
        matches = self.alignment.compute(imgI, imgJ, 
                                         matchr=self.matchr)
        matches1 = self.matches_filter(matches, self.std, dfI, dfJ)
        spacemap.Info("SiftGraph Filter Matches: %d -> %d" % (len(matches), len(matches1)))
        H, index = spacemap.AffineBlockSiftEach.compute_each_match(matches1, self.minMatch, dfI, dfJ, finder, imgI)
        if self.show_match:
            self.show_matches(matches1[:index], dfI, dfJ, H)
        return H
    
    @staticmethod
    def compute_distance_matrix(points1, points2):
        """ p1[i] - p2[j] = dis[i][j] """
        size = len(points1)
        p1x = np.repeat(np.expand_dims(points1[:, 0], axis=1),
                        size, axis=1)
        p2x = np.repeat(np.expand_dims(points2[:, 0], axis=0), 
                        size, axis=0)
        x_dis = abs(p1x - p2x)
        p1y = np.repeat(np.expand_dims(points1[:, 1], axis=1), 
                        size, axis=1)
        p2y = np.repeat(np.expand_dims(points2[:, 1], axis=0), 
                        size, axis=0)
        y_dis = abs(p1y - p2y)
        dis = x_dis **2 + y_dis **2
        return dis
        
    def matches_filter(self, matches, stdd, dfI, dfJ):
        p1s = matches[:, :2]
        p2s = matches[:, 2:4]
        graph_dis = np.sum((p1s - p2s) ** 2, axis=1)
        index = 0

        I = spacemap.show_img3(dfI)
        J = spacemap.show_img3(dfJ)
        
        while True:
            maxIndex = np.argmax(graph_dis)
            maxV = graph_dis[maxIndex]
            mean = np.mean(graph_dis[graph_dis > 0])
            std = np.std(graph_dis[graph_dis > 0])
            self.history.append([index, mean, std])
            if abs(maxV - mean) > std * 2:
                graph_dis[maxIndex] = 0
                matches1 = matches[graph_dis > 0]
                if index % 10 == 0 and self.show_graph_match:
                    print(self.history[-1])
                    plt.figure(figsize=(10,10))
                    plt.imshow(np.concatenate((I, J), axis=1))
                    for m in matches1:
                        plt.plot([m[1], m[3]+I.shape[1]], [m[0], m[2]], 'r-', linewidth=2)
                    plt.show()
            else:
                break
            index += 1
            if len(matches1) < 8:
                break
            
        matches1 = matches[graph_dis > 0]
        return matches1
