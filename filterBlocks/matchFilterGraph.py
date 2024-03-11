import spacemap
import matplotlib.pyplot as plt
import numpy as np

class MatchFilterGraph(spacemap.AffineBlock):
    def __init__(self, std):
        super().__init__("MatchFilterGraph")
        self.update_matches = True
        self.std = std
        self.show_graph_match = False
        self.history = []
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        matches = self.matches
        matches1 = self.matches_filter(matches, self.std, dfI, dfJ)
        spacemap.Info("Graph Filter Matches: %d -> %d" % (len(matches), len(matches1)))
        self.matches = matches1
        return None
    
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
            if abs(maxV - mean) > std * stdd:
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
    