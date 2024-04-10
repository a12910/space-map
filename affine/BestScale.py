import spacemap
import cv2
import numpy as np

class BestScale(spacemap.AffineBlock):
    def __init__(self, minS, maxS):
        super().__init__("Scale")
        self.minS = minS
        self.maxS = maxS
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        xyd = spacemap.XYD
        meanJX, meanJY = np.mean(dfJ[:, 0]), np.mean(dfJ[:, 1])
        xJ, yJ = (int(meanJX // xyd), int(meanJY // xyd))
        imgI = spacemap.show_img3(dfI)
        imgJ = spacemap.show_img3(dfJ)
        minS, maxS = self.minS, maxS
        w, h, = imgI.shape[:2]
        
        for xs_ in range(int(minS*100), int(maxS * 100) + 1, 1):
            for ys_ in range(int(minS*100), int(maxS * 100) + 1, 1):
                xs = xs_ / 100
                ys = ys_ / 100
                xJ_ = int((100 - xs_) * xJ / 100)
                yJ_ = int((100 - ys_) * yJ / 100)
                wJ_ = int(xs * w)
                hJ_ = int(ys * h)
                imgJ_ = cv2.resize(imgJ, (hJ_, wJ_))
                H = np.eye(3, 3)
                H[0, 0] = xs
                H[1, 1] = ys
                if xs_ <= 100 and ys_ <= 100:
                    graph = np.zeros_like(imgI)
                    graph[xJ_:xJ_+wJ_, yJ_:yJ_+hJ_] = imgJ_
                    finder.add_result((xs, ys), H, graph, imgI)
                elif xs_ >= 100 and ys_ >= 100:
                    graph = np.zeros_like(imgJ_)
                    graph[-xJ_:-xJ_+w, -yJ_:-yJ_+h] = imgI
                    finder.add_result((xs, ys), H, graph, imgJ_)
                elif xs_ >= 100:
                    graph1 = np.zeros((wJ_, h))
                    graph2 = np.zeros((wJ_, h))
                    graph1[-xJ_:-xJ_+w, :] = imgI
                    graph2[:, yJ_:yJ_+hJ_] = imgJ_
                    finder.add_result((xs, ys), H, graph1, graph2)
                else:
                    graph1 = np.zeros((w, hJ_))
                    graph2 = np.zeros((w, hJ_))
                    graph1[:, -yJ_:-yJ_+h] = imgI
                    graph2[xJ_:xJ_+wJ_, :] = imgJ_
                    finder.add_result((xs, ys), H, graph1, graph2)
        return finder.bestH()
