import spacemap
import numpy as np

class AffineBlockSiftNear(spacemap.AffineBlockSiftEach):
    def __init__(self, matchr=0.75, k=3, disr=0.2, outputH=True):
        super().__init__("AffineBlockSiftEach")
        self.matchr = matchr
        self.k = k
        self.disr = disr
        self.outputH = outputH
    
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        """ dfI, dfJ -> H """
        xyd = spacemap.XYD
        imgI = spacemap.show_img3(dfI)
        imgJ = spacemap.show_img3(dfJ)
        matches = spacemap.siftImageAlignment2(imgI, imgJ, matchr=self.matchr, k=self.k)
        distances = np.zeros((len(matches), self.k))
        goodMatches = []
        meanIX, meanIY = np.mean(dfI[:, 0]), np.mean(dfI[:, 1])
        meanJ2X, meanJ2Y = np.mean(dfJ[:, 0]), np.mean(dfJ[:, 1])
        for i in range(len(matches)):
            for j in range(self.k):
                p1x, p1y, p2x, p2y, _ = matches[i][j]
                c1x, c1y = (meanIX // xyd), (meanIY // xyd)
                c2x, c2y = (meanJ2X // xyd), (meanJ2Y // xyd)
                distances[i][j] = self.compute_sift_distance2(p1x, p1y, c1x, c1y, p2x, p2y, c2x, c2y)
            maxID = np.argmax(distances[i])
            maxDis = distances[i][maxID]
            if abs(1 - maxDis) < self.disr:
                goodMatches.append(matches[i][maxID])
        spacemap.Info("LDDMM6 AlignMatchFilter get %d -> %d" % (len(matches), len(goodMatches)))
        goodMatches.sort(key=lambda x: x[-1])
        if self.match_rate_filter > 0:
            goodMatches = self.compute_match_filter(goodMatches, 
                                                   self.match_rate_filter,
                                                    width=imgI.shape[0])
           
        self.matches = goodMatches
        if self.show_match:
            spacemap.AffineBlock.show_matches(goodMatches, dfI, dfJ, H)
        if self.outputH:
            bestH, _ = self.compute_each_match(goodMatches, 4, dfI, dfJ,finder, imgI)
            return bestH
        else:
            return None
    
    def compute_sift_distance(self, p1x, p1y, c1x, c1y, p2x, p2y, c2x, c2y):
        l1x, l1y = p1x - c1x, p1y - c1y
        l2x, l2y = p2x - c2x, p2y - c2y
        ll = max((l1x **2 + l1y **2), (l2x **2 + l2y **2))
        result = (l1x * l2x + l1y * l2y) / ll
        return result

    def compute_sift_distance2(self, p1x, p1y, c1x, c1y, p2x, p2y, c2x, c2y):
        l1x, l1y = p1x - c1x, p1y - c1y
        l2x, l2y = p2x - c2x, p2y - c2y
        l1 = np.sqrt(l1x **2 + l1y **2)
        l2 = np.sqrt(l2x **2 + l2y **2)
        dis1 = 1 - (abs(l1 - l2) / (c1x + c2x)) * 10
        dis2 = (l1x * l2x + l1y * l2y) / (l1 * l2)
        result = dis1 * dis2
        if dis1 < 0 or dis2 < 0:
            result = -abs(result)
        return result
    
# class AffineBlockSiftShow(lddmm.AffineBlock):
#     def __init__(self, output: lddmm.AffineBlock):
#         super().__init__("AffineBlockSiftShow")
#         self.output = output
    
#     def compute(self, dfI: np.array, dfJ: np.array, finder=None):
#         matches = self.output.matches
#         I = lddmm.show_img3(dfI)
#         J = lddmm.show_img3(dfJ)
#         plt.figure(figsize=(10,10))
#         IJ = np.concatenate((I, J), axis=1)
#         print(matches)
#         for m in matches:
#             plt.plot([m[0], m[2]+I.shape[1]], [m[1], m[3]], 'r-', linewidth=5)
#         plt.imshow(IJ)
#         plt.show()
#         return None
    