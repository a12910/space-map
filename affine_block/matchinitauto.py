import spacemap
import matplotlib.pyplot as plt
import numpy as np

class MatchInitAuto(spacemap.AffineBlock):
    def __init__(self, tmp_use_df=None):
        super().__init__("MatchInitAuto")
        self.updateMatches = True
        self.tmp_use_df = tmp_use_df
        
    def compute(self, dfI: np.array, dfJ: np.array, finder=None):
        """ dfI, dfJ -> H """
        conf_ = spacemap.IMGCONF.copy()
        conf = conf_.copy()
        conf["raw"] = 1
        spacemap.IMGCONF = conf
        imgI = spacemap.show_img(dfI)
        imgJ = spacemap.show_img(dfJ)
        align1 = spacemap.matches.LOFTR()
        matches1 = align1.compute(imgI, imgJ)
        align2 = spacemap.AffineAlignment("sift_vgg")
        matches2 = align2.compute(imgI, imgJ)
        matches = self.merge_matches(matches1, matches2)
        self.matches = matches
        spacemap.Info("Init Matches finished: %d %d" % (len(matches1), len(matches2)))
        spacemap.IMGCONF = conf_
        return None

    def compute_df(self, dfI, dfJ, finder=None):
        xyd = spacemap.XYD
        conf = spacemap.IMGCONF
        all_matches = []
        def sift_match(conf, xyd, raw_xyd):
            imgI = spacemap.show_img(dfI, conf, xyd=xyd)
            imgJ = spacemap.show_img(dfJ, conf, xyd=xyd)
            matches = spacemap.AffineAlignment("sift_vgg").compute(imgI, imgJ)
            matches = matches * xyd / raw_xyd
            return matches
        for xyd_ in [xyd, xyd * 2, xyd // 2]:
            for raw in [0, 1]:
                for kernel in [0, 1]:
                    conf1 = conf.copy()
                    conf1["raw"] = raw
                    conf1["kernel"] = kernel
                    matches = sift_match(conf1, xyd_, xyd)
                    if matches.shape[0] > 0:
                        all_matches.append(matches)
        all_matches = np.concatenate(all_matches, axis=0)
        self.matches = all_matches
        spacemap.Info("Init Multi-DF-Matches finished: %d" % (len(all_matches)))
        return None
    
    def compute_img(self, imgI: np.array, imgJ: np.array, finder=None):
        spacemap.Info("Init Matches start")
        if self.tmp_use_df:
            dfI, dfJ = self.tmp_use_df
            return self.compute_df(dfI, dfJ)
        align1 = spacemap.matches.LOFTR()
        matches1 = align1.compute(imgI, imgJ)
        imgI = imgI.copy()
        imgJ = imgJ.copy()
        # imgI[imgI > 0] = 1
        # imgJ[imgJ > 0] = 1
        align2 = spacemap.AffineAlignment("sift_vgg")
        matches2 = align2.compute(imgI, imgJ)
        matches = self.merge_matches(matches1, matches2)
        self.matches = matches
        spacemap.Info("Init Matches finished: %d %d" % (len(matches1), len(matches2)))
        return None
    
    def merge_matches(self, matches1, matches2):
        matches = []
        if len(matches1) < len(matches2):
            matches1, matches2 = matches2, matches1
        for i, match1 in enumerate(matches1):
            matches.append(match1)
            if i >= len(matches2):
                continue
            matches.append(matches2[i])
        return matches
    