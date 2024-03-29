import spacemap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def show_cells(df: pd.DataFrame, xyr=None, 
               cells: dict={}, s=1, alpha=0.2, outTag=""):
    df = df.copy()
    fig,ax = plt.subplots()
    if xyr is None:
        xyr = spacemap.XYRANGE
    minx, maxx, miny, maxy = xyr
    df = df[(df["x"] > minx) & (df["y"] > miny) & (df["x"] < maxx) & (df["y"] < maxy)].copy()
    df["select"] = 0
    for typ, c in cells.items():
        if typ == "OTHER":
            ps = np.array(df[df["select"] == 0][["x", "y"]].values)
        else:
            ps = np.array(df[df["cell_type"] == typ][["x", "y"]].values)
            df.loc[df["cell_type"] == typ, "select"] = 1
        xI = np.array(ps[:, 0])
        yI = np.array(ps[:, 1])
        if c == "":
            ax.scatter(xI,yI,s=s,alpha=alpha, linewidths=0, label=typ)
        else:
            ax.scatter(xI,yI,s=s,alpha=alpha, linewidths=0, label=typ, c=c)
    ax.legend(markerscale = 10)
    plt.axis('off')
    ax.get_legend().remove()
    import time
    tags = outTag.split("-")
    spacemap.mkdir("%s/imgs/cells-%s" % (spacemap.BASE, tags[0]))
    tags1 = ""
    if len(tags) > 1:
        tags1 = tags[1]
    path = "%s/imgs/cells-%s/%s.png" % (spacemap.BASE, tags[0], tags1)
    fig.savefig(path, transparent=True)

def show_cells_layers(df: pd.DataFrame, start=0, end=1, 
                      xyr=None, cells: dict={}, s=1, 
                      alpha=0.2, outTag=""):
    for layer in range(start, end+1):
        df1 = df[df["layer"] == layer].copy()
        tagg = outTag + "-" + str(layer)
        show_cells(df1, xyr, cells, s, alpha, tagg)
        