import numpy as np
import tqdm, json
import spacemap

def _process_cells(expressions, cell_positions, maxShape, skip, overlap, tqdmShow=True):
    # 减去每种产物的最小值
    expressions -= np.min(expressions, axis=0)
    
    # 计算方格数量
    grid_x = int((maxShape[0] - overlap) / (skip - overlap))
    grid_y = int((maxShape[1] - overlap) / (skip - overlap))
    
    # 初始化结果数组
    result = np.zeros((grid_x, grid_y, expressions.shape[1]), dtype=np.float32)
    
    # 遍历每个方格
    for i in tqdm.trange(grid_x, disable=not tqdmShow):
        for j in range(grid_y):
            # 计算方格的边界
            x_min = i * (skip - overlap)
            x_max = x_min + skip
            y_min = j * (skip - overlap)
            y_max = y_min + skip
            
            # 筛选在方格内的细胞
            inside = (cell_positions[:, 0] >= x_min) & (cell_positions[:, 0] < x_max) & \
                     (cell_positions[:, 1] >= y_min) & (cell_positions[:, 1] < y_max)
            
            result[i, j] = np.sum(expressions[inside], axis=0)
            total = np.sum(result[i, j])
            if total > 0:
                result[i, j] /= total
    return result

def _barcode_load(index, folder):
    """ features * cells """
    from scipy.io import mmread
    if folder is None:
        folder = "/Users/hrd/CODE/code_works/3ddcell/points2/data/flow/matrix/%d" % index
    else:
        folder = folder % index
    matrix = folder + "/matrix.mtx.gz"
    data = mmread(matrix)
    data = data.toarray()
    return data.T

def calculate_layer_distances(dff, size, overlay, maxShape, barcodePath=None, savePath=None):
    # 获取层的范围
    layers = sorted(dff['layer'].unique())
    matrixs = {}
    for l in layers:
        xy = dff[dff["layer"] == l][["x", "y"]].values
        spacemap.Info("Load Layer %d: %d cells" % (l, len(xy)))
        mat = _barcode_load(l, barcodePath)
        mat2 = _process_cells(mat, xy, maxShape, size, overlay, True)
        matrixs[l] = mat2
    distances = []
    
    def compute(i, j):
        mat1 = matrixs[i]
        mat2 = matrixs[j]
        gene = mat1.shape[2]
        mat1 = mat1.reshape((-1, gene))
        mat2 = mat2.reshape((-1, gene))
        dis = np.sum(np.abs(mat1 - mat2))
        label = np.sum(mat1, axis=1) + np.sum(mat2, axis=1)
        label[label > 0] = 1
        dis /= np.sum(label)
        return dis

    # 遍历每一对相邻层
    for i in range(len(layers) - 1):
        dis = compute(layers[i], layers[i + 1])
        distances.append(dis)
        
    if savePath is not None:
        with open(savePath, "w") as f:
            dis1 = list(map(float, dis[1]))
            pack = {"mean": float(dis[0]), 
                "distances": dis1, 
                "size": size, 
                "overlay": overlay
            }
            f.write(json.dumps(pack))
    spacemap.Info("Mean Distance: %f" % np.mean(distances))
    # 返回所有层对的平均距离
    return np.mean(distances), distances
