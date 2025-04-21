import spacemap
from spacemap import Slice
import pandas as pd

# step1: 载入数据
xys = [] # xy坐标 N*2 np.array
ids = [] # 层ID

df = pd.read_csv("../../data/flow/cells.csv.gz")
groups = df.groupby("layer")
for layer, dff in groups:
    xy = dff[["x", "y"]].values
    ids.append(layer)
    xys.append(xy)

# 项目base文件夹 后面所有数据都会放在里面
base = "data/flow"

# 自动载入项目
flow = spacemap.flow.FlowImport(base)

# 初始化xy坐标和层ID
flow.init_xys(xys, ids)

# 获取切片
slices = flow.slices

# 进行自动仿射变换配准
mgr = spacemap.flow.AutoFlowMultiCenter3(slices)

# 自动对齐
mgr.alignMethod = "auto"
mgr.affine("DF", show=True)

# 进行LDDMM高精度配准
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=True)

# 导出数据结果
export = spacemap.flow.FlowExport(slices)
imgs = export.export_imgs()