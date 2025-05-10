# 3D Reconstruction

> 相关源文件：
> - [affine/FilterGraphPart.py](https://github.com/a12910/space-map/blob/ad208055/affine/FilterGraphPart.py)
> - [flow/__init__.py](https://github.com/a12910/space-map/blob/ad208055/flow/__init__.py)
> - [flow/afFlow2Basic.py](https://github.com/a12910/space-map/blob/ad208055/flow/afFlow2Basic.py)
> - [flow/afFlow2MultiDF.py](https://github.com/a12910/space-map/blob/ad208055/flow/afFlow2MultiDF.py)
> - [matches/__init__.py](https://github.com/a12910/space-map/blob/ad208055/matches/__init__.py)
> - [matches/matchInitMulti.py](https://github.com/a12910/space-map/blob/ad208055/matches/matchInitMulti.py)
> - [registration/ldm/torch_LDDMM2D.py](https://github.com/a12910/space-map/blob/ad208055/registration/ldm/torch_LDDMM2D.py)
> - [utils/AutoGradHE.py](https://github.com/a12910/space-map/blob/ad208055/utils/AutoGradHE.py)
> - [utils/model3d.py](https://github.com/a12910/space-map/blob/ad208055/utils/model3d.py)

## 简介

SpaceMap 的 3D 重建系统将一系列已配准的 2D 组织切片转换为统一的 3D 模型。

- 先对每个切片生成坐标变换 grid
- 应用变换将切片数据映射到统一 3D 空间
- 合并所有切片数据，形成完整 3D 组织结构

For more about image alignment, see: [Image Alignment](../alignment/image-alignment.md).

## Grid 生成

- 通过 `model3d.generate_grid` 为每层生成稠密变换场（grid），实现原始与配准空间的精确映射
- 支持并行处理提升效率
- grid 以 numpy 数组保存，便于后续应用

相关代码片段：
- [model3d.py#L18-L47](https://github.com/a12910/space-map/blob/ad208055/utils/model3d.py#L18-L47)

## 变换应用

### 图像变换
- `grid_sample_img` 利用 PyTorch grid sampling 对图像进行双线性插值变换
- 支持高效 GPU 加速

相关代码片段：
- [model3d.py#L68-L104](https://github.com/a12910/space-map/blob/ad208055/utils/model3d.py#L68-L104)

### 点集变换
- 细胞等点数据通过 AutoFlow 流程先仿射再 grid 非线性变换

相关代码片段：
- [afFlow2Basic.py#L202-L210](https://github.com/a12910/space-map/blob/ad208055/flow/afFlow2Basic.py#L202-L210)

## 多分辨率与中心切片策略

- 采用多分辨率重建，兼顾效率与精度
- 以中心切片为参考，双向配准，减少误差累积
- 支持并行处理

相关代码片段：
- [afFlow2MultiDF.py#L62-L73](https://github.com/a12910/space-map/blob/ad208055/flow/afFlow2MultiDF.py#L62-L73)

## LDM Pair 配准

- 邻近切片间采用 LDDMM（LDM Pair）非线性配准，生成高精度 grid
- 结果用于后续 grid 合成与 3D 重建

相关代码片段：
- [afFlow2MultiDF.py#L21-L60](https://github.com/a12910/space-map/blob/ad208055/flow/afFlow2MultiDF.py#L21-L60)

## Grid 合成

- 支持正向（raw→aligned）与逆向（aligned→raw）grid 合成，形成全局一致的 3D 坐标系

相关代码片段：
- [afFlow2Basic.py#L211-L223](https://github.com/a12910/space-map/blob/ad208055/flow/afFlow2Basic.py#L211-L223)

## 3D 模型导出

- 可导出每层变换后图像、grid、点坐标等
- 便于可视化与后续分析

相关代码片段：
- [flow/__init__.py#L1-L15](https://github.com/a12910/space-map/blob/ad208055/flow/__init__.py#L1-L15)
- [model3d.py#L9-L17](https://github.com/a12910/space-map/blob/ad208055/utils/model3d.py#L9-L17)

## 误差评估

- 提供切片间误差评估工具，量化重建质量

相关代码片段：
- [model3d.py#L48-L62](https://github.com/a12910/space-map/blob/ad208055/utils/model3d.py#L48-L62)

## 全流程集成

- 3D 重建流程集成了 grid 生成、变换应用、配准、合成、导出与误差评估
- 关键函数包括：`generate_imgs_basic`、`generate_grid`、`grid_sample_img`、`ldm_pair`、`ldm_merge_pair`

相关代码片段：
- [model3d.py#L9-L104](https://github.com/a12910/space-map/blob/ad208055/utils/model3d.py#L9-L104)
- [afFlow2MultiDF.py#L21-L158](https://github.com/a12910/space-map/blob/ad208055/flow/afFlow2MultiDF.py#L21-L158)
- [registration/ldm/torch_LDDMM2D.py#L47-L63](https://github.com/a12910/space-map/blob/ad208055/registration/ldm/torch_LDDMM2D.py#L47-L63)

## 性能优化

- 并行处理 grid 生成
- GPU 加速变换
- 中心切片策略减少误差累积

相关代码片段：
- [model3d.py#L32-L34](https://github.com/a12910/space-map/blob/ad208055/utils/model3d.py#L32-L34)
- [afFlow2MultiDF.py#L120-L124](https://github.com/a12910/space-map/blob/ad208055/flow/afFlow2MultiDF.py#L120-L124)
- [registration/ldm/torch_LDDMM2D.py#L19-L22](https://github.com/a12910/space-map/blob/ad208055/registration/ldm/torch_LDDMM2D.py#L19-L22)

## 重要组件表

| 组件                | 说明                   | 实现位置                       |
|---------------------|------------------------|-------------------------------|
| GridGenerate        | 生成变换 grid          | model3d.py                    |
| apply_grid          | 应用 grid 到坐标       | Slice 类                      |
| _merge_grid         | grid 合成              | model3d.py                    |
| grid_sample_img     | grid 可视化/图像变换   | model3d.py                    |

---

SpaceMap 的 3D 重建系统通过高效的 grid 变换、LDDMM 配准、多分辨率与中心切片策略，实现了高精度、可扩展的组织 3D 重建。 