# Space-map 大修改进计划

## 第一阶段：审稿人必要项

### 1. Benchmark 复现脚本
- [ ] 创建 `benchmarks/` 目录结构
- [ ] 编写 Space-map 在 benchmark 数据集上的运行脚本
- [ ] 编写 PASTE 对比脚本
- [ ] 编写 STalign 对比脚本
- [ ] 编写统一评估脚本（调用 `find.err` 模块的 Dice/SSIM/PSNR）
- [ ] 为每个实验提供参数配置文件（xenium_colon.json, codex_colon.json）
- [ ] 编写论文关键图表的复现脚本（fig2, fig3 等）
- [ ] 编写 `benchmarks/README.md` 说明如何复现

### 2. Tutorial 体系（6 个 notebook）
- [ ] T1: 最小示例（< 1MB toy data，5 分钟跑完，验证安装）
- [ ] T2: Xenium 空间转录组端到端流程
- [ ] T3: CODEX 空间蛋白组端到端流程
- [ ] T4: 参数调优指南（sift / loftr / auto 选择、LDDMM 参数）
- [ ] T5: 结果评估与 QC（Dice/SSIM 解读、对齐质量判断）
- [ ] T6: 与下游工具对接（导出到 AnnData → scanpy/squidpy）

### 3. 数据可及性
- [ ] 制作 < 1MB 的 toy dataset（3-5 个 slice，几千个细胞）
- [ ] 将完整 benchmark 数据上传 Zenodo / Figshare，获取 DOI
- [ ] 为 cells2.csv.gz 补充元信息（来源实验、slice 数量、坐标范围）
- [ ] 编写数据格式规范文档（列名、坐标系、layer 编号规则）

---

## 第二阶段：生态集成与 API 改进

### 4. AnnData / SpatialData 集成
- [ ] FlowImport 增加 `init_from_anndata(adata, layer_key=...)` 方法
- [ ] FlowExport 增加导出到 `adata.obsm["spatial_aligned"]` 的方法
- [ ] 支持 .h5ad 文件直接读写

### 5. 高层 API 封装
- [ ] 提供 `spacemap.align(xys, ids=, method=)` 一行式接口
- [ ] 提供 `spacemap.align_from_csv(path, layer_col=)` 便捷入口
- [ ] 提供 `spacemap.align_from_anndata(adata)` 便捷入口
- [ ] 底层分步 API 保留给高级用户

### 6. 清理公开 API 与命名
- [ ] 消除 `from X import *`，每个模块定义 `__all__`
- [ ] 统一对外暴露的类名（`AutoFlowMultiCenter4` → `RegistrationPipeline` 或类似）
- [ ] 历史版本类加 `_` 前缀标记为私有
- [ ] 修正 Python 版本声明（`>=3.7` → `>=3.9`，或加 `from __future__ import annotations`）

---

## 第三阶段：代码质量与可维护性

### 7. 测试
- [ ] 创建 `tests/` 目录
- [ ] 端到端集成测试（toy data 走完 affine → LDDMM → export）
- [ ] 核心数学函数单元测试（变换、网格插值）
- [ ] CI 中跑 pytest（替换当前的 import 冒烟测试）

### 8. 依赖与环境
- [ ] requirement.txt 加最低版本约束（`torch>=1.10` 等）
- [ ] 提供 `environment.yml`（conda，精确复现论文环境）
- [ ] 编写 Dockerfile
- [ ] 考虑将 PyTorch GPU 版本设为可选依赖

### 9. 日志与进度
- [ ] 将 304 个 `print()` 迁移到 `logging` 模块
- [ ] 计算密集循环统一用 `tqdm` 进度条
- [ ] 提供 `verbose` 参数或日志级别控制

### 10. 错误处理
- [ ] 定义 `spacemap.errors` 模块（DataNotFoundError, RegistrationError 等）
- [ ] 替换 `raise Exception("no Data")` 为具体异常 + 上下文信息
- [ ] 公开 API 方法增加输入校验

### 11. 配置管理
- [ ] 将全局变量（XYRANGE, XYD, DEVICE）封装为 Config 类
- [ ] 支持配置对象传入，允许多组参数并行运行
- [ ] GPU 自动检测（CUDA 可用时自动启用）

### 12. 代码重构
- [ ] 消除 `ldm/` 和 `ldm2/` 之间的代码重复（提取公共基类）
- [ ] 拆分 1700+ 行的大文件（数学核心 / 可视化 / IO 分离）

---

## 第四阶段：文档与发布

### 13. 补全文档
- [ ] 用 mkdocstrings 自动生成完整 API Reference
- [ ] 核心公开类补充 docstring（AutoFlow, LDDMM2D, FlowImport, FlowExport, Slice）
- [ ] 添加 Algorithm Overview 页面（两阶段配准原理图示）
- [ ] 添加 FAQ / Troubleshooting 页面
- [ ] 添加 Data Format Specification 页面
- [ ] 将 CHANGELOG.md 纳入 mkdocs 导航
- [ ] 清理 docs/ 中的重复文件

### 14. LoFTR 预训练权重说明
- [ ] 文档说明权重下载方式（是否自动下载）
- [ ] 说明是否支持 fine-tune（如不支持，明确标注"开箱即用"）
- [ ] 离线环境下的权重使用方案

### 15. 发布与引用
- [ ] 为代码创建 Zenodo DOI
- [ ] 添加 `CITATION.cff` 文件
- [ ] 发布到 PyPI
- [ ] 发布 Docker image 到 GitHub Container Registry

### 16. 可选：CLI 工具
- [ ] `spacemap align --input cells.csv --method auto --output result/`
- [ ] `spacemap evaluate --aligned result/ --metrics dice,ssim`
- [ ] 方便集成到 Snakemake / Nextflow pipeline
