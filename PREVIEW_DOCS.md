# 文档预览指南

在推送之前，你可以在本地预览文档网站，确保一切正常。

## 快速预览

### 方法1: 使用 mkdocs serve（推荐）

```bash
# 在项目根目录运行
mkdocs serve
```

然后在浏览器中打开: **http://127.0.0.1:8000**

这将启动一个本地服务器，你可以实时预览文档。修改文件后会自动刷新。

### 方法2: 构建静态网站

```bash
# 构建网站到 site/ 目录
mkdocs build

# 然后用任何HTTP服务器查看
python -m http.server 8000 -d site
```

## 检查清单

在推送之前，请检查以下内容：

### ✅ Home 页面
- [ ] 打开 http://127.0.0.1:8000
- [ ] 检查所有章节都完整显示
- [ ] 检查代码示例格式正确
- [ ] 确认没有 "[占位符]" 文本
- [ ] 确认所有链接都可点击

### ✅ Installation 页面
- [ ] 打开 http://127.0.0.1:8000/getting-started/installation/
- [ ] 检查所有安装步骤清晰
- [ ] 检查代码块格式正确
- [ ] 检查故障排除部分完整

### ✅ Quick Start 页面
- [ ] 打开 http://127.0.0.1:8000/getting-started/quickstart/
- [ ] 检查完整工作流程显示正确
- [ ] 检查所有代码示例有语法高亮
- [ ] 检查常见问题部分完整

### ✅ Contributing 页面
- [ ] 打开 http://127.0.0.1:8000/contributing/
- [ ] 检查所有章节完整
- [ ] 检查代码块格式正确
- [ ] 检查链接都有效

### ✅ 导航和搜索
- [ ] 检查导航栏只显示3个部分
- [ ] 检查搜索功能正常
- [ ] 检查移动端响应式布局（缩小浏览器窗口）

### ✅ 样式和排版
- [ ] 检查标题层级正确
- [ ] 检查代码块有背景色
- [ ] 检查列表缩进正确
- [ ] 检查表格显示正常

## 如果发现问题

### 文档内容问题
编辑相应的 `.md` 文件：
- `docs/index.md` - Home页面
- `docs/getting-started/installation.md` - 安装指南
- `docs/getting-started/quickstart.md` - 快速开始
- `docs/contributing.md` - 贡献指南

保存后，如果 `mkdocs serve` 在运行，页面会自动刷新。

### 导航结构问题
编辑 `mkdocs.yml` 的 `nav:` 部分。

### 构建错误
运行以下命令查看详细错误：
```bash
mkdocs build --verbose
```

## 推送和部署

确认一切正常后：

```bash
# 提交更改
git add docs/ mkdocs.yml
git commit -m "Docs: simplify documentation, remove placeholders"

# 推送到GitHub
git push origin master
```

GitHub Actions 会自动构建和部署文档到:
**https://a12910.github.io/space-map**

部署通常需要 2-5 分钟。你可以在这里查看部署状态:
**https://github.com/a12910/space-map/actions**

## 故障排除

### mkdocs命令未找到
```bash
pip install mkdocs mkdocs-material
```

### 端口8000已被占用
```bash
mkdocs serve -a 127.0.0.1:8001
```

### 构建警告
当前的警告是关于旧文档文件的链接，这些文件不在导航中，可以忽略。

---

**准备好了吗？** 运行 `mkdocs serve` 开始预览！
