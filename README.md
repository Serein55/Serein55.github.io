# Serein55.github.io

甘仁达的个人主页 · 数学与应用数学 · 上海交通大学 2024 级

极简清新风格静态站点，托管于 GitHub Pages。

## Learning 笔记

Learning 收录 4 篇 Embodied AI 笔记，默认收起，点击分类标题后展开。整理时保留原文、公式与代码，统一标题层级、列表和图片路径，并生成目录及独立阅读页面。

- `learning/embodied-ai/notes/`：整理后的 Markdown，后续直接编辑这些文件。
- `learning/embodied-ai/notes.json`：文章顺序、标题、分类与简介。
- `learning/embodied-ai/images/`：集中保存的 5 张配图。
- `learning.css`：笔记列表和正文排版样式。
- `assets/katex/`：数学排版样式、字体和许可证，随网站一起托管。

更新笔记后，在仓库根目录执行：

```sh
npm ci
npm run build
npm run preview
```

预览地址为 `http://127.0.0.1:4173/learning.html`。构建生成 `learning.html` 和各篇静态 HTML；发布时连同 Markdown、图片、数学字体一起提交即可，浏览器不需要执行 Markdown 或公式渲染脚本。

原始 Obsidian 文件没有修改。`pi 0.5.md` 引用的 `Pasted image 20260721142412.png` 在原笔记目录和整个 Vault 中均未找到，网页保留“配图待补”说明。找到后，将图片加入 `images/`，替换 `notes/pi05.md` 中对应的说明，再重新构建。
