# Serein55.github.io

甘仁达的个人主页 · 数学与应用数学 · 上海交通大学 2024 级

极简清新风格静态站点，托管于 GitHub Pages。

## 排版与字体

全站正文、标题、导航与阅读器界面统一使用**华文中宋**（`Text Song`），中西文同一字体，不再按字符集混排。字族定义在 `style.css` 顶部，字号/颜色等通过 `--font-text`、`--font-code` 两个变量控制。

- 华文中宋为系统字体（Windows/Office 自带的 `STZhongsong`），不随站点分发；未安装时按 `Songti SC` → `Noto Serif CJK SC` → `Source Han Serif SC` → `SimSun` → `serif` 逐字回退。
- 例外：代码块保留 `Consolas` 等宽字体（华文中宋为比例字体，代码缩进无法对齐），其中的中文注释仍走华文中宋；数学公式沿用 `assets/katex/` 的 KaTeX 字体，以免破坏公式度量与间距；PDF 讲义内部字体由 PDF 自身决定。

## Learning 笔记与讲义

Learning 分为两个分类，默认收起，点击分类标题后展开，内容全部由 `scripts/build-learning.mjs` 生成。

### Embodied AI（Markdown 笔记）

收录 4 篇笔记，整理时保留原文、公式与代码，统一标题层级、列表和图片路径，并生成目录及独立阅读页面。

- `learning/embodied-ai/notes/`：整理后的 Markdown，后续直接编辑这些文件。
- `learning/embodied-ai/notes.json`：文章顺序、标题、分类与简介。
- `learning/embodied-ai/images/`：集中保存的 5 张配图。
- `learning.css`：笔记列表和正文排版样式。
- `assets/katex/`：数学排版样式、字体和许可证，随网站一起托管。

### 强化学习（PDF 讲义）

以 PDF 原文收录讲义，页面内置自托管的 PDF.js 阅读器：书签目录、连续滚动、页码跳转、缩放、专注模式与阅读进度记忆。阅读器不依赖 CDN，也不改动原始 PDF。

- `learning/reinforcement-learning/rl-mdp-to-ppo.pdf`：《强化学习：从 MDP 到 PPO》讲义原文，53 页。
- `learning/reinforcement-learning/readings.json`：讲义标题、分类、简介、文件名与页数。
- `reader.mjs` / `reader.css`：所有讲义共用的阅读器逻辑与样式。
- `assets/pdfjs/`：随站点托管的 PDF.js 构建产物与许可证。

新增一份讲义：把 PDF 放进同一目录，在 `readings.json` 里追加一条记录（`slug`、`title`、`description`、`file`、`pages`），再执行 `npm run build`。若想新建分类，在 `scripts/build-learning.mjs` 顶部的 `collections` 中加一项，并让新目录包含 `notes.json` 或 `readings.json`。

更新内容后，在仓库根目录执行：

```sh
npm ci
npm run build
npm run preview
```

预览地址为 `http://127.0.0.1:4173/learning.html`。构建生成 `learning.html`、各篇笔记页面与讲义阅读页；发布时连同 Markdown、PDF、图片、数学字体一起提交即可，浏览器不需要执行 Markdown 或公式渲染脚本。

原始 Obsidian 文件没有修改。`pi 0.5.md` 引用的 `Pasted image 20260721142412.png` 在原笔记目录和整个 Vault 中均未找到，网页保留“配图待补”说明。找到后，将图片加入 `images/`，替换 `notes/pi05.md` 中对应的说明，再重新构建。
