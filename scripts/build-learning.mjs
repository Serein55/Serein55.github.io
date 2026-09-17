import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Marked, Renderer } from 'marked';
import katex from 'katex';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const shell = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const escape = text => String(text).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));

// 站内品牌名：Learning 各页面统一用完整姓名，不跟随首页的简写。
const BRAND = 'Darren Gan';

// 每个分类一个目录：notes.json 存 Markdown 笔记，readings.json 存 PDF 讲义。
const collections = [
  { id: 'embodied-ai', title: 'Embodied AI', unit: '篇笔记', type: 'notes' },
  {
    id: 'reinforcement-learning',
    title: '强化学习',
    unit: '份讲义',
    type: 'readings',
  },
];

function page(title, description, content, depth = 0, head = '') {
  const prefix = '../'.repeat(depth);
  return shell
    .replace(/<title>.*?<\/title>/, `<title>Darren Gan · ${escape(title)}</title>\n<meta name="description" content="${escape(description)}">`)
    .replace(/(<a class="brand"[^>]*>)[^<]*(<\/a>)/, `$1${BRAND}$2`)
    .replace('href="index.html" class="active"', 'href="index.html"')
    .replace('href="learning.html"', 'href="learning.html" class="active" aria-current="page"')
    .replace(/href="((?:index|research|learning|life)\.html|style\.css)"/g, `href="${prefix}$1"`)
    .replace('</head>', `<link rel="stylesheet" href="${prefix}learning.css">\n${head}</head>`)
    .replace(/<main class="wrap">[\s\S]*?<\/main>/, () => content);
}

// Protect code before extracting TeX, including fenced code inside blockquotes.
// Math is extracted before Markdown so pipes and underscores remain intact.
function prepareMath(markdown) {
  const code = [];
  let text = markdown.replace(/^([ \t>]*)(`{3,}|~{3,})[^\n]*\n[\s\S]*?^\1\2[ \t]*$/gm, value => `NOTECODE${code.push(value) - 1}END`);
  text = text.replace(/(`+)[\s\S]*?\1/g, value => `NOTECODE${code.push(value) - 1}END`);
  const math = [];
  text = text.replace(/\$\$([\s\S]+?)\$\$|\$([^\n$]+?)\$/g, (source, block, inline) => {
    const display = block !== undefined;
    const tex = (block ?? inline).trim();
    const output = katex.renderToString(tex, { displayMode: display, throwOnError: true, strict: 'ignore', trust: false, output: 'htmlAndMathml' });
    const index = math.push({ source, display, output }) - 1;
    return display ? `\n\nNOTEMATH${index}END\n\n` : `NOTEMATH${index}END`;
  });
  text = text.replace(/NOTECODE(\d+)END/g, (_, index) => code[Number(index)]);
  return { text, math };
}

function renderNote(collection, note) {
  const source = fs.readFileSync(path.join(root, collection.dir, 'notes', `${note.slug}.md`), 'utf8');
  const { text, math } = prepareMath(source.replace(/^# .+\n/, ''));
  const headings = [];
  const usedIds = new Set();
  const renderer = new Renderer();
  const defaultTable = renderer.table;
  const defaultParagraph = renderer.paragraph;
  let codeBlocks = 0;
  let images = 0;
  renderer.heading = function ({ tokens, depth }) {
    const html = this.parser.parseInline(tokens);
    const label = html.replace(/<[^>]*>/g, '').replace(/NOTEMATH(\d+)END/g, (_, i) => math[Number(i)].source.replace(/\$/g, ''));
    const base = label.toLowerCase().replace(/&[^;]+;/g, '').replace(/[^\p{L}\p{N}\s-]/gu, '').trim().replace(/\s+/g, '-') || 'section';
    let id = base, count = 2;
    while (usedIds.has(id)) id = `${base}-${count++}`;
    usedIds.add(id);
    headings.push({ id, label, depth });
    return `<h${depth} id="${id}">${html}</h${depth}>\n`;
  };
  renderer.html = ({ text }) => escape(text);
  renderer.code = ({ text, lang }) => {
    codeBlocks++;
    const language = (lang || 'text').split(/\s/)[0];
    const lines = text.split('\n').length;
    const code = `<pre tabindex="0" aria-label="${escape(language)} 代码"><code>${escape(text)}</code></pre>`;
    return lines > 60
      ? `<details class="code-fold"><summary>展开代码 <span>${escape(language)} · ${lines} 行</span></summary>${code}</details>\n`
      : `<div class="code-block"><div class="code-label">${escape(language)}</div>${code}</div>\n`;
  };
  renderer.table = function (token) { return `<div class="table-scroll" role="region" tabindex="0" aria-label="笔记表格，可横向滚动">${defaultTable.call(this, token)}</div>\n`; };
  renderer.image = ({ href, text }) => {
    if (!href.startsWith('../images/')) throw new Error(`Unexpected image path: ${href}`);
    const file = path.resolve(root, collection.dir, 'notes', href);
    if (!fs.existsSync(file)) throw new Error(`Missing image: ${file}`);
    images++;
    const url = `images/${path.basename(file)}`;
    return `<figure><a href="${url}" target="_blank" rel="noopener" aria-label="查看原图：${escape(text)}"><img src="${url}" alt="${escape(text)}" loading="lazy" decoding="async"></a><figcaption>${escape(text)} · <a href="${url}" target="_blank" rel="noopener">查看原图</a></figcaption></figure>`;
  };
  renderer.paragraph = function (token) {
    // A standalone figure is a block, and must not be nested inside a paragraph.
    if (token.tokens.filter(t => t.type !== 'space').length === 1 && token.tokens[0].type === 'image') return this.parser.parseInline(token.tokens) + '\n';
    return defaultParagraph.call(this, token);
  };
  renderer.link = function ({ href, title, tokens }) {
    if (!/^(?:https?:\/\/|#|\.\.\/images\/)/i.test(href)) throw new Error(`Unexpected link: ${href}`);
    return `<a href="${escape(href)}"${title ? ` title="${escape(title)}"` : ''}>${this.parser.parseInline(tokens)}</a>`;
  };
  let html = new Marked({ gfm: true, renderer }).parse(text);
  html = html.replace(/<p>NOTEMATH(\d+)END<\/p>/g, (_, i) => `<div class="math-block" tabindex="0" role="region" aria-label="数学公式，可横向滚动">${math[Number(i)].output}</div>`);
  html = html.replace(/NOTEMATH(\d+)END/g, (_, i) => math[Number(i)].output);
  if (/NOTE(?:MATH|CODE)\d+END|!\[\[/.test(html)) throw new Error(`Unresolved markup in ${note.slug}`);
  return { html, headings, codeBlocks, images, formulas: math.length };
}

function notePage(collection, note, rendered, previous, next) {
  const toc = rendered.headings.filter(heading => heading.depth === 2);
  return `<main class="wrap note-page" id="top">
  <header class="note-header">
    <div class="breadcrumb"><a href="../../learning.html">Learning</a><span aria-hidden="true">/</span><a href="../../learning.html#${collection.id}">${escape(collection.title)}</a></div>
    <p class="eyebrow">${escape(note.category)}</p>
    <h1>${escape(note.title)}</h1>
    <p class="note-description">${escape(note.description)}</p>
  </header>
  <details class="note-toc" open>
    <summary>本文目录 <span>${toc.length} 个章节</span></summary>
    <ol>${toc.map(h => `<li><a href="#${h.id}">${h.label}</a></li>`).join('\n')}</ol>
  </details>
  <article class="note-content" aria-label="${escape(note.title)}">${rendered.html}</article>
  <footer class="note-footer">
    <div class="note-tools"><a href="../../learning.html#${collection.id}">← 全部笔记</a><a href="notes/${note.slug}.md" download>下载 Markdown</a><a href="#top">回到顶部 ↑</a></div>
    <div class="note-pagination">${previous ? `<a href="${previous.slug}.html"><span>上一篇</span>${escape(previous.title)}</a>` : '<span></span>'}${next ? `<a href="${next.slug}.html"><span>下一篇</span>${escape(next.title)}</a>` : ''}</div>
  </footer>
</main>`;
}

// PDF 讲义阅读页：外壳交给 reader.mjs，页面只提供文件、页码与目录容器。
function readingPage(collection, reading) {
  const file = escape(reading.file);
  return `<main class="wrap note-page" id="top">
  <header class="note-header">
    <div class="breadcrumb"><a href="../../learning.html">Learning</a><span aria-hidden="true">/</span><a href="../../learning.html#${collection.id}">${escape(collection.title)}</a></div>
    <p class="eyebrow">${escape(reading.category || 'PDF 讲义')}</p>
    <h1>${escape(reading.title)}</h1>
    <p class="note-description">${escape(reading.description)}</p>
  </header>
  <div class="reader" data-reader data-file="${file}" data-slug="${escape(reading.slug)}">
    <div class="reader-bar" role="toolbar" aria-label="讲义阅读工具栏">
      <button type="button" class="reader-btn" data-action="toc" aria-expanded="false" aria-controls="reader-toc">目录</button>
      <div class="reader-group">
        <button type="button" class="reader-btn" data-action="prev" aria-label="上一页">←</button>
        <label class="reader-pager"><span class="reader-sr">页码</span><input type="text" inputmode="numeric" value="1" data-page-input aria-label="页码"><span class="reader-total">/ <span data-page-total>–</span></span></label>
        <button type="button" class="reader-btn" data-action="next" aria-label="下一页">→</button>
      </div>
      <div class="reader-group">
        <button type="button" class="reader-btn" data-action="zoom-out" aria-label="缩小">−</button>
        <span class="reader-zoom" data-zoom-label>100%</span>
        <button type="button" class="reader-btn" data-action="zoom-in" aria-label="放大">＋</button>
        <button type="button" class="reader-btn" data-action="fit" aria-pressed="true">适应宽度</button>
      </div>
      <div class="reader-group reader-group-end">
        <button type="button" class="reader-btn" data-action="zen" aria-pressed="false">专注阅读</button>
        <a class="reader-btn" href="${file}" download>下载 PDF</a>
        <a class="reader-btn" href="${file}" target="_blank" rel="noopener">新窗口打开</a>
      </div>
    </div>
    <div class="reader-progress" aria-hidden="true"><span data-progress></span></div>
    <div class="reader-body">
      <aside class="reader-toc" id="reader-toc" data-toc aria-label="讲义目录">
        <div class="reader-toc-head"><span>书签目录</span><button type="button" class="reader-toc-close" data-action="toc-close" aria-label="关闭目录">×</button></div>
        <nav class="reader-outline" data-outline><p class="reader-empty">正在读取目录…</p></nav>
      </aside>
      <div class="reader-backdrop" data-backdrop></div>
      <div class="reader-view" data-view tabindex="0" aria-label="讲义正文">
        <p class="reader-status" data-status>正在加载讲义…</p>
        <div class="reader-pages" data-pages></div>
      </div>
    </div>
    <p class="reader-hint" data-hint hidden></p>
    <noscript><p class="reader-hint">当前浏览器未启用 JavaScript，可<a href="${file}">直接打开 PDF</a>阅读。</p></noscript>
  </div>
  <footer class="note-footer">
    <div class="note-tools"><a href="../../learning.html#${collection.id}">← 全部讲义</a><a href="${file}" download>下载 PDF</a><a href="#top">回到顶部 ↑</a></div>
  </footer>
</main>`;
}

function entry(index, href, title, description, badge) {
  return `      <li><a class="note-entry" href="${href}"><span class="note-number">${String(index + 1).padStart(2, '0')}</span><div><span class="note-category">${escape(badge || '')}</span><h3>${escape(title)}</h3><p>${escape(description)}</p></div><span class="note-arrow" aria-hidden="true">↗</span></a></li>`;
}

function section(collection, count, items) {
  return `  <details class="note-collection" id="${collection.id}">
    <summary class="collection-heading"><h2>${escape(collection.title)}</h2><span>${count} ${escape(collection.unit)}</span></summary>
${collection.description ? `    <p class="collection-description">${escape(collection.description)}</p>\n` : ''}    <ol class="note-list">
${items.join('\n')}
    </ol>
  </details>`;
}

const summary = [];

for (const collection of collections) {
  collection.dir = path.join('learning', collection.id);
  const dir = path.join(root, collection.dir);
  const manifest = JSON.parse(fs.readFileSync(path.join(dir, `${collection.type}.json`), 'utf8'));

  if (collection.type === 'notes') {
    let totalFormulas = 0;
    const items = [];
    for (let index = 0; index < manifest.length; index++) {
      const note = manifest[index];
      const rendered = renderNote(collection, note);
      totalFormulas += rendered.formulas;
      const html = page(note.title, note.description, notePage(collection, note, rendered, manifest[index - 1], manifest[index + 1]), 2, `<link rel="stylesheet" href="../../assets/katex/katex.min.css">\n`);
      fs.writeFileSync(path.join(dir, `${note.slug}.html`), html);
      items.push(entry(index, `learning/${collection.id}/${note.slug}.html`, note.title, note.description, note.badge));
      console.log(`${note.slug}: ${rendered.headings.length} headings, ${rendered.formulas} formulas, ${rendered.codeBlocks} code blocks, ${rendered.images} images`);
    }
    summary.push(section(collection, manifest.length, items));
    console.log(`Built ${manifest.length} notes; ${totalFormulas} formulas rendered locally.`);
    continue;
  }

  const items = [];
  for (let index = 0; index < manifest.length; index++) {
    const reading = manifest[index];
    const file = path.join(dir, reading.file);
    if (!fs.existsSync(file)) throw new Error(`Missing PDF: ${file}`);
    if (!reading.pages) throw new Error(`Missing page count for ${reading.slug}: add "pages" to readings.json`);
    const html = page(reading.title, reading.description, readingPage(collection, reading), 2, `<link rel="stylesheet" href="../../reader.css">\n<script type="module" src="../../reader.mjs"></script>\n`);
    fs.writeFileSync(path.join(dir, `${reading.slug}.html`), html);
    items.push(entry(index, `learning/${collection.id}/${reading.slug}.html`, reading.title, reading.description, `PDF · ${reading.pages} 页`));
    console.log(`${reading.slug}: ${reading.pages} pages, ${(fs.statSync(file).size / 1024).toFixed(0)} KB`);
  }
  summary.push(section(collection, manifest.length, items));
}

const main = `<main class="wrap learning-page">
  <h1>Learning</h1>
${summary.join('\n')}
</main>`;
fs.writeFileSync(path.join(root, 'learning.html'), page('Learning', '具身智能与强化学习的学习笔记：模型架构、论文阅读与讲义整理。', main));

// Commit the generated CSS, fonts and reader so GitHub Pages needs no runtime or CDN.
function vendor(from, to) {
  fs.mkdirSync(path.dirname(path.join(root, to)), { recursive: true });
  fs.copyFileSync(path.join(root, from), path.join(root, to));
}
vendor('node_modules/katex/dist/katex.min.css', 'assets/katex/katex.min.css');
fs.cpSync(path.join(root, 'node_modules/katex/dist/fonts'), path.join(root, 'assets/katex/fonts'), { recursive: true });
vendor('node_modules/katex/LICENSE', 'assets/katex/LICENSE');
vendor('node_modules/pdfjs-dist/build/pdf.min.mjs', 'assets/pdfjs/pdf.min.mjs');
vendor('node_modules/pdfjs-dist/build/pdf.worker.min.mjs', 'assets/pdfjs/pdf.worker.min.mjs');
vendor('node_modules/pdfjs-dist/LICENSE', 'assets/pdfjs/LICENSE');

console.log(`Built ${collections.length} collections and Learning index.`);
