import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Marked, Renderer } from 'marked';
import katex from 'katex';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const collection = path.join(root, 'learning/embodied-ai');
const notes = JSON.parse(fs.readFileSync(path.join(collection, 'notes.json'), 'utf8'));
const shell = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const escape = text => String(text).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));

function page(title, description, content, nested = false) {
  const prefix = nested ? '../../' : '';
  return shell
    .replace(/<title>.*?<\/title>/, `<title>Darren Gan · ${escape(title)}</title>\n<meta name="description" content="${escape(description)}">`)
    .replace('href="index.html" class="active"', 'href="index.html"')
    .replace('href="learning.html"', 'href="learning.html" class="active" aria-current="page"')
    .replace(/href="((?:index|research|learning|life)\.html|style\.css)"/g, `href="${prefix}$1"`)
    .replace('</head>', `<link rel="stylesheet" href="${prefix}learning.css">\n${nested ? `<link rel="stylesheet" href="${prefix}assets/katex/katex.min.css">\n` : ''}</head>`)
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

function renderNote(note) {
  const source = fs.readFileSync(path.join(collection, 'notes', `${note.slug}.md`), 'utf8');
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
    const file = path.resolve(collection, 'notes', href);
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

let totalFormulas = 0;
for (let index = 0; index < notes.length; index++) {
  const note = notes[index];
  const rendered = renderNote(note);
  totalFormulas += rendered.formulas;
  const toc = rendered.headings.filter(heading => heading.depth === 2);
  const previous = notes[index - 1], next = notes[index + 1];
  const main = `<main class="wrap note-page" id="top">
  <header class="note-header">
    <div class="breadcrumb"><a href="../../learning.html">Learning</a><span aria-hidden="true">/</span><a href="../../learning.html#embodied-ai">Embodied AI</a></div>
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
    <div class="note-tools"><a href="../../learning.html#embodied-ai">← 全部笔记</a><a href="notes/${note.slug}.md" download>下载 Markdown</a><a href="#top">回到顶部 ↑</a></div>
    <div class="note-pagination">${previous ? `<a href="${previous.slug}.html"><span>上一篇</span>${escape(previous.title)}</a>` : '<span></span>'}${next ? `<a href="${next.slug}.html"><span>下一篇</span>${escape(next.title)}</a>` : ''}</div>
  </footer>
</main>`;
  fs.writeFileSync(path.join(collection, `${note.slug}.html`), page(note.title, note.description, main, true));
  console.log(`${note.slug}: ${rendered.headings.length} headings, ${rendered.formulas} formulas, ${rendered.codeBlocks} code blocks, ${rendered.images} images`);
}

const main = `<main class="wrap learning-page">
  <h1>Learning</h1>
  <details class="note-collection" id="embodied-ai">
    <summary class="collection-heading"><h2>Embodied AI</h2><span>${notes.length} 篇笔记</span></summary>
    <ol class="note-list">
${notes.map((note, i) => `      <li><a class="note-entry" href="learning/embodied-ai/${note.slug}.html"><span class="note-number">${String(i + 1).padStart(2, '0')}</span><div><h3>${escape(note.title)}</h3><p>${escape(note.description)}</p></div><span class="note-arrow" aria-hidden="true">↗</span></a></li>`).join('\n')}
    </ol>
  </details>
</main>`;
fs.writeFileSync(path.join(root, 'learning.html'), page('Learning', '具身智能学习笔记：模型架构、论文阅读与训练实践。', main));

// Commit the generated CSS and fonts so GitHub Pages needs no runtime or CDN.
const vendor = path.join(root, 'assets/katex');
fs.mkdirSync(vendor, { recursive: true });
fs.copyFileSync(path.join(root, 'node_modules/katex/dist/katex.min.css'), path.join(vendor, 'katex.min.css'));
fs.cpSync(path.join(root, 'node_modules/katex/dist/fonts'), path.join(vendor, 'fonts'), { recursive: true });
fs.copyFileSync(path.join(root, 'node_modules/katex/LICENSE'), path.join(vendor, 'LICENSE'));
console.log(`Built ${notes.length} notes and Learning index; ${totalFormulas} formulas rendered locally.`);
