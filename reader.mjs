/* 讲义阅读器：用自托管的 PDF.js 在站内阅读 PDF，提供书签目录、连续滚动、缩放与阅读进度记忆。
   页面只需给出 data-reader / data-file / data-slug，其余交给本文件。 */

const MIN_SCALE = 0.4;
const MAX_SCALE = 3.2;
const MAX_RENDERED = 10;
const PIXEL_RATIO = Math.min(window.devicePixelRatio || 1, 2);

document.querySelectorAll('[data-reader]').forEach(root => boot(root));

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function element(tag, attributes = {}, text) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

async function boot(root) {
  const ui = {
    view: root.querySelector('[data-view]'),
    pages: root.querySelector('[data-pages]'),
    status: root.querySelector('[data-status]'),
    hint: root.querySelector('[data-hint]'),
    toc: root.querySelector('[data-toc]'),
    backdrop: root.querySelector('[data-backdrop]'),
    outline: root.querySelector('[data-outline]'),
    pageInput: root.querySelector('[data-page-input]'),
    pageTotal: root.querySelector('[data-page-total]'),
    zoomLabel: root.querySelector('[data-zoom-label]'),
    progress: root.querySelector('[data-progress]'),
    fit: root.querySelector('[data-action="fit"]'),
    prev: root.querySelector('[data-action="prev"]'),
    next: root.querySelector('[data-action="next"]'),
  };

  const storageKey = `reader:${root.dataset.slug || 'pdf'}`;
  const saved = readSaved(storageKey);
  const records = [];
  let outlineEntries = [];
  let activeOutline = null;
  let pdfjs;
  let doc;
  let total = 0;
  let current = 0;
  let scale = saved?.scale ?? 1;
  let fit = saved?.fit ?? true;
  let maxBaseWidth = 1;
  let frameRequest = 0;
  const visible = new Set();

  const fail = (message, error) => {
    if (error) console.warn(message, error);
    ui.status.textContent = message;
    ui.status.hidden = false;
    if (error) {
      ui.hint.innerHTML = '';
      ui.hint.append('也可以', link(root.dataset.file, '直接打开 PDF'), '阅读。');
      ui.hint.hidden = false;
    }
  };

  function link(href, text, attributes = {}) {
    return element('a', { href, ...attributes }, text);
  }

  // ---------- 加载 ----------
  try {
    pdfjs = await import('./assets/pdfjs/pdf.min.mjs');
    pdfjs.GlobalWorkerOptions.workerSrc = new URL('./assets/pdfjs/pdf.worker.min.mjs', import.meta.url).href;
  } catch (error) {
    fail('阅读组件加载失败，请刷新页面重试。', error);
    return;
  }

  try {
    doc = await pdfjs.getDocument({ url: new URL(root.dataset.file, document.baseURI).href }).promise;
  } catch (error) {
    fail(location.protocol === 'file:'
      ? '浏览器不允许直接以文件方式打开时读取 PDF，请用本地服务器预览（npm run preview）或直接打开 PDF。'
      : '讲义加载失败，请检查网络后刷新重试。', error);
    return;
  }

  total = doc.numPages;
  ui.pageTotal.textContent = String(total);

  // ---------- 页面骨架 ----------
  const byWrapper = new Map();
  for (let number = 1; number <= total; number++) {
    const page = await doc.getPage(number);
    const base = page.getViewport({ scale: 1 });
    const wrapper = element('div', { class: 'pdf-page' });
    const sheet = element('div', { class: 'pdf-sheet', 'data-page': String(number), 'data-rendered': 'false' });
    const canvas = element('canvas');
    const layer = element('div', { class: 'textLayer' });
    sheet.append(canvas, layer);
    wrapper.append(sheet, element('span', { class: 'pdf-page-label', 'aria-hidden': 'true' }, String(number)));
    ui.pages.append(wrapper);
    const record = { number, page, base, wrapper, sheet, canvas, layer, rendered: 0, pending: false, task: null, textLayer: null };
    records.push(record);
    byWrapper.set(wrapper, record);
    maxBaseWidth = Math.max(maxBaseWidth, base.width);
  }

  // ---------- 排版与渲染 ----------
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      const record = byWrapper.get(entry.target);
      if (!record) continue;
      if (entry.isIntersecting) visible.add(record.number);
      else visible.delete(record.number);
    }
    pump();
  }, { root: ui.view, rootMargin: '700px 0px' });
  records.forEach(record => observer.observe(record.wrapper));

  function contentWidth() {
    const styles = getComputedStyle(ui.view);
    const padding = parseFloat(styles.paddingLeft) + parseFloat(styles.paddingRight);
    return Math.max(240, ui.view.clientWidth - padding);
  }

  function layout() {
    if (fit) scale = clamp(contentWidth() / maxBaseWidth, MIN_SCALE, MAX_SCALE);
    scale = Math.round(scale * 1000) / 1000;
    for (const record of records) {
      record.wrapper.style.setProperty('--sheet-w', `${(record.base.width * scale).toFixed(2)}px`);
      record.wrapper.style.setProperty('--sheet-h', `${(record.base.height * scale).toFixed(2)}px`);
      record.wrapper.style.setProperty('--scale-factor', String(scale));
      if (record.rendered && record.rendered !== scale) drop(record);
    }
    ui.zoomLabel.textContent = `${Math.round(scale * 100)}%`;
    ui.fit.setAttribute('aria-pressed', fit ? 'true' : 'false');
    pump();
  }

  function pump() {
    const order = [...visible].sort((a, b) => Math.abs(a - current) - Math.abs(b - current));
    for (const number of order) {
      const record = records[number - 1];
      if (record && record.rendered !== scale) render(record);
    }
    evict();
  }

  function render(record) {
    if (record.pending) return;
    const target = scale;
    const viewport = record.page.getViewport({ scale: target });
    const canvas = record.canvas;
    canvas.width = Math.floor(viewport.width * PIXEL_RATIO);
    canvas.height = Math.floor(viewport.height * PIXEL_RATIO);
    const context = canvas.getContext('2d', { alpha: false });
    record.pending = true;
    const task = record.page.render({
      canvasContext: context,
      viewport,
      transform: PIXEL_RATIO === 1 ? null : [PIXEL_RATIO, 0, 0, PIXEL_RATIO, 0, 0],
    });
    record.task = task;
    task.promise.then(async () => {
      record.pending = false;
      record.task = null;
      if (record.rendered === target) return;
      record.rendered = target;
      record.sheet.dataset.rendered = 'true';
      record.sheet.classList.remove('is-failed');
      await paintText(record, viewport);
      evict();
    }, error => {
      record.pending = false;
      record.task = null;
      if (error && error.name === 'RenderingCancelledException') return;
      record.sheet.classList.add('is-failed');
      console.warn(`第 ${record.number} 页渲染失败`, error);
    });
  }

  async function paintText(record, viewport) {
    try {
      const content = await record.page.getTextContent();
      record.layer.replaceChildren();
      const layer = new pdfjs.TextLayer({ textContentSource: content, container: record.layer, viewport });
      await layer.render();
      record.textLayer = layer;
    } catch {
      // 文本层只影响选中与检索，失败时保留画布即可。
    }
  }

  function drop(record) {
    if (record.task) {
      record.task.cancel();
      record.task = null;
    }
    if (record.textLayer) {
      record.textLayer.cancel();
      record.textLayer = null;
    }
    record.pending = false;
    record.rendered = 0;
    record.layer.replaceChildren();
    record.canvas.width = 0;
    record.canvas.height = 0;
    record.sheet.dataset.rendered = 'false';
    record.sheet.classList.remove('is-failed');
  }

  function evict() {
    const done = records.filter(record => record.rendered);
    if (done.length <= MAX_RENDERED) return;
    done.sort((a, b) => Math.abs(b.number - current) - Math.abs(a.number - current));
    for (const record of done.slice(0, done.length - MAX_RENDERED)) {
      if (!visible.has(record.number)) drop(record);
    }
  }

  // ---------- 位置与进度 ----------
  function goTo(number, smooth = false) {
    const record = records[clamp(Math.round(number), 1, total) - 1];
    const target = record.wrapper.getBoundingClientRect().top - ui.view.getBoundingClientRect().top + ui.view.scrollTop - 4;
    ui.view.scrollTo({ top: Math.max(0, target), behavior: smooth ? 'smooth' : 'auto' });
    if (!smooth) setCurrent(record.number);
  }

  function anchor() {
    const record = records[current - 1] || records[0];
    const box = record.wrapper.getBoundingClientRect();
    const frame = ui.view.getBoundingClientRect();
    return { number: record.number, ratio: box.height ? clamp((frame.top - box.top) / box.height, 0, 1) : 0 };
  }

  function restoreAnchor(position) {
    const record = records[position.number - 1];
    if (!record) return;
    const box = record.wrapper.getBoundingClientRect();
    const frame = ui.view.getBoundingClientRect();
    ui.view.scrollTop += (box.top - frame.top) + position.ratio * box.height;
  }

  function track() {
    const frame = ui.view.getBoundingClientRect();
    const probe = frame.top + frame.height * 0.3;
    let best = 1;
    let distance = Infinity;
    for (const record of records) {
      const box = record.wrapper.getBoundingClientRect();
      if (box.top <= probe && box.bottom >= probe) {
        best = record.number;
        break;
      }
      const gap = Math.min(Math.abs(box.top - probe), Math.abs(box.bottom - probe));
      if (gap < distance) {
        distance = gap;
        best = record.number;
      }
    }
    setCurrent(best);
  }

  function setCurrent(number) {
    if (number === current) return;
    current = number;
    if (document.activeElement !== ui.pageInput) ui.pageInput.value = String(number);
    ui.progress.style.width = `${(number / total) * 100}%`;
    ui.prev.disabled = number <= 1;
    ui.next.disabled = number >= total;
    remember();
    try {
      history.replaceState(null, '', `#page=${number}`);
    } catch {
      // 某些沙箱环境禁止改动地址栏，忽略即可。
    }
    markOutline(number);
  }

  function remember() {
    try {
      localStorage.setItem(storageKey, JSON.stringify({ page: current, scale, fit }));
    } catch {
      // 隐私模式下无法写入，不影响阅读。
    }
  }

  // ---------- 书签目录 ----------
  async function destPage(dest) {
    try {
      let target = dest;
      if (typeof target === 'string') target = await doc.getDestination(target);
      if (!Array.isArray(target) || !target.length) return 1;
      const ref = target[0];
      const index = ref && typeof ref === 'object' ? await doc.getPageIndex(ref) : Number(ref);
      return Number.isFinite(index) ? clamp(index + 1, 1, total) : 1;
    } catch {
      return 1;
    }
  }

  async function buildOutline() {
    let items = null;
    try {
      items = await doc.getOutline();
    } catch {
      items = null;
    }
    if (!items || !items.length) {
      ui.outline.replaceChildren(element('p', { class: 'reader-empty' }, '这份讲义没有书签目录。'));
      return;
    }
    ui.outline.replaceChildren(await outlineList(items, 1));
  }

  async function outlineList(items, level) {
    const list = element('ol', { class: 'reader-outline-list' });
    const pages = await Promise.all(items.map(item => destPage(item.dest)));
    for (let index = 0; index < items.length; index++) {
      const item = items[index];
      const page = pages[index];
      const entry = element('li', { class: `reader-outline-level-${Math.min(level, 3)}` });
      const anchor = link(`#page=${page}`, item.title || `第 ${page} 页`, { title: item.title || '' });
      entry.append(anchor);
      const target = { page, link: anchor };
      outlineEntries.push(target);
      anchor.addEventListener('click', event => {
        event.preventDefault();
        goTo(page, true);
        setToc(false);
      });
      if (item.items && item.items.length) entry.append(await outlineList(item.items, level + 1));
      list.append(entry);
    }
    return list;
  }

  function markOutline(number) {
    // 同一页上可能压着好几个标题，取页码最大的一项；并列时取更靠前的一项，避免高亮跳到家。
    let active = null;
    for (const entry of outlineEntries) {
      if (entry.page > number) continue;
      if (!active || entry.page > active.page) active = entry;
    }
    if (active === activeOutline) return;
    activeOutline = active;
    for (const entry of outlineEntries) entry.link.classList.toggle('is-current', entry === active);
    if (active && ui.toc) reveal(active.link);
  }

  function reveal(node) {
    const box = node.getBoundingClientRect();
    const frame = ui.toc.getBoundingClientRect();
    if (box.top < frame.top + 48) ui.toc.scrollTop -= frame.top + 48 - box.top;
    else if (box.bottom > frame.bottom - 16) ui.toc.scrollTop += box.bottom - frame.bottom + 16;
  }

  // ---------- 交互 ----------
  function setFit(next) {
    fit = next;
    relayout();
  }

  function zoomBy(factor) {
    fit = false;
    scale = clamp(scale * factor, MIN_SCALE, MAX_SCALE);
    relayout();
  }

  function relayout() {
    const position = anchor();
    layout();
    restoreAnchor(position);
    remember();
  }

  function setZen(next) {
    if (next) root.dataset.zen = 'true';
    else delete root.dataset.zen;
    root.querySelector('[data-action="zen"]').setAttribute('aria-pressed', next ? 'true' : 'false');
    root.querySelector('[data-action="zen"]').textContent = next ? '退出专注' : '专注阅读';
    if (next) document.documentElement.dataset.readerZen = 'true';
    else delete document.documentElement.dataset.readerZen;
    setToc(false);
    relayout();
  }

  function setToc(next) {
    if (next) root.dataset.tocOpen = 'true';
    else delete root.dataset.tocOpen;
    root.querySelector('[data-action="toc"]').setAttribute('aria-expanded', next ? 'true' : 'false');
  }

  root.querySelector('.reader-bar').addEventListener('click', event => {
    const button = event.target.closest('[data-action]');
    if (!button) return;
    const action = button.dataset.action;
    if (action === 'prev') goTo(current - 1, true);
    else if (action === 'next') goTo(current + 1, true);
    else if (action === 'zoom-in') zoomBy(1.15);
    else if (action === 'zoom-out') zoomBy(1 / 1.15);
    else if (action === 'fit') setFit(fit ? false : true);
    else if (action === 'zen') setZen(root.dataset.zen !== 'true');
    else if (action === 'toc') setToc(root.dataset.tocOpen !== 'true');
    else if (action === 'toc-close') setToc(false);
  });

  ui.backdrop?.addEventListener('click', () => setToc(false));

  ui.pageInput.addEventListener('change', () => {
    const value = Number.parseInt(ui.pageInput.value, 10);
    if (Number.isFinite(value)) goTo(value);
    else ui.pageInput.value = String(current);
  });

  ui.pageInput.addEventListener('keydown', event => {
    if (event.key === 'Enter') {
      event.preventDefault();
      ui.pageInput.blur();
    }
  });

  ui.view.addEventListener('scroll', () => {
    if (frameRequest) return;
    frameRequest = requestAnimationFrame(() => {
      frameRequest = 0;
      track();
    });
  }, { passive: true });

  document.addEventListener('keydown', event => {
    if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return;
    const target = event.target;
    if (target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))) return;
    if (event.key === 'ArrowLeft') { event.preventDefault(); goTo(current - 1, true); }
    else if (event.key === 'ArrowRight') { event.preventDefault(); goTo(current + 1, true); }
    else if (event.key === '+' || event.key === '=') { event.preventDefault(); zoomBy(1.15); }
    else if (event.key === '-' || event.key === '_') { event.preventDefault(); zoomBy(1 / 1.15); }
    else if (event.key === '0') { event.preventDefault(); setFit(true); }
    else if (event.key === 'Escape') {
      if (root.dataset.zen === 'true') setZen(false);
      else if (root.dataset.tocOpen === 'true') setToc(false);
    }
  });

  let resizeTimer = 0;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(relayout, 160);
  });

  // ---------- 启动 ----------
  layout();
  ui.status.hidden = true;
  const fromHash = Number.parseInt((location.hash.match(/^#page=(\d+)$/) || [])[1], 10);
  const start = Number.isFinite(fromHash) ? fromHash : saved?.page ?? 1;
  if (start > 1) goTo(start);
  else setCurrent(1);
  buildOutline();
}

function readSaved(key) {
  try {
    const value = JSON.parse(localStorage.getItem(key) || 'null');
    return value && typeof value === 'object' ? value : null;
  } catch {
    return null;
  }
}
