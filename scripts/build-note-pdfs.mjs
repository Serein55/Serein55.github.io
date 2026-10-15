import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright';
import { PDFDocument } from 'pdf-lib';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const sourceDir = path.join(root, 'tmp', 'pdfs');
const noteDir = path.join(root, 'learning', 'embodied-ai');
const manifestPath = path.join(noteDir, 'notes.json');
const notes = JSON.parse(await fs.readFile(manifestPath, 'utf8'));

execFileSync(process.execPath, ['scripts/build-learning.mjs', '--pdf-sources'], { cwd: root, stdio: 'inherit' });
const browser = await chromium.launch({ headless: true });
try {
  for (const note of notes) {
    const page = await browser.newPage({ viewport: { width: 794, height: 1123 } });
    await page.goto(pathToFileURL(path.join(sourceDir, `${note.slug}.html`)).href);
    await page.addStyleTag({ path: path.join(root, 'scripts', 'note-print.css') });
    await page.emulateMedia({ media: 'print' });
    await page.evaluate(async () => {
      document.querySelectorAll('details').forEach(node => { node.open = true; });
      document.querySelectorAll('figure a').forEach(link => link.removeAttribute('href'));
      document.querySelectorAll('img').forEach(image => { image.loading = 'eager'; });
      await Promise.all([...document.images].map(image => image.decode()));
      await document.fonts.ready;
      // Keep asset URLs absolute, then restore the document base so Chromium
      // writes TOC links as PDF destinations instead of local file URLs.
      document.querySelectorAll('link[href]').forEach(link => { link.href = link.href; });
      document.querySelectorAll('img').forEach(image => { image.src = image.src; });
      document.querySelector('base').href = window.location.href;
      // Shrink only formulas that exceed the printable column, never crop them.
      for (const block of document.querySelectorAll('.math-block')) {
        const formula = block.querySelector('.katex-html');
        if (!formula) continue;
        const width = formula.getBoundingClientRect().width;
        if (width > block.clientWidth) {
          block.style.fontSize = `${parseFloat(getComputedStyle(block).fontSize) * block.clientWidth / width * 0.98}px`;
        }
      }
    });
    const overflow = await page.evaluate(() => [...document.querySelectorAll('.table-scroll, pre, .math-block')]
      .filter(node => node.scrollWidth > node.clientWidth + 2)
      .map(node => node.textContent.trim().slice(0, 80)));
    if (overflow.length) throw new Error(`${note.slug}: content exceeds printable width: ${overflow.join('; ')}`);
    const bytes = await page.pdf({
      format: 'A4', preferCSSPageSize: true, printBackground: true, outline: true, tagged: true,
      displayHeaderFooter: true,
      headerTemplate: '<div></div>',
      footerTemplate: '<div style="font-size:8px;color:#697a76;width:100%;padding:0 18mm;display:flex;justify-content:space-between"><span>Darren · Embodied AI</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>',
    });
    const pdf = await PDFDocument.load(bytes);
    pdf.setTitle(note.title);
    pdf.setAuthor('Darren');
    pdf.setSubject(note.description);
    note.file = `${note.slug}.pdf`;
    note.pages = pdf.getPageCount();
    await fs.writeFile(path.join(sourceDir, note.file), await pdf.save());
    console.log(`${note.file}: ${note.pages} pages`);
    await page.close();
  }
  // Publish only after every note has exported successfully.
  for (const note of notes) await fs.copyFile(path.join(sourceDir, note.file), path.join(noteDir, note.file));
  await fs.writeFile(manifestPath, JSON.stringify(notes, null, 2) + '\n');
} finally {
  await browser.close();
}
execFileSync(process.execPath, ['scripts/build-learning.mjs'], { cwd: root, stdio: 'inherit' });
