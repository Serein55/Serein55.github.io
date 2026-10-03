import { setupGuitarAudio } from './guitar-audio.mjs';

// 首页现在只剩吉他一个动画，引体向上（robot-motion.mjs）已整段移除。
// 这里只驱动吉他：右手拨弦、左手按弦、音波扩散，以及全局播放/暂停。
const studies = document.querySelector('.motion-studies');

if (studies) {
  const find = selector => studies.querySelector(selector);
  const hand = find('#picking-hand');
  const fingers = [...studies.querySelectorAll('[data-finger]')];
  const frets = [...studies.querySelectorAll('[data-fret]')];
  const waves = [find('#sound-wave-1'), find('#sound-wave-2')];
  const toggle = find('.motion-toggle');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let paused = reducedMotion.matches;
  let visible = true;
  let frame = null;
  let previous = null;
  let elapsed = 0;

  function draw(time) {
    const beat = time * Math.PI * 2 / 1.8;
    hand.setAttribute('transform', `rotate(${Math.sin(beat) * 5} 130 191)`);
    fingers.forEach((finger, index) => {
      const curl = Math.sin(beat * 2 - index * .9) * 7;
      finger.setAttribute('transform', `rotate(${curl} ${index ? 151 : 149} ${index ? 202 : 183})`);
    });
    frets.forEach((finger, index) => {
      finger.setAttribute('transform', `translate(0 ${Math.sin(beat / 2 + index * 1.8) * 1.8})`);
    });
    waves.forEach((wave, index) => wave.setAttribute('opacity', .2 + .5 * (1 + Math.sin(beat - index)) / 2));
  }

  function tick(now) {
    const delta = previous === null ? 0 : Math.min((now - previous) / 1000, .05);
    elapsed += delta;
    previous = now;
    draw(elapsed);
    frame = requestAnimationFrame(tick);
  }

  function syncPlayback() {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    previous = null;
    if (!paused && visible && !document.hidden) frame = requestAnimationFrame(tick);
    toggle.setAttribute('aria-label', paused ? '播放动画' : '暂停动画');
    find('.motion-toggle-icon').setAttribute('d', paused ? 'M4 2L13 8L4 14Z' : 'M5 3V13M11 3V13');
  }

  toggle.hidden = false;
  toggle.addEventListener('click', () => {
    paused = !paused;
    syncPlayback();
  });
  reducedMotion.addEventListener('change', () => { paused = reducedMotion.matches; syncPlayback(); });
  document.addEventListener('visibilitychange', syncPlayback);
  const observer = new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    syncPlayback();
  });
  observer.observe(studies);
  draw(0);
  syncPlayback();
  setupGuitarAudio(studies);
}
