import { setupGuitarAudio } from './guitar-audio.mjs';
import { muscleUpDuration, muscleUpPose, armPose, torsoLean, bindMuscleUpGesture } from './robot-motion.mjs';

const studies = document.querySelector('.motion-studies');

if (studies) {
  const find = selector => studies.querySelector(selector);
  const body = find('#robot-body');
  const robot = find('.robot-play');
  const shadow = find('#robot-shadow');
  const arms = ['left', 'right'].map(side => ({
    paths: studies.querySelectorAll(`[data-arm="${side}"]`),
    elbow: find(`#${side}-elbow`),
    shoulder: find(`#${side}-shoulder`),
  }));
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
  let pullTime = 0;
  let currentLift = 0;
  let muscleUp = null;

  function draw(time) {
    // armPose 里肩关节带躯干后仰，body 必须同步旋转，否则肩部圆点会与躯干脱节。
    const pose = muscleUp ? muscleUpPose(muscleUp.time, muscleUp.startLift, muscleUp.returnLift) : null;
    // 空闲摆动保持在 [0, 44]：不触发锁定姿态，且不越过 maxLift=76 的可达边界。
    const lift = pose ? pose.lift : 22 * (1 - Math.cos(pullTime * Math.PI * 2 / 5.6));
    currentLift = lift;
    if (pose) robot.dataset.phase = pose.phase;
    const lean = torsoLean(lift);
    body.setAttribute('transform', `translate(0 ${-lift}) rotate(${lean} 160 210)`);
    shadow.setAttribute('rx', 58 - lift * .12);
    arms.forEach((arm, index) => {
      const { sx, sy, hx, hy, ex, ey } = armPose(index, lift);
      arm.paths.forEach(path => path.setAttribute('d', `M${sx} ${sy}L${ex} ${ey}L${hx} ${hy}`));
      arm.elbow.setAttribute('cx', ex);
      arm.elbow.setAttribute('cy', ey);
      arm.shoulder.setAttribute('cx', sx);
      arm.shoulder.setAttribute('cy', sy);
    });

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
    if (muscleUp) muscleUp.time += delta;
    else pullTime += delta;
    previous = now;
    draw(elapsed);
    if (muscleUp && muscleUp.time >= muscleUpDuration) {
      paused = muscleUp.restorePaused;
      if (!paused) pullTime = 0;
      muscleUp = null;
      delete robot.dataset.action;
      delete robot.dataset.phase;
      syncPlayback();
      return;
    }
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
    if (muscleUp) {
      muscleUp.restorePaused = false;
      muscleUp.returnLift = 0;
    }
    syncPlayback();
  });
  reducedMotion.addEventListener('change', () => { paused = reducedMotion.matches; syncPlayback(); });
  document.addEventListener('visibilitychange', syncPlayback);
  const cancelHold = bindMuscleUpGesture(robot, () => {
    if (muscleUp || document.hidden || !visible) return;
    muscleUp = { time: 0, startLift: currentLift, returnLift: paused ? currentLift : 0, restorePaused: paused };
    robot.dataset.action = 'muscle-up';
    paused = false;
    syncPlayback();
  });
  const observer = new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (!visible) cancelHold();
    syncPlayback();
  });
  observer.observe(studies);
  draw(0);
  syncPlayback();
  setupGuitarAudio(studies);
}
