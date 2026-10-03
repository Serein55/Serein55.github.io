export const muscleUpDuration = 4.2;

export function muscleUpPose(time, startLift, returnLift = 0) {
  const poses = [
    [0, startLift, 'pull'],
    [.65, 84, 'transition'],
    [1.15, 132, 'press'],
    [1.8, 188, 'support'],
    [2.45, 188, 'lower'],
    [3.2, 105, 'lower'],
    [muscleUpDuration, returnLift, 'complete'],
  ];
  for (let i = 1; i < poses.length; i++) {
    const [end, to] = poses[i];
    const [start, from, phase] = poses[i - 1];
    if (time <= end) {
      const progress = Math.max(0, (time - start) / (end - start));
      const eased = progress * progress * (3 - 2 * progress);
      return { lift: from + (to - from) * eased, phase };
    }
  }
  return { lift: returnLift, phase: 'complete' };
}

export function armPose(index, lift) {
  const sx = index ? 184 : 136;
  const sy = 155 - lift;
  const hx = index ? 212 : 108;
  const hy = 65;
  const dx = hx - sx;
  const dy = hy - sy;
  const distance = Math.hypot(dx, dy);
  const bend = Math.sqrt(Math.max(0, 52 ** 2 - (distance / 2) ** 2));
  const side = index ? 1 : -1;
  return {
    sx, sy, hx, hy,
    ex: (sx + hx) / 2 + side * (-dy / distance) * bend,
    ey: (sy + hy) / 2 + side * (dx / distance) * bend,
  };
}

export function bindMuscleUpGesture(button, activate) {
  let timer = null;
  let pointer = null;
  let origin = null;
  const clearHold = () => {
    clearTimeout(timer);
    timer = null;
    button.classList.remove('is-holding');
  };
  const cancel = () => { clearHold(); pointer = null; origin = null; };
  button.disabled = false;
  button.addEventListener('pointerdown', event => {
    if (!event.isPrimary || event.button !== 0 || pointer !== null) return;
    pointer = event.pointerId;
    origin = [event.clientX, event.clientY];
    button.setPointerCapture(pointer);
    button.classList.add('is-holding');
    timer = setTimeout(() => { clearHold(); activate(); }, 550);
  });
  button.addEventListener('pointermove', event => {
    if (event.pointerId === pointer && origin && Math.hypot(event.clientX - origin[0], event.clientY - origin[1]) > 12) clearHold();
  });
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    button.addEventListener(type, event => { if (event.pointerId === pointer) cancel(); });
  }
  button.addEventListener('contextmenu', event => event.preventDefault());
  // Native keyboard/assistive activation has no pointer clicks and does not require holding.
  button.addEventListener('click', event => { if (event.detail === 0) activate(); });
  button.addEventListener('keydown', event => {
    if (event.repeat && (event.key === 'Enter' || event.key === ' ')) event.preventDefault();
  });
  button.addEventListener('blur', cancel);
  window.addEventListener('blur', cancel);
  document.addEventListener('visibilitychange', () => { if (document.hidden) cancel(); });
  return cancel;
}
