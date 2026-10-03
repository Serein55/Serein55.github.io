// 引体向上的手臂运动学。
//
// 坐标沿用 index.html 中机器人 SVG 的本地坐标（viewBox 内，y 轴向下）：
// 身体中线 x=160，肩关节基准 (±24, 155)，单杠握点 (±28, 65)。
//
// 手臂是「上臂 + 前臂」两段刚体。给定肩与握点，肘部是两圆交点（离散解），
// 所以肘部方位不能作为连续驱动量——真正连续的自由度是肩关节自身的轨迹。
// 因此这里用标准的垂直平分线法求肘（骨长天然守恒），再通过让肩部沿一条
// 符合人体工学的弧线移动，来得到连续、可信的肘部轨迹。
//
// 两个关键约束：
// 1. 行程上限。肩到单杠只有 shoulderY − barY = 90px，maxLift 必须让肩部
//    停在单杠高度附近。原实现让 lift 涨到 188，把肩拉到单杠上方近 100px，
//    手臂被迫折叠——这是原轨迹不符合人体力学的根因。
// 2. 骨段长度。两段之和须略小于肩—握点的最小距离，否则顶点处两圆无交点、
//    sqrt 负数被截断后肘部会飞出画面。

const bodyMidX = 160;     // 身体中线
const barY = 65;          // 握点（单杠）高度
const shoulderY = 155;    // 静止时肩关节高度
const shoulderX = 24;     // 肩关节相对中线的水平偏移
const gripX = 28;         // 握点相对中线的水平偏移
const upperArm = 42;      // 上臂长：肩 → 肘（略长于前臂，符合人体比例）
const foreArm = 38;       // 前臂长：肘 → 握点

// 顶点行程：肩部升到接近单杠高度即锁定。上限不只是「不超过单杠」——
// 行程再大，两圆夹角变小、法向分量趋零，肘部会在顶点翻到外侧（张开成翅膀）。
// 76px 是全程肘部保持内收的最大值，由数值扫描确定。
const maxLift = 76;

// 肩部水平内收（px）：拉起时肩胛后缩，肩关节略向中线靠拢。
const shoulderTighten = 6;

// 躯干倾角（度）：真实引体向上在中段会有轻微前后晃动，顶点则躯干竖直、
// 仅靠肩胛上提锁定。所以倾角走「先小幅后仰、再回正」的非线性曲线，
// 而非单调增大——单调增大会让顶点像摔倒。
const leanBottom = 1;
const leanMid = 5;

/**
 * 拉起进度 → 姿态参数。
 * @param {number} lift 行程位移（向上为正）
 */
function liftPose(lift) {
  const progress = Math.max(0, Math.min(1, lift / maxLift));
  const eased = progress * progress * (3 - 2 * progress);
  // 倾角在中段达峰、顶点回正：用 sin 形曲线而非线性插值。
  const swing = Math.sin(progress * Math.PI);
  const lean = leanBottom + (leanMid - leanBottom) * swing;
  return { progress, eased, lean };
}

/**
 * 求解单侧手臂的肩、肘、握点坐标。
 *
 * 不变量：|E - S| === upperArm，|H - E| === foreArm。
 *
 * @param {0|1} index 0 = 左臂，1 = 右臂
 * @param {number} lift 行程位移
 */
export function armPose(index, lift) {
  const side = index ? 1 : -1;
  const { progress, eased, lean } = liftPose(lift);

  // 拉起时肩胛后缩，肩关节略向中线收拢。
  const tighten = shoulderX - shoulderTighten * eased;

  // 躯干后仰：绕身体中线 x = bodyMidX 旋转，左右臂对称后倾。
  // 旋转中心若落在髋部中线上，单侧肩会横向漂移，甚至漂到握点外侧导致
  // 两圆无交点、肘部飞掉——必须绕中线旋转。
  const pivotY = shoulderY + 55;
  const radians = lean * Math.PI / 180;
  const offX = side * tighten;
  const offY = shoulderY - pivotY;
  const sx = bodyMidX + offX * Math.cos(radians) - offY * Math.sin(radians);
  const sy = pivotY + offX * Math.sin(radians) + offY * Math.cos(radians) - lift;

  const hx = bodyMidX + side * gripX;
  const hy = barY;

  // 肩 → 握点 的方向与距离。
  const vx = hx - sx;
  const vy = hy - sy;
  const reach = Math.max(1e-3, Math.hypot(vx, vy));
  const ux = vx / reach;
  const uy = vy / reach;

  // 有效握距：双臂构成的两圆只在 [|upperArm−foreArm|, upperArm+foreArm] 内有交点。
  // 超出上界（够不到）收到上界，超出下界（肩与握点太近，两圆分离）收到下界，
  // 否则 sqrt 负数被截成 0 后 along 会爆炸、肘部飞出画面。
  const minSpan = Math.abs(upperArm - foreArm) + 0.5;
  const span = Math.max(minSpan, Math.min(reach, upperArm + foreArm - 0.5));
  const gx = sx + ux * span;
  const gy = sy + uy * span;

  // 垂直平分线法：肘 = 肩 + (投影 + 法向偏移)，两个分量平方和恒为 upperArm²。
  const along = (span * span + upperArm * upperArm - foreArm * foreArm) / (2 * span);
  const bend = Math.sqrt(Math.max(0, upperArm * upperArm - along * along));

  // 法向：取 (uy, −ux)·side 这一支，肘部落在肩的下外侧并向中线收拢，
  // 形成真实引体向上的腋下闭合。另一支 (−uy, ux)·side 会把肘部翻到肩上方
  // 或张成翅膀——原实现正是如此，收拢时形成「帐篷」。
  const nx = uy * side;
  const ny = -ux * side;

  return {
    sx, sy, hx: gx, hy: gy,
    ex: sx + ux * along + nx * bend,
    ey: sy + uy * along + ny * bend,
  };
}

/** 供 intro.mjs 使用的躯干后仰角。 */
export function torsoLean(lift) {
  return liftPose(lift).lean;
}

export const muscleUpDuration = 4.2;

export function muscleUpPose(time, startLift, returnLift = 0) {
  // 行程全部落在 [0, maxLift] 内：肩部升到单杠高度即顶点，不越过单杠，
  // 避免手臂进入「肩在杠上方」的折叠姿态。
  const poses = [
    [0, startLift, 'pull'],
    [.65, 42, 'transition'],
    [1.15, 60, 'press'],
    [1.8, maxLift, 'support'],
    [2.45, maxLift, 'lower'],
    [3.2, 48, 'lower'],
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
