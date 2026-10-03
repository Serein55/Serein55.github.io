// 引体向上的手臂运动学。
//
// 坐标沿用 index.html 中机器人 SVG 的本地坐标（viewBox 内，y 轴向下，
// 外层还有 translate(32 76) scale(.8)）：
// 身体中线 x=160，肩关节基准 (±24, 155)，单杠握点 (±52, 65)。
// 握点偏移 52 不是估的：SVG 里手部方块画在 x=100~116 / 204~220，中心
// 正好是中线两侧 52。若用别的值，手臂末端会停在半空中、与手脱开。
//
// 手臂是「上臂 + 前臂」两段刚体。给定肩与握点，肘部是两圆交点（离散解），
// 所以肘部方位不能作为连续驱动量——真正连续的自由度是肩关节自身的轨迹。
// 因此这里用标准的垂直平分线法求肘（骨长天然守恒），再通过让肩部沿一条
// 符合人体工学的弧线移动，来得到连续、可信的肘部轨迹。
//
// 三个关键约束：
// 1. 行程区间。maxLift 不能只看「肩不超过单杠」——那样下巴只比杠高 1px，
//    视觉上等于没上去。真正的判据是**下巴是否明显高过单杠**（见 maxLift 处）。
//    原实现让 lift 涨到 188，把肩拉到单杠上方近 100px，手臂被迫折叠。
// 2. 骨段长度。53/52 取自 SVG 静态路径 M136 155L99 117L108 65，
//    是原作者画好的比例；两段之和略小于肩—握点的最小距离，否则顶点处
//    两圆无交点、sqrt 负数被截断后肘部会飞出画面。
// 3. 肘部法向的符号。见 armPose 里的说明——乘 side 会让肘部穿过身体中线。

const bodyMidX = 160;     // 身体中线
const barY = 65;          // 握点（单杠）高度
const shoulderY = 155;    // 静止时肩关节高度
const shoulderX = 24;     // 肩关节相对中线的水平偏移
const gripX = 52;         // 握点相对中线的水平偏移
const upperArm = 53;      // 上臂长：肩 → 肘
const foreArm = 52;       // 前臂长：肘 → 握点（与 SVG 静态路径 M136 155L99 117L108 65 一致）

// 顶点行程：肩部升到下巴明显高过单杠为止。
//
// 上限由两个相互竞争的约束夹住，扫描确定：
//   · 下限（美学）：下巴在 SVG 里约y=138，单杠 y=63。lift=76 时下巴只超杠
//     1px，肉眼看不出「举上去」，长按和空闲摆动几乎分不出来。
//   · 上限（几何）：lift 继续增大，肩—握点距离 reach 趋近 forearm 侧，肘部
//     外展量从 36.7px 单调降到-0.9px（lift=100）乃至-10.8px（lift=108），
//     肘部越过身体中线、手臂在胸前交叉，重现早期「翅膀」bug。
// 92px 时下巴超杠 17px（清晰可见），肘部仍外展 15.7px，两端都留有余量。
export const maxLift = 92;

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

  // 法向：取 (uy, −ux)·(−side)，即「肩 → 握点」方向绕中线镜像后逆时针转 90°，
  // 肘部落在肩的外侧偏下方（远离中线），并随拉起由外张逐渐转为竖起——
  // 与 SVG 静态路径肘部 (99, 117) 一致，也符合真实引体向上「肘部先后展、
  // 后收拢」的运动顺序。
  //
  // 系数必须逐臂镜像，否则两臂肘部会朝同一侧偏、失去对称：
  //   ·  乘 +side → 法向翻向身体中线，肘部 x 越过 bodyMidX，在胸前交叉成
  //      一个三角形（顶点最明显），既不符合人体力学，也会与躯干重叠。
  //   ·  完全不乘 → 左右臂共用同一法向，右臂肘部朝左偏，整个动作歪斜。
  const nx = uy * -side;
  const ny = -ux * -side;

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
  // 关键帧的峰值直接引用 maxLift，避免两处数值不同步（曾因硬编码 76 导致
  // 顶点与几何上限不一致）。中间帧按比例铺开，保证上行比下行快——真实引体
  // 向上是爆发拉起、缓慢控制下放。
  const top = maxLift;
  const poses = [
    [0, startLift, 'pull'],
    [.6, top * .38, 'transition'],
    [1.1, top * .66, 'press'],
    [1.75, top, 'support'],
    [2.5, top, 'lower'],
    [3.3, top * .52, 'lower'],
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
