// research 页展示视频（guitar-loop.mp4，无音轨）。这个模块负责弹吉他声：
// 合成与滤波逻辑沿用原版，入口从「点按钮弹一个和弦」改成
// 「鼠标移入画面时播放视频并按节拍弹和弦；移开后播完当前一轮循环再停」。
// 视频本身 muted + 无 autoplay，全程由 play()/pause() 控制。
//
// 自动播放策略：AudioContext 必须在用户手势里创建/恢复，鼠标移入属于
// 用户激活，所以第一次移入通常就能出声；若被策略挡下（部分浏览器对
// 非点击交互更严格），会提示「请再移入一次」，再次移入时解锁成功。
export const chords = [
  { name: 'Em', notes: [40, 47, 52, 55, 59, 64] },
  { name: 'C', notes: [48, 52, 55, 60, 64] },
  { name: 'G', notes: [43, 47, 50, 55, 59, 67] },
  { name: 'Dm', notes: [50, 57, 62, 65] },
];

export function createPluckSamples(sampleRate, midi) {
  const frequency = 440 * 2 ** ((midi - 69) / 12);
  // The averaging filter adds half a sample of delay; compensate to keep the string in tune.
  const delay = sampleRate / frequency - .5;
  const seedLength = Math.ceil(delay) + 1;
  const samples = new Float32Array(Math.ceil(sampleRate * 3.2));
  let mean = 0;
  for (let i = 0; i < seedLength; i++) {
    samples[i] = Math.random() * 2 - 1;
    mean += samples[i];
  }
  mean /= seedLength;
  for (let i = 0; i < seedLength; i++) samples[i] -= mean;
  for (let i = seedLength; i < samples.length; i++) {
    const position = i - delay;
    const index = Math.floor(position);
    const fraction = position - index;
    const current = samples[index] * (1 - fraction) + samples[index + 1] * fraction;
    const previous = samples[index - 1] * (1 - fraction) + samples[index] * fraction;
    samples[i] = .998 * (current + previous) / 2;
  }
  // Gentle attack and release avoid clicks, including at the end of the buffer.
  for (let i = 0; i < samples.length; i++) {
    const time = i / sampleRate;
    const attack = Math.sin(Math.min(1, time / .018) * Math.PI / 2) ** 2;
    const release = Math.min(1, (samples.length - 1 - i) / (sampleRate * .08));
    samples[i] *= attack * release * Math.exp(-time / 2.4);
  }
  return samples;
}

export function setupGuitarAudio(root) {
  const video = root.querySelector('video.motion-art');
  const status = root.querySelector('#guitar-status');
  const Context = window.AudioContext || window.webkitAudioContext;
  if (!Context || !video) return;

  let context;
  let output;
  let queue = Promise.resolve();
  const buffers = new Map();
  const voices = new Set();

  function stopVoices() {
    if (!context) return;
    const now = context.currentTime;
    for (const voice of voices) {
      voice.gain.gain.cancelScheduledValues(now);
      voice.gain.gain.setTargetAtTime(0, now, .025);
      voice.source.stop(now + .15);
    }
    voices.clear();
  }

  function playChord(chord) {
    stopVoices();
    const start = context.currentTime + .015;
    chord.notes.forEach((midi, index) => {
      if (!buffers.has(midi)) {
        const samples = createPluckSamples(context.sampleRate, midi);
        const buffer = context.createBuffer(1, samples.length, context.sampleRate);
        buffer.copyToChannel(samples, 0);
        buffers.set(midi, buffer);
      }
      const source = context.createBufferSource();
      source.buffer = buffers.get(midi);
      const gain = context.createGain();
      gain.gain.value = .16;
      source.connect(gain).connect(output);
      const voice = { source, gain };
      voices.add(voice);
      source.onended = () => {
        source.disconnect();
        gain.disconnect();
        voices.delete(voice);
      };
      source.start(start + index * .13);
    });
  }
  // 每个视频循环内弹几个和弦。视频 10.3s，取 4 拍 ≈ 每 2.6s 一个和弦。
  const beatsPerLoop = 4;
  let lastBeat = -1;
  let unlocked = false;
  let frame = null;

  function ensureContext() {
    if (context) return context.resume();
    context = new Context();
    // Round off the bright pluck transient for a softer, fingerpicked timbre.
    output = context.createBiquadFilter();
    output.type = 'lowpass';
    output.frequency.value = 1700;
    output.Q.value = .5;
    const compressor = context.createDynamicsCompressor();
    compressor.threshold.value = -18;
    compressor.knee.value = 18;
    compressor.ratio.value = 4;
    output.connect(compressor).connect(context.destination);
    return context.resume();
  }

  function beatAtVideo() {
    const duration = video.duration || 10.3;
    return Math.floor((video.currentTime / duration) * beatsPerLoop);
  }

  // 悬停 = 播放视频 + 出声；移开 = 播完当前一轮循环再停。
  // 视频 muted + 无 autoplay，纯靠 play()/pause() 驱动；
  // AudioContext 在首次 mouseenter 时创建（mousemove 属用户激活）。
  //
  // 注意：视频带 loop 属性，**永远不会触发 ended 事件**，所以「播完一轮再停」
  // 只能靠监测 currentTime 回绕（见 tick 里的回绕检测），不能等 ended。
  const figure = video.closest('.motion-study') || video;
  // 鼠标已移开、正在等这一轮播完
  let pendingStop = false;

  function start() {
    if (document.hidden) return;
    pendingStop = false;
    if (video.paused) video.play().catch(() => {});
    // rAF 循环必须一直跑着：即使还没解锁声音，也要靠它监测回绕来实现
    // 「播完这一轮再停」，所以这里不能在 !unlocked 时提前 return。
    if (frame === null) frame = requestAnimationFrame(tick);
  }

  // 立即停：直接暂停并静音。用于立即中止（页面隐藏、解锁失败等）。
  function stopNow() {
    pendingStop = false;
    if (!video.paused) video.pause();
    stopVoices();
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
  }

  // 延迟停：等当前这一轮循环播完再停。鼠标移开时用。
  // 这里不动任何 DOM 状态 —— 画面上已经没有状态图标（2026-10-12 用户要求去掉），
  // 所以只需置标志位，等 tick 检测到回绕后由 stopNow() 收尾。
  function stopAfterLoop() {
    if (video.paused || !Number.isFinite(video.duration) || video.duration <= 0) {
      stopNow();
      return;
    }
    pendingStop = true;
  }

  // 按视频进度推进和弦：视频循环时 currentTime 归零，节拍号也跟着归零，
  // 所以接缝之后自然续上下一个和弦，不会因时间跳变而漏拍或连击。
  function tick() {
    frame = requestAnimationFrame(tick);
    if (document.hidden || video.paused) return;
    const beat = beatAtVideo();
    // 循环回绕时 beat 会突然变小 —— 这就是「一轮播完」的信号。
    if (pendingStop && lastBeat >= 0 && beat < lastBeat) {
      stopNow();
      return;
    }
    if (!unlocked) return;
    if (beat === lastBeat) return;
    // 回绕时把lastBeat 退回一拍，避免瞬间叠三个和弦。
    if (lastBeat >= 0 && beat < lastBeat) lastBeat = beat - 1;
    lastBeat = beat;
    const chord = chords[beat % chords.length];
    queue = queue.then(async () => {
      await context.resume();
      if (document.hidden || video.paused) return;
      playChord(chord);
      // 无样式消费者（状态图标已删），保留作为调试钩子：
      // 自动化验证靠读 video.dataset.chord 判断和弦是否还在推进。
      video.dataset.chord = chord.name;
      status.textContent = '';
    }).catch(() => {});
  }

  // 第一次悬停当作解锁手势（mousemove 算用户激活，AudioContext 能 resume）。
  function unlockAndStart() {
    if (unlocked) { start(); return; }
    queue = queue.then(() => ensureContext()).then(() => {
      unlocked = true;
      lastBeat = beatAtVideo() - 1;
      status.textContent = '';
      start();
    }).catch(() => {
      unlocked = false;
      stopNow();
      status.textContent = '声音暂时无法播放，请再移入一次。';
    });
  }

  figure.addEventListener('mouseenter', unlockAndStart);
  figure.addEventListener('mouseleave', stopAfterLoop);
  // 键盘路径用立即停：焦点移走时用户多半已经看别处去了，等一轮太久。
  figure.addEventListener('focus', unlockAndStart);
  figure.addEventListener('blur', stopNow);
  // 页面切走/切回标签时立刻停，理由同上——用户已经不在看了。
  document.addEventListener('visibilitychange', () => { if (document.hidden) stopNow(); });
}

// 这个模块以前由 intro.mjs 调用；intro.mjs 随 SVG 动画一起删了，
// 所以改成模块自己启动。type="module" 是 defer 的，此时 DOM 已就绪。
const studies = document.querySelector('.motion-studies');
if (studies) setupGuitarAudio(studies);
