// Standard open-guitar voicings, ordered from the lowest played string upward.
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
  const button = root.querySelector('.guitar-play');
  const status = root.querySelector('#guitar-status');
  const Context = window.AudioContext || window.webkitAudioContext;
  if (!Context) {
    button.setAttribute('aria-label', '此浏览器暂不支持吉他声音');
    return;
  }

  let context;
  let output;
  let nextChord = 0;
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

  button.disabled = false;
  button.addEventListener('click', () => {
    // Create/resume audio only inside a user gesture; page load and animation stay silent.
    let ready;
    try {
      if (!context) {
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
      }
      ready = context.resume();
    } catch {
      status.textContent = '声音暂时无法播放，请再点击一次。';
      return;
    }
    // Preserve click order even when the browser takes time to unlock audio.
    queue = queue.then(async () => {
      await ready;
      if (document.hidden) return;
      const chord = chords[nextChord];
      playChord(chord);
      nextChord = (nextChord + 1) % chords.length;
      button.dataset.currentChord = chord.name;
      status.textContent = '';
    }).catch(() => {
      status.textContent = '声音暂时无法播放，请再点击一次。';
    });
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stopVoices();
  });
}
