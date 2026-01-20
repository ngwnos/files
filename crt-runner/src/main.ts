import './style.css';
import { WebGPURenderer } from 'three/webgpu';
import Stats from 'three/examples/jsm/libs/stats.module.js';
import { CRTScreenScene } from './CRTScreenScene';

const canvas = document.querySelector<HTMLCanvasElement>('#app');
if (!canvas) {
  throw new Error('Missing #app canvas');
}

type QualityPreset = 'low' | 'medium' | 'high';

const QUALITY_PRESETS: Record<QualityPreset, {
  maxPixelRatio: number;
  screenResolution: string;
  bloomStrength: number;
  keyboardLightSampleGrid: number;
}> = {
  low: {
    maxPixelRatio: 1,
    screenResolution: '320x240',
    bloomStrength: 0.0,
    keyboardLightSampleGrid: 2
  },
  medium: {
    maxPixelRatio: 1.5,
    screenResolution: '480x270',
    bloomStrength: 0.8,
    keyboardLightSampleGrid: 4
  },
  high: {
    maxPixelRatio: 2,
    screenResolution: '960x540',
    bloomStrength: 1.88,
    keyboardLightSampleGrid: 8
  }
};

const scene = new CRTScreenScene();
const stats = new Stats();
document.body.appendChild(stats.dom);
const qualityButtons = Array.from(
  document.querySelectorAll<HTMLButtonElement>('#quality-controls [data-quality]')
);

(async () => {
  const renderer = new WebGPURenderer({ canvas, antialias: true });
  let activeQuality: QualityPreset = 'medium';
  const applyQualityPreset = (preset: QualityPreset) => {
    const settings = QUALITY_PRESETS[preset];
    activeQuality = preset;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, settings.maxPixelRatio));
    scene.updateParameters({
      screenResolution: settings.screenResolution,
      bloomStrength: settings.bloomStrength,
      keyboardLightSampleGrid: settings.keyboardLightSampleGrid
    });
    scene.onResize(window.innerWidth, window.innerHeight);
    qualityButtons.forEach((button) => {
      const isActive = button.dataset.quality === preset;
      button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
    });
  };

  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, QUALITY_PRESETS[activeQuality].maxPixelRatio));
  await renderer.init();

  await scene.init(canvas, renderer);
  scene.updateParameters({ displayMode: 'shader', keyboardEnabled: false });
  applyQualityPreset(activeQuality);

  const resize = () => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;
    scene.onResize(width, height);
  };
  resize();
  window.addEventListener('resize', resize);
  qualityButtons.forEach((button) => {
    button.addEventListener('click', () => {
      const quality = button.dataset.quality as QualityPreset | undefined;
      if (!quality) {
        return;
      }
      applyQualityPreset(quality);
    });
  });
  window.addEventListener('keydown', (event) => {
    if (event.repeat) {
      return;
    }
    if (event.key === '1') {
      applyQualityPreset('low');
    } else if (event.key === '2') {
      applyQualityPreset('medium');
    } else if (event.key === '3') {
      applyQualityPreset('high');
    }
  });

  let last = performance.now();
  const loop = (time: number) => {
    const delta = (time - last) / 1000;
    last = time;
    scene.update(delta);
    scene.render();
    stats.update();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
})();
