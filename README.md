# CRT Runner

![CRT screenshot](ss-crt.png)

CRT Runner is a small WebGPU demo that renders a virtual CRT screen using the
`CRTScreenScene` class. It is built with Vite + Three.js WebGPU and includes
quality presets, a keyboard toggle, and a Stats.js FPS panel.

## Repository layout

- `crt-runner/` - Vite app that runs `CRTScreenScene`.
- `CRTScreenScene.ts` - The CRT scene implementation.
- `docs/` - Production build output target for GitHub Pages.

## Requirements

- Node.js 18+ (Vite 7).
- A browser with WebGPU enabled (Chrome or Edge).

## Install

```sh
cd crt-runner
npm install
```

## Run

```sh
cd crt-runner
npm run dev
```

Open `http://localhost:5173/`.

Quality presets can be switched with the UI or with keys `1`, `2`, `3`.

## Deploy to GitHub Pages

```sh
cd crt-runner
npm run build:gh-pages
```

This writes a production build to `docs/` at the repo root using relative
paths for assets. Commit the `docs/` folder and configure GitHub Pages to
serve from `docs/` on the main branch.

## Assets

Keyboard and terminal fonts are optional. To enable them, add these to
`crt-runner/public/`:

- `keyboard-layout-104.svg`
- `fonts/oldschool_pc/Web437_IBM_VGA_8x16.woff`
- `fonts/oldschool_pc/Web437_CompaqThin_8x16.woff`

Emulator assets (if you enable emulator mode) should live under:

- `roms/`
- `dos/`

## Integration example

```ts
import { WebGPURenderer } from 'three/webgpu';
import { CRTScreenScene } from './CRTScreenScene';

const canvas = document.querySelector<HTMLCanvasElement>('canvas');
if (!canvas) throw new Error('Missing canvas');

const renderer = new WebGPURenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
await renderer.init();

const scene = new CRTScreenScene();
await scene.init(canvas, renderer);
scene.updateParameters({ displayMode: 'shader', keyboardEnabled: false });

const resize = () => {
  const width = window.innerWidth;
  const height = window.innerHeight;
  canvas.width = width;
  canvas.height = height;
  scene.onResize(width, height);
};
resize();
window.addEventListener('resize', resize);

let last = performance.now();
const loop = (time: number) => {
  const delta = (time - last) / 1000;
  last = time;
  scene.update(delta);
  scene.render();
  requestAnimationFrame(loop);
};
requestAnimationFrame(loop);
```
