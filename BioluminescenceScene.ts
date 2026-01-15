import * as THREE from 'three';
import {
  MeshBasicNodeMaterial,
  PostProcessing,
  Storage3DTexture,
  StorageBufferAttribute,
  WebGPURenderer
} from 'three/webgpu';
import {
  Fn,
  uniform,
  vec4,
  vec3,
  float,
  instanceIndex,
  positionLocal,
  attributeArray,
  cameraProjectionMatrix,
  cameraViewMatrix,
  cameraPosition,
  normalize,
  cross,
  dot,
  abs,
  select,
  sin,
  cos,
  mix,
  length,
  max,
  clamp,
  hash,
  exp,
  texture3D,
  textureStore,
  uvec3,
  mat3,
  If,
  pass,
  mrt,
  output,
  emissive,
  storage,
  atomicAdd,
  atomicStore,
  uint
} from 'three/tsl';
import { bloom } from 'three/examples/jsm/tsl/display/BloomNode.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import { BIOLUMINESCENCE_SCENE_DEFAULTS, type BioluminescenceSceneParameters } from './BioluminescenceSceneParameters';

export class BioluminescenceScene {
  private renderer: WebGPURenderer | null = null;
  private scene: THREE.Scene | null = null;
  private camera: THREE.PerspectiveCamera | null = null;
  private controls: OrbitControls | null = null;
  private canvas: HTMLCanvasElement | null = null;

  private particleMesh: THREE.InstancedMesh | null = null;
  private particleGeometry: THREE.PlaneGeometry | null = null;
  private particleMaterial: MeshBasicNodeMaterial | null = null;
  private sparkleRandomStorage: any = null;
  private particlePositionsStorage: any = null;
  private particleVelocitiesStorage: any = null;
  private particleInitCompute: any = null;
  private energyStorage: any = null;
  private activationStorage: any = null;
  private refractoryStorage: any = null;
  private computeUpdate: any = null;
  private bubbleMesh: THREE.InstancedMesh | null = null;
  private bubbleGeometry: THREE.SphereGeometry | null = null;
  private bubbleMaterial: MeshBasicNodeMaterial | null = null;
  private bubblePositionsStorage: any = null;
  private bubbleVelocityStorage: any = null;
  private bubbleComputeUpdate: any = null;
  private bubblePositionsArray: Float32Array | null = null;
  private bubbleVelocitiesArray: Float32Array | null = null;
  private readonly bubblePoolSize = 128;
  private bubbleSizeMin = 2;
  private bubbleSizeMax = 6;
  private bubbleColor = new THREE.Color('#66ccff');
  private fieldVelocityTextures: [Storage3DTexture, Storage3DTexture] | null = null;
  private fieldVelocitySamplers: [ReturnType<typeof texture3D>, ReturnType<typeof texture3D>] | null =
    null;
  private fieldClearVelocityPasses: [any, any] = [null, null];
  private fieldSplatPasses: [any, any] = [null, null];
  private fieldVelocityReadIndex = 0;
  private fieldResolution = new THREE.Vector3(64, 32, 64);
  private densityBufferAttribute: StorageBufferAttribute | null = null;
  private densityBufferNode: ReturnType<typeof storage> | null = null;
  private densityClearCompute: any = null;
  private densityDepositCompute: any = null;
  private densityGradientTexture: Storage3DTexture | null = null;
  private densityGradientSampler: ReturnType<typeof texture3D> | null = null;
  private densityGradientCompute: any = null;
  private bubbleDensityBufferAttribute: StorageBufferAttribute | null = null;
  private bubbleDensityBufferNode: ReturnType<typeof storage> | null = null;
  private bubbleDensityClearCompute: any = null;
  private bubbleDensityDepositCompute: any = null;
  private bubbleDensityGradientTexture: Storage3DTexture | null = null;
  private bubbleDensityGradientSampler: ReturnType<typeof texture3D> | null = null;
  private bubbleDensityGradientCompute: any = null;
  private bubbleInjectionTexture: Storage3DTexture | null = null;
  private bubbleInjectionSampler: ReturnType<typeof texture3D> | null = null;
  private bubbleInjectionClearPass: any = null;
  private bubbleInjectionCompute: any = null;
  private densityResolution = new THREE.Vector3(32, 16, 32);
  private densityParticleCount = 0;
  private volumeWireframe: THREE.LineSegments | null = null;
  private readonly disposeCallbacks: Array<() => void> = [];
  private isEmittingBubbles = false;
  private bubbleKeyEmit = false;
  private particleResetPending = false;
  private fieldResetPending = false;
  private bubbleResetPending = false;

  private postProcessing: PostProcessing | null = null;
  private bloomNode: any = null;

  private timeUniform = uniform(0.0);
  private deltaTimeUniform = uniform(0.0);
  private gridXUniform = uniform(1);
  private gridYUniform = uniform(1);
  private gridZUniform = uniform(1);
  private volumeSizeUniform = uniform(new THREE.Vector3(6, 4, 6));
  private volumeCenterUniform = uniform(new THREE.Vector3(0, 0, 0));
  private particleSizeUniform = uniform(0.03);
  private particleWakeStrengthUniform = uniform(0.6);
  private particleDragUniform = uniform(1.0);
  private particleNoiseStrengthUniform = uniform(0.1);
  private particleNoiseFrequencyUniform = uniform(0.6);
  private particleNoiseDragUniform = uniform(0.5);
  private particleGravityUniform = uniform(0.0);
  private activationScaleUniform = uniform(3.0);
  private activationSpeedUniform = uniform(1.0);
  private bubbleEmitUniform = uniform(0.0);
  private bubbleResetUniform = uniform(0.0);
  private bubbleSizeMinUniform = uniform(2.0);
  private bubbleSizeMaxUniform = uniform(6.0);
  private bubbleSpawnRadiusUniform = uniform(0.0);
  private bubbleSpawnRateUniform = uniform(20.0);
  private bubbleSpawnAreaUniform = uniform(0.0);
  private bubbleDriftStrengthUniform = uniform(0.0);
  private bubbleDriftFrequencyUniform = uniform(0.5);
  private bubbleDriftDragUniform = uniform(1.2);
  private bubbleRepelStrengthUniform = uniform(0.8);
  private bubbleRepelRadiusUniform = uniform(0.08);
  private bubbleGravityUniform = uniform(9.81);
  private bubbleViscosityUniform = uniform(0.001);
  private bubbleWakeStrengthUniform = uniform(0.6);
  private bubbleWakeLengthUniform = uniform(1.5);
  private bubbleWakeAngleUniform = uniform(0.4);
  private activationVelocityScaleUniform = uniform(1.0);
  private energyAccumulationUniform = uniform(1.2);
  private energyDecayUniform = uniform(0.8);
  private activationThresholdUniform = uniform(1.0);
  private activationDurationUniform = uniform(1.0);
  private refractoryPeriodUniform = uniform(1.5);
  private sparkleMinUniform = uniform(0.5);
  private sparkleMaxUniform = uniform(1.4);
  private baseColorUniform = uniform(new THREE.Color('#00bcd4'));
  private activeColorUniform = uniform(new THREE.Color('#7dffb0'));
  private baseAlphaUniform = uniform(0.2);
  private activeAlphaUniform = uniform(0.95);
  private emissiveStrengthUniform = uniform(1.6);
  private fieldResXUniform = uniform(64);
  private fieldResYUniform = uniform(32);
  private fieldResZUniform = uniform(64);
  private fieldDissipationUniform = uniform(0.98);
  private fieldSplatStrengthUniform = uniform(1.0);
  private fieldSplatRadiusUniform = uniform(0.3);
  private fieldVelocityIndexUniform = uniform(0.0);
  private densityResXUniform = uniform(32);
  private densityResYUniform = uniform(16);
  private densityResZUniform = uniform(32);
  private particleDensityStrengthUniform = uniform(0.3);

  private gridCounts = { x: 1, y: 1, z: 1 };
  private volumeSize = new THREE.Vector3(6, 4, 6);
  private volumeCenter = new THREE.Vector3(0, 0, 0);

  private parameters: BioluminescenceSceneParameters = {
    ...BIOLUMINESCENCE_SCENE_DEFAULTS
  } as BioluminescenceSceneParameters;

  async init(canvas: HTMLCanvasElement): Promise<void> {
    this.canvas = canvas;
    const width = canvas.clientWidth || canvas.width || window.innerWidth || 1;
    const height = canvas.clientHeight || canvas.height || window.innerHeight || 1;

    this.camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 200);
    this.camera.position.set(8, 6, 10);
    this.camera.lookAt(0, 0, 0);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.parameters.backgroundColor as string);

    this.renderer = new WebGPURenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(width, height);
    await this.renderer.init();

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0, 0);

    this.updateGrid();
    this.bubbleSizeMin = this.parameters.bubbleSizeMin as number;
    this.bubbleSizeMax = this.parameters.bubbleSizeMax as number;
    this.bubbleSizeMinUniform.value = this.bubbleSizeMin;
    this.bubbleSizeMaxUniform.value = this.bubbleSizeMax;
    if (typeof this.parameters.bubbleColor === 'string') {
      this.bubbleColor.set(this.parameters.bubbleColor as string);
    }
    if (typeof this.parameters.fieldResX === 'number') {
      this.fieldResXUniform.value = this.parameters.fieldResX as number;
    }
    if (typeof this.parameters.fieldResY === 'number') {
      this.fieldResYUniform.value = this.parameters.fieldResY as number;
    }
    if (typeof this.parameters.fieldResZ === 'number') {
      this.fieldResZUniform.value = this.parameters.fieldResZ as number;
    }
    this.fieldResolution.set(
      this.fieldResXUniform.value as number,
      this.fieldResYUniform.value as number,
      this.fieldResZUniform.value as number
    );
    if (typeof this.parameters.densityResX === 'number') {
      this.densityResXUniform.value = this.parameters.densityResX as number;
    }
    if (typeof this.parameters.densityResY === 'number') {
      this.densityResYUniform.value = this.parameters.densityResY as number;
    }
    if (typeof this.parameters.densityResZ === 'number') {
      this.densityResZUniform.value = this.parameters.densityResZ as number;
    }
    this.densityResolution.set(
      this.densityResXUniform.value as number,
      this.densityResYUniform.value as number,
      this.densityResZUniform.value as number
    );
    if (typeof this.parameters.particleDensityStrength === 'number') {
      this.particleDensityStrengthUniform.value =
        this.parameters.particleDensityStrength as number;
    }
    if (typeof this.parameters.fieldDissipation === 'number') {
      this.fieldDissipationUniform.value = this.parameters.fieldDissipation as number;
    }
    if (typeof this.parameters.fieldSplatStrength === 'number') {
      this.fieldSplatStrengthUniform.value = this.parameters.fieldSplatStrength as number;
    }
    if (typeof this.parameters.fieldSplatRadius === 'number') {
      this.fieldSplatRadiusUniform.value = this.parameters.fieldSplatRadius as number;
    }
    if (typeof this.parameters.activationVelocityScale === 'number') {
      this.activationVelocityScaleUniform.value =
        this.parameters.activationVelocityScale as number;
    }
    if (typeof this.parameters.bubbleGravity === 'number') {
      this.bubbleGravityUniform.value = this.parameters.bubbleGravity as number;
    }
    if (typeof this.parameters.bubbleViscosity === 'number') {
      this.bubbleViscosityUniform.value = this.parameters.bubbleViscosity as number;
    }
    if (typeof this.parameters.bubbleSpawnRate === 'number') {
      this.bubbleSpawnRateUniform.value = this.parameters.bubbleSpawnRate as number;
    }
    if (typeof this.parameters.bubbleSpawnArea === 'number') {
      this.bubbleSpawnAreaUniform.value = this.parameters.bubbleSpawnArea as number;
    }
    if (typeof this.parameters.bubbleWakeStrength === 'number') {
      this.bubbleWakeStrengthUniform.value = this.parameters.bubbleWakeStrength as number;
    }
    if (typeof this.parameters.bubbleWakeLength === 'number') {
      this.bubbleWakeLengthUniform.value = (this.parameters.bubbleWakeLength as number) * 0.001;
    }
    if (typeof this.parameters.bubbleWakeAngle === 'number') {
      this.bubbleWakeAngleUniform.value = (this.parameters.bubbleWakeAngle as number) * 0.001;
    }
    if (typeof this.parameters.particleWakeStrength === 'number') {
      this.particleWakeStrengthUniform.value =
        this.parameters.particleWakeStrength as number;
    }
    if (typeof this.parameters.particleDrag === 'number') {
      this.particleDragUniform.value = this.parameters.particleDrag as number;
    }
    if (typeof this.parameters.particleNoiseStrength === 'number') {
      this.particleNoiseStrengthUniform.value =
        this.parameters.particleNoiseStrength as number;
    }
    if (typeof this.parameters.particleNoiseFrequency === 'number') {
      this.particleNoiseFrequencyUniform.value =
        this.parameters.particleNoiseFrequency as number;
    }
    if (typeof this.parameters.particleNoiseDrag === 'number') {
      this.particleNoiseDragUniform.value =
        this.parameters.particleNoiseDrag as number;
    }
    if (typeof this.parameters.particleGravity === 'number') {
      this.particleGravityUniform.value = this.parameters.particleGravity as number;
    }
    if (typeof this.parameters.bubbleDriftStrength === 'number') {
      this.bubbleDriftStrengthUniform.value =
        this.parameters.bubbleDriftStrength as number;
    }
    if (typeof this.parameters.bubbleDriftFrequency === 'number') {
      this.bubbleDriftFrequencyUniform.value =
        this.parameters.bubbleDriftFrequency as number;
    }
    if (typeof this.parameters.bubbleDriftDrag === 'number') {
      this.bubbleDriftDragUniform.value = this.parameters.bubbleDriftDrag as number;
    }
    if (typeof this.parameters.bubbleRepelStrength === 'number') {
      this.bubbleRepelStrengthUniform.value =
        this.parameters.bubbleRepelStrength as number;
    }
    if (typeof this.parameters.bubbleRepelRadius === 'number') {
      this.bubbleRepelRadiusUniform.value =
        (this.parameters.bubbleRepelRadius as number) * 0.001;
    }
    this.ensureBubbles();
    this.createParticles();
    this.ensureFluidField();
    this.createVolumeWireframe();
    this.setupPostProcessing();
    this.attachKeyHandlers();
  }

  update(deltaSeconds = 0): void {
    if (!this.renderer) return;
    this.timeUniform.value += deltaSeconds;
    this.deltaTimeUniform.value = deltaSeconds;
    const autoEmit = this.parameters.bubbleAutoEmit as boolean | undefined;
    this.isEmittingBubbles = Boolean(autoEmit) || this.bubbleKeyEmit;
    this.bubbleEmitUniform.value = this.isEmittingBubbles ? 1 : 0;
    this.bubbleResetUniform.value = this.bubbleResetPending ? 1 : 0;
    this.bubbleResetPending = false;
    if (this.particleResetPending && this.particleInitCompute) {
      this.renderer.compute(this.particleInitCompute);
      this.particleResetPending = false;
    }
    this.controls?.update();
    if (!this.fieldVelocityTextures || this.fieldResetPending) {
      this.ensureFluidField();
    }
    this.ensureDensityField();
    if (this.fieldResetPending) {
      this.clearFluidField();
      this.fieldResetPending = false;
    }
    if (this.densityClearCompute) {
      this.renderer.compute(this.densityClearCompute);
    }
    if (this.densityDepositCompute) {
      this.renderer.compute(this.densityDepositCompute);
    }
    if (this.densityGradientCompute) {
      this.renderer.compute(this.densityGradientCompute);
    }
    if (this.bubbleDensityClearCompute) {
      this.renderer.compute(this.bubbleDensityClearCompute);
    }
    if (this.bubbleDensityDepositCompute) {
      this.renderer.compute(this.bubbleDensityDepositCompute);
    }
    if (this.bubbleDensityGradientCompute) {
      this.renderer.compute(this.bubbleDensityGradientCompute);
    }
    if (this.bubbleInjectionClearPass) {
      this.renderer.compute(this.bubbleInjectionClearPass);
    }
    if (this.bubbleInjectionCompute) {
      this.renderer.compute(this.bubbleInjectionCompute);
    }
    if (this.bubbleComputeUpdate) {
      this.renderer.compute(this.bubbleComputeUpdate);
    }
    this.stepField();
    if (this.computeUpdate) {
      this.renderer.compute(this.computeUpdate);
    }
  }

  render(): void {
    if (!this.renderer || !this.scene || !this.camera) return;
    if (this.postProcessing) {
      this.postProcessing.render();
    } else {
      this.renderer.render(this.scene, this.camera);
    }
  }

  updateParameters(changes: Record<string, unknown>): void {
    let needsRebuild = false;
    let needsVolume = false;
    let needsBloom = false;

    if (typeof changes.particleCount === 'number') {
      const clamped = Math.max(1, Math.min(1000000, Math.round(changes.particleCount)));
      if (clamped !== this.parameters.particleCount) {
        this.parameters.particleCount = clamped as BioluminescenceSceneParameters['particleCount'];
        needsRebuild = true;
      }
    }

    if (typeof changes.particleSize === 'number') {
      const clamped = Math.max(0.001, Math.min(0.2, changes.particleSize));
      this.parameters.particleSize = clamped as BioluminescenceSceneParameters['particleSize'];
      this.particleSizeUniform.value = clamped;
    }

    if (typeof changes.particleWakeStrength === 'number') {
      const clamped = Math.max(0, Math.min(10, changes.particleWakeStrength));
      this.parameters.particleWakeStrength =
        clamped as BioluminescenceSceneParameters['particleWakeStrength'];
      this.particleWakeStrengthUniform.value = clamped;
    }

    if (typeof changes.particleDrag === 'number') {
      const clamped = Math.max(0, Math.min(10, changes.particleDrag));
      this.parameters.particleDrag =
        clamped as BioluminescenceSceneParameters['particleDrag'];
      this.particleDragUniform.value = clamped;
    }

    if (typeof changes.particleNoiseStrength === 'number') {
      const clamped = Math.max(0, Math.min(2, changes.particleNoiseStrength));
      this.parameters.particleNoiseStrength =
        clamped as BioluminescenceSceneParameters['particleNoiseStrength'];
      this.particleNoiseStrengthUniform.value = clamped;
    }

    if (typeof changes.particleNoiseFrequency === 'number') {
      const clamped = Math.max(0, Math.min(5, changes.particleNoiseFrequency));
      this.parameters.particleNoiseFrequency =
        clamped as BioluminescenceSceneParameters['particleNoiseFrequency'];
      this.particleNoiseFrequencyUniform.value = clamped;
    }

    if (typeof changes.particleNoiseDrag === 'number') {
      const clamped = Math.max(0, Math.min(5, changes.particleNoiseDrag));
      this.parameters.particleNoiseDrag =
        clamped as BioluminescenceSceneParameters['particleNoiseDrag'];
      this.particleNoiseDragUniform.value = clamped;
    }

    if (typeof changes.densityResX === 'number') {
      const clamped = Math.max(4, Math.min(128, Math.round(changes.densityResX)));
      if (clamped !== this.parameters.densityResX) {
        this.parameters.densityResX = clamped as BioluminescenceSceneParameters['densityResX'];
        this.densityResXUniform.value = clamped;
      }
    }

    if (typeof changes.densityResY === 'number') {
      const clamped = Math.max(4, Math.min(128, Math.round(changes.densityResY)));
      if (clamped !== this.parameters.densityResY) {
        this.parameters.densityResY = clamped as BioluminescenceSceneParameters['densityResY'];
        this.densityResYUniform.value = clamped;
      }
    }

    if (typeof changes.densityResZ === 'number') {
      const clamped = Math.max(4, Math.min(128, Math.round(changes.densityResZ)));
      if (clamped !== this.parameters.densityResZ) {
        this.parameters.densityResZ = clamped as BioluminescenceSceneParameters['densityResZ'];
        this.densityResZUniform.value = clamped;
      }
    }

    if (typeof changes.particleDensityStrength === 'number') {
      const clamped = Math.max(0, Math.min(5, changes.particleDensityStrength));
      this.parameters.particleDensityStrength =
        clamped as BioluminescenceSceneParameters['particleDensityStrength'];
      this.particleDensityStrengthUniform.value = clamped;
    }

    if (typeof changes.particleGravity === 'number') {
      const clamped = Math.max(-20, Math.min(20, changes.particleGravity));
      this.parameters.particleGravity =
        clamped as BioluminescenceSceneParameters['particleGravity'];
      this.particleGravityUniform.value = clamped;
    }

    if (typeof changes.activationScale === 'number') {
      const clamped = Math.max(1, Math.min(10, changes.activationScale));
      this.parameters.activationScale =
        clamped as BioluminescenceSceneParameters['activationScale'];
      this.activationScaleUniform.value = clamped;
    }

    if (typeof changes.activationSpeed === 'number') {
      const clamped = Math.max(0, Math.min(30, changes.activationSpeed));
      this.parameters.activationSpeed =
        clamped as BioluminescenceSceneParameters['activationSpeed'];
      this.activationSpeedUniform.value = clamped;
    }

    if (typeof changes.fieldResX === 'number') {
      const clamped = Math.max(8, Math.min(128, Math.round(changes.fieldResX)));
      if (clamped !== this.parameters.fieldResX) {
        this.parameters.fieldResX = clamped as BioluminescenceSceneParameters['fieldResX'];
        this.fieldResXUniform.value = clamped;
        this.fieldResetPending = true;
      }
    }

    if (typeof changes.fieldResY === 'number') {
      const clamped = Math.max(8, Math.min(128, Math.round(changes.fieldResY)));
      if (clamped !== this.parameters.fieldResY) {
        this.parameters.fieldResY = clamped as BioluminescenceSceneParameters['fieldResY'];
        this.fieldResYUniform.value = clamped;
        this.fieldResetPending = true;
      }
    }

    if (typeof changes.fieldResZ === 'number') {
      const clamped = Math.max(8, Math.min(128, Math.round(changes.fieldResZ)));
      if (clamped !== this.parameters.fieldResZ) {
        this.parameters.fieldResZ = clamped as BioluminescenceSceneParameters['fieldResZ'];
        this.fieldResZUniform.value = clamped;
        this.fieldResetPending = true;
      }
    }

    if (typeof changes.fieldDissipation === 'number') {
      const clamped = Math.max(0, Math.min(5, changes.fieldDissipation));
      this.parameters.fieldDissipation =
        clamped as BioluminescenceSceneParameters['fieldDissipation'];
      this.fieldDissipationUniform.value = clamped;
    }

    if (typeof changes.fieldSplatStrength === 'number') {
      const clamped = Math.max(0, Math.min(10, changes.fieldSplatStrength));
      this.parameters.fieldSplatStrength =
        clamped as BioluminescenceSceneParameters['fieldSplatStrength'];
      this.fieldSplatStrengthUniform.value = clamped;
    }

    if (typeof changes.fieldSplatRadius === 'number') {
      const clamped = Math.max(0.01, Math.min(2, changes.fieldSplatRadius));
      this.parameters.fieldSplatRadius =
        clamped as BioluminescenceSceneParameters['fieldSplatRadius'];
      this.fieldSplatRadiusUniform.value = clamped;
    }

    if (typeof changes.activationVelocityScale === 'number') {
      const clamped = Math.max(0, Math.min(10, changes.activationVelocityScale));
      this.parameters.activationVelocityScale =
        clamped as BioluminescenceSceneParameters['activationVelocityScale'];
      this.activationVelocityScaleUniform.value = clamped;
    }

    if (typeof changes.bubbleSizeMin === 'number') {
      const clamped = Math.max(0.5, Math.min(50, changes.bubbleSizeMin));
      this.parameters.bubbleSizeMin =
        clamped as BioluminescenceSceneParameters['bubbleSizeMin'];
      this.bubbleSizeMin = clamped;
      this.bubbleSizeMinUniform.value = clamped;
    }

    if (typeof changes.bubbleSizeMax === 'number') {
      const clamped = Math.max(0.5, Math.min(50, changes.bubbleSizeMax));
      this.parameters.bubbleSizeMax =
        clamped as BioluminescenceSceneParameters['bubbleSizeMax'];
      this.bubbleSizeMax = clamped;
      this.bubbleSizeMaxUniform.value = clamped;
    }

    if (typeof changes.bubbleGravity === 'number') {
      const clamped = Math.max(0, Math.min(30, changes.bubbleGravity));
      this.parameters.bubbleGravity =
        clamped as BioluminescenceSceneParameters['bubbleGravity'];
      this.bubbleGravityUniform.value = clamped;
    }

    if (typeof changes.bubbleViscosity === 'number') {
      const clamped = Math.max(0, Math.min(0.1, changes.bubbleViscosity));
      this.parameters.bubbleViscosity =
        clamped as BioluminescenceSceneParameters['bubbleViscosity'];
      this.bubbleViscosityUniform.value = clamped;
    }

    if (typeof changes.bubbleSpawnRate === 'number') {
      const clamped = Math.max(0, Math.min(200, changes.bubbleSpawnRate));
      this.parameters.bubbleSpawnRate =
        clamped as BioluminescenceSceneParameters['bubbleSpawnRate'];
      this.bubbleSpawnRateUniform.value = clamped;
    }

    if (typeof changes.bubbleSpawnArea === 'number') {
      const clamped = Math.max(0, Math.min(1, changes.bubbleSpawnArea));
      this.parameters.bubbleSpawnArea =
        clamped as BioluminescenceSceneParameters['bubbleSpawnArea'];
      this.bubbleSpawnAreaUniform.value = clamped;
    }

    if (typeof changes.bubbleAutoEmit === 'boolean') {
      const wasEnabled = Boolean(this.parameters.bubbleAutoEmit);
      this.parameters.bubbleAutoEmit =
        changes.bubbleAutoEmit as BioluminescenceSceneParameters['bubbleAutoEmit'];
      if (!wasEnabled && this.parameters.bubbleAutoEmit) {
        this.bubbleResetPending = true;
      }
    }

    if (typeof changes.bubbleColor === 'string') {
      this.parameters.bubbleColor =
        changes.bubbleColor as BioluminescenceSceneParameters['bubbleColor'];
      this.bubbleColor.set(this.parameters.bubbleColor as string);
      if (this.bubbleMaterial) {
        this.bubbleMaterial.color.set(this.parameters.bubbleColor as string);
      }
    }

    if (typeof changes.bubbleDriftStrength === 'number') {
      const clamped = Math.max(0, Math.min(5, changes.bubbleDriftStrength));
      this.parameters.bubbleDriftStrength =
        clamped as BioluminescenceSceneParameters['bubbleDriftStrength'];
      this.bubbleDriftStrengthUniform.value = clamped;
    }

    if (typeof changes.bubbleDriftFrequency === 'number') {
      const clamped = Math.max(0, Math.min(5, changes.bubbleDriftFrequency));
      this.parameters.bubbleDriftFrequency =
        clamped as BioluminescenceSceneParameters['bubbleDriftFrequency'];
      this.bubbleDriftFrequencyUniform.value = clamped;
    }

    if (typeof changes.bubbleDriftDrag === 'number') {
      const clamped = Math.max(0, Math.min(10, changes.bubbleDriftDrag));
      this.parameters.bubbleDriftDrag =
        clamped as BioluminescenceSceneParameters['bubbleDriftDrag'];
      this.bubbleDriftDragUniform.value = clamped;
    }

    if (typeof changes.bubbleRepelStrength === 'number') {
      const clamped = Math.max(0, Math.min(10, changes.bubbleRepelStrength));
      this.parameters.bubbleRepelStrength =
        clamped as BioluminescenceSceneParameters['bubbleRepelStrength'];
      this.bubbleRepelStrengthUniform.value = clamped;
    }

    if (typeof changes.bubbleRepelRadius === 'number') {
      const clamped = Math.max(1, Math.min(500, changes.bubbleRepelRadius));
      this.parameters.bubbleRepelRadius =
        clamped as BioluminescenceSceneParameters['bubbleRepelRadius'];
      this.bubbleRepelRadiusUniform.value = clamped * 0.001;
    }

    if (typeof changes.bubbleWakeStrength === 'number') {
      const clamped = Math.max(0, Math.min(10, changes.bubbleWakeStrength));
      this.parameters.bubbleWakeStrength =
        clamped as BioluminescenceSceneParameters['bubbleWakeStrength'];
      this.bubbleWakeStrengthUniform.value = clamped;
    }

    if (typeof changes.bubbleWakeLength === 'number') {
      const clamped = Math.max(10, Math.min(5000, changes.bubbleWakeLength));
      this.parameters.bubbleWakeLength =
        clamped as BioluminescenceSceneParameters['bubbleWakeLength'];
      this.bubbleWakeLengthUniform.value = clamped * 0.001;
    }

    if (typeof changes.bubbleWakeAngle === 'number') {
      const clamped = Math.max(10, Math.min(2000, changes.bubbleWakeAngle));
      this.parameters.bubbleWakeAngle =
        clamped as BioluminescenceSceneParameters['bubbleWakeAngle'];
      this.bubbleWakeAngleUniform.value = clamped * 0.001;
    }

    if (typeof changes.energyAccumulationRate === 'number') {
      const clamped = Math.max(0, Math.min(20, changes.energyAccumulationRate));
      this.parameters.energyAccumulationRate =
        clamped as BioluminescenceSceneParameters['energyAccumulationRate'];
      this.energyAccumulationUniform.value = clamped;
    }

    if (typeof changes.energyDecayRate === 'number') {
      const clamped = Math.max(0, Math.min(20, changes.energyDecayRate));
      this.parameters.energyDecayRate =
        clamped as BioluminescenceSceneParameters['energyDecayRate'];
      this.energyDecayUniform.value = clamped;
    }

    if (typeof changes.activationEnergyThreshold === 'number') {
      const clamped = Math.max(0.01, Math.min(20, changes.activationEnergyThreshold));
      this.parameters.activationEnergyThreshold =
        clamped as BioluminescenceSceneParameters['activationEnergyThreshold'];
      this.activationThresholdUniform.value = clamped;
    }

    if (typeof changes.activationDuration === 'number') {
      const clamped = Math.max(0.05, Math.min(10, changes.activationDuration));
      this.parameters.activationDuration =
        clamped as BioluminescenceSceneParameters['activationDuration'];
      this.activationDurationUniform.value = clamped;
    }

    if (typeof changes.refractoryPeriod === 'number') {
      const clamped = Math.max(0, Math.min(10, changes.refractoryPeriod));
      this.parameters.refractoryPeriod =
        clamped as BioluminescenceSceneParameters['refractoryPeriod'];
      this.refractoryPeriodUniform.value = clamped;
    }

    if (typeof changes.sparkleMin === 'number') {
      const clamped = Math.max(0, Math.min(5, changes.sparkleMin));
      this.parameters.sparkleMin = clamped as BioluminescenceSceneParameters['sparkleMin'];
      this.sparkleMinUniform.value = clamped;
    }

    if (typeof changes.sparkleMax === 'number') {
      const clamped = Math.max(0, Math.min(5, changes.sparkleMax));
      this.parameters.sparkleMax = clamped as BioluminescenceSceneParameters['sparkleMax'];
      this.sparkleMaxUniform.value = clamped;
    }
    if (
      typeof this.parameters.sparkleMin === 'number' &&
      typeof this.parameters.sparkleMax === 'number' &&
      this.parameters.sparkleMin > this.parameters.sparkleMax
    ) {
      this.parameters.sparkleMax =
        this.parameters.sparkleMin as BioluminescenceSceneParameters['sparkleMax'];
      this.sparkleMaxUniform.value = this.parameters.sparkleMax as number;
    }

    if (
      typeof this.parameters.bubbleSizeMin === 'number' &&
      typeof this.parameters.bubbleSizeMax === 'number' &&
      this.parameters.bubbleSizeMin > this.parameters.bubbleSizeMax
    ) {
      this.parameters.bubbleSizeMax =
        this.parameters.bubbleSizeMin as BioluminescenceSceneParameters['bubbleSizeMax'];
      this.bubbleSizeMax = this.parameters.bubbleSizeMax as number;
      this.bubbleSizeMaxUniform.value = this.bubbleSizeMax;
    }

    if (typeof changes.volumeWidth === 'number') {
      const clamped = Math.max(0.5, Math.min(50, changes.volumeWidth));
      if (clamped !== this.parameters.volumeWidth) {
        this.parameters.volumeWidth = clamped as BioluminescenceSceneParameters['volumeWidth'];
        needsVolume = true;
      }
    }

    if (typeof changes.volumeHeight === 'number') {
      const clamped = Math.max(0.5, Math.min(50, changes.volumeHeight));
      if (clamped !== this.parameters.volumeHeight) {
        this.parameters.volumeHeight = clamped as BioluminescenceSceneParameters['volumeHeight'];
        needsVolume = true;
      }
    }

    if (typeof changes.volumeDepth === 'number') {
      const clamped = Math.max(0.5, Math.min(50, changes.volumeDepth));
      if (clamped !== this.parameters.volumeDepth) {
        this.parameters.volumeDepth = clamped as BioluminescenceSceneParameters['volumeDepth'];
        needsVolume = true;
      }
    }

    if (typeof changes.showVolume === 'boolean') {
      this.parameters.showVolume = changes.showVolume as BioluminescenceSceneParameters['showVolume'];
      if (this.volumeWireframe) {
        this.volumeWireframe.visible = this.parameters.showVolume as boolean;
      }
    }

    if (typeof changes.baseColor === 'string') {
      this.parameters.baseColor = changes.baseColor as BioluminescenceSceneParameters['baseColor'];
      this.baseColorUniform.value.set(this.parameters.baseColor as string);
    }

    if (typeof changes.activeColor === 'string') {
      this.parameters.activeColor =
        changes.activeColor as BioluminescenceSceneParameters['activeColor'];
      this.activeColorUniform.value.set(this.parameters.activeColor as string);
    }

    if (typeof changes.baseAlpha === 'number') {
      const clamped = Math.max(0, Math.min(1, changes.baseAlpha));
      this.parameters.baseAlpha = clamped as BioluminescenceSceneParameters['baseAlpha'];
      this.baseAlphaUniform.value = clamped;
    }

    if (typeof changes.activeAlpha === 'number') {
      const clamped = Math.max(0, Math.min(1, changes.activeAlpha));
      this.parameters.activeAlpha = clamped as BioluminescenceSceneParameters['activeAlpha'];
      this.activeAlphaUniform.value = clamped;
    }

    if (typeof changes.emissiveStrength === 'number') {
      const clamped = Math.max(0, Math.min(10, changes.emissiveStrength));
      this.parameters.emissiveStrength =
        clamped as BioluminescenceSceneParameters['emissiveStrength'];
      this.emissiveStrengthUniform.value = clamped;
    }

    if (typeof changes.bloomStrength === 'number') {
      const clamped = Math.max(0, Math.min(5, changes.bloomStrength));
      this.parameters.bloomStrength = clamped as BioluminescenceSceneParameters['bloomStrength'];
      needsBloom = true;
    }

    if (typeof changes.bloomRadius === 'number') {
      const clamped = Math.max(0, Math.min(1, changes.bloomRadius));
      this.parameters.bloomRadius = clamped as BioluminescenceSceneParameters['bloomRadius'];
      needsBloom = true;
    }

    if (typeof changes.bloomThreshold === 'number') {
      const clamped = Math.max(0, Math.min(1, changes.bloomThreshold));
      this.parameters.bloomThreshold =
        clamped as BioluminescenceSceneParameters['bloomThreshold'];
      needsBloom = true;
    }

    if (typeof changes.backgroundColor === 'string') {
      this.parameters.backgroundColor =
        changes.backgroundColor as BioluminescenceSceneParameters['backgroundColor'];
      if (this.scene) {
        this.scene.background = new THREE.Color(this.parameters.backgroundColor as string);
      }
    }

    if (needsVolume) {
      this.updateGrid();
      this.updateVolumeWireframe();
      this.particleResetPending = true;
      this.fieldResetPending = true;
    }

    if (needsRebuild) {
      this.updateGrid();
      this.createParticles();
      this.updateVolumeWireframe();
    }

    if (needsBloom) {
      this.updateBloom();
    }
  }

  onResize(width: number, height: number): void {
    if (!this.camera || !this.renderer) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  cleanup(): void {
    if (this.controls) this.controls.dispose();
    if (this.particleMesh && this.scene) this.scene.remove(this.particleMesh);
    if (this.particleGeometry) this.particleGeometry.dispose();
    if (this.particleMaterial) this.particleMaterial.dispose();
    this.sparkleRandomStorage = null;
    this.energyStorage = null;
    this.activationStorage = null;
    this.refractoryStorage = null;
    this.computeUpdate = null;
    if (this.volumeWireframe && this.scene) {
      this.scene.remove(this.volumeWireframe);
      this.volumeWireframe.geometry.dispose();
      (this.volumeWireframe.material as THREE.Material).dispose();
    }
    if (this.postProcessing) this.postProcessing.dispose();
    if (this.renderer) this.renderer.dispose();
    if (this.bubbleMesh && this.scene) {
      this.scene.remove(this.bubbleMesh);
    }
    if (this.bubbleGeometry) this.bubbleGeometry.dispose();
    if (this.bubbleMaterial) this.bubbleMaterial.dispose();
    if (this.fieldVelocityTextures) {
      this.fieldVelocityTextures.forEach(texture => texture.dispose());
      this.fieldVelocityTextures = null;
    }
    this.fieldVelocitySamplers = null;
    this.fieldClearVelocityPasses = [null, null];
    this.fieldSplatPasses = [null, null];
    if (this.densityBufferAttribute) {
      try {
        this.densityBufferAttribute.dispose();
      } catch (error) {
        void error;
      }
    }
    if (this.bubbleDensityBufferAttribute) {
      try {
        this.bubbleDensityBufferAttribute.dispose();
      } catch (error) {
        void error;
      }
    }
    if (this.densityGradientTexture) {
      this.densityGradientTexture.dispose();
      this.densityGradientTexture = null;
    }
    if (this.bubbleDensityGradientTexture) {
      this.bubbleDensityGradientTexture.dispose();
      this.bubbleDensityGradientTexture = null;
    }
    if (this.bubbleInjectionTexture) {
      this.bubbleInjectionTexture.dispose();
      this.bubbleInjectionTexture = null;
    }
    this.densityGradientSampler = null;
    this.densityGradientCompute = null;
    this.bubbleDensityGradientSampler = null;
    this.bubbleDensityGradientCompute = null;
    this.bubbleDensityClearCompute = null;
    this.bubbleDensityDepositCompute = null;
    this.bubbleInjectionSampler = null;
    this.bubbleInjectionClearPass = null;
    this.bubbleInjectionCompute = null;
    this.densityBufferAttribute = null;
    this.densityBufferNode = null;
    this.densityClearCompute = null;
    this.densityDepositCompute = null;
    this.bubbleDensityBufferAttribute = null;
    this.bubbleDensityBufferNode = null;

    this.disposeCallbacks.forEach(cb => cb());
    this.disposeCallbacks.length = 0;

    this.renderer = null;
    this.scene = null;
    this.camera = null;
    this.controls = null;
    this.canvas = null;
    this.particleMesh = null;
    this.particleGeometry = null;
    this.particleMaterial = null;
    this.volumeWireframe = null;
    this.postProcessing = null;
    this.bloomNode = null;
    this.bubbleMesh = null;
    this.bubbleGeometry = null;
    this.bubbleMaterial = null;
    this.bubblePositionsStorage = null;
    this.bubbleVelocityStorage = null;
    this.bubbleComputeUpdate = null;
    this.particlePositionsStorage = null;
    this.particleVelocitiesStorage = null;
    this.particleInitCompute = null;
    this.bubblePositionsArray = null;
    this.bubbleVelocitiesArray = null;
    this.fieldVelocityTextures = null;
    this.fieldVelocitySamplers = null;
    this.fieldClearVelocityPasses = [null, null];
    this.fieldSplatPasses = [null, null];
    this.densityBufferAttribute = null;
    this.densityBufferNode = null;
    this.densityClearCompute = null;
    this.densityDepositCompute = null;
    this.bubbleDensityBufferAttribute = null;
    this.bubbleDensityBufferNode = null;
    this.bubbleDensityClearCompute = null;
    this.bubbleDensityDepositCompute = null;
    this.bubbleDensityGradientTexture = null;
    this.bubbleDensityGradientSampler = null;
    this.bubbleDensityGradientCompute = null;
    this.bubbleInjectionTexture = null;
    this.bubbleInjectionSampler = null;
    this.bubbleInjectionClearPass = null;
    this.bubbleInjectionCompute = null;
    this.densityGradientTexture = null;
    this.densityGradientSampler = null;
    this.densityGradientCompute = null;
  }

  private updateGrid(): void {
    const count = Math.max(1, Math.round(this.parameters.particleCount as number));
    const width = Math.max(0.5, this.parameters.volumeWidth as number);
    const height = Math.max(0.5, this.parameters.volumeHeight as number);
    const depth = Math.max(0.5, this.parameters.volumeDepth as number);
    const volume = Math.max(1e-6, width * height * depth);
    const cellSize = Math.cbrt(volume / count);

    let nx = Math.max(1, Math.round(width / cellSize));
    let ny = Math.max(1, Math.round(height / cellSize));
    let nz = Math.max(1, Math.round(depth / cellSize));

    const clampGrid = () => {
      nx = Math.max(1, nx);
      ny = Math.max(1, ny);
      nz = Math.max(1, nz);
    };

    clampGrid();
    while (nx * ny * nz < count) {
      const ratioX = width / nx;
      const ratioY = height / ny;
      const ratioZ = depth / nz;
      if (ratioX >= ratioY && ratioX >= ratioZ) nx += 1;
      else if (ratioY >= ratioZ) ny += 1;
      else nz += 1;
      clampGrid();
    }

    this.gridCounts = { x: nx, y: ny, z: nz };
    this.gridXUniform.value = nx;
    this.gridYUniform.value = ny;
    this.gridZUniform.value = nz;
    this.volumeSizeUniform.value.set(width, height, depth);
    this.volumeSize.set(width, height, depth);
    this.bubbleSpawnRadiusUniform.value = 0;
  }

  private createParticles(): void {
    if (!this.scene) return;

    if (this.particleMesh) {
      this.scene.remove(this.particleMesh);
      this.particleMesh = null;
    }

    this.ensureBubbles();

    const count = Math.max(1, Math.round(this.parameters.particleCount as number));
    const sparkleRandom = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      sparkleRandom[i] = Math.random();
    }
    this.sparkleRandomStorage = attributeArray(sparkleRandom, 'float');
    this.sparkleRandomStorage.setPBO(true);
    this.particlePositionsStorage = attributeArray(new Float32Array(count * 3), 'vec3');
    this.particleVelocitiesStorage = attributeArray(new Float32Array(count * 3), 'vec3');
    this.particlePositionsStorage.setPBO(true);
    this.particleVelocitiesStorage.setPBO(true);
    this.energyStorage = attributeArray(new Float32Array(count), 'float');
    this.activationStorage = attributeArray(new Float32Array(count), 'float');
    this.refractoryStorage = attributeArray(new Float32Array(count), 'float');
    this.energyStorage.setPBO(true);
    this.activationStorage.setPBO(true);
    this.refractoryStorage.setPBO(true);

    if (!this.particleGeometry) {
      this.particleGeometry = new THREE.PlaneGeometry(1, 1);
    }

    if (!this.particleMaterial) {
      this.particleMaterial = new MeshBasicNodeMaterial();
      this.particleMaterial.transparent = true;
      this.particleMaterial.depthWrite = false;
      this.particleMaterial.depthTest = false;
      this.particleMaterial.blending = THREE.AdditiveBlending;
      this.particleMaterial.side = THREE.DoubleSide;
      this.particleMaterial.toneMapped = false;
    }

    this.particleMaterial.vertexNode = this.buildVertexNode();
    this.particleMaterial.colorNode = this.buildColorNode();
    this.particleMaterial.emissiveNode = this.buildEmissiveNode();

    this.particleMesh = new THREE.InstancedMesh(this.particleGeometry, this.particleMaterial, count);
    this.particleMesh.frustumCulled = false;
    this.scene.add(this.particleMesh);

    this.ensureDensityField(count);
    this.computeUpdate = this.buildComputeUpdate(count);
    this.particleInitCompute = this.buildParticleInitCompute(count);
    if (this.renderer && this.particleInitCompute) {
      this.renderer.compute(this.particleInitCompute);
      this.particleResetPending = false;
    }
  }

  private buildVertexNode() {
    return Fn(() => {
      const particlePosition = this.buildParticlePositionNode();
      const activation = this.buildActivationNode(particlePosition);
      const size = this.particleSizeUniform.mul(
        mix(float(1.0), this.activationScaleUniform, activation)
      );

      const toCameraWorld = normalize(cameraPosition.sub(particlePosition));
      const worldUp = vec3(0.0, 1.0, 0.0);
      const worldForward = vec3(0.0, 0.0, 1.0);
      const upDot = abs(dot(toCameraWorld, worldUp));
      const useAlternateUp = upDot.greaterThan(0.9);
      const chosenUp = select(useAlternateUp, worldForward, worldUp);
      const rightVector = normalize(cross(chosenUp, toCameraWorld));
      const upVector = normalize(cross(toCameraWorld, rightVector));

      const billboardMatrix = mat3(
        rightVector.x, rightVector.y, rightVector.z,
        upVector.x, upVector.y, upVector.z,
        toCameraWorld.x, toCameraWorld.y, toCameraWorld.z
      );

      const billboardVertex = billboardMatrix.mul(positionLocal.mul(size));
      const finalPosition = vec4(particlePosition, 1.0).add(vec4(billboardVertex, 0.0));

      return cameraProjectionMatrix.mul(cameraViewMatrix).mul(finalPosition);
    })();
  }

  private buildColorNode() {
    return Fn(() => {
      const activation = this.buildActivationNode(this.buildParticlePositionNode());
      const sparkle = this.buildSparkleNode();
      const color = mix(this.baseColorUniform, this.activeColorUniform, activation);
      const alpha = mix(this.baseAlphaUniform, this.activeAlphaUniform, activation)
        .mul(sparkle);
      return vec4(color, alpha);
    })();
  }

  private buildEmissiveNode() {
    return Fn(() => {
      const activation = this.buildActivationNode(this.buildParticlePositionNode());
      const sparkle = this.buildSparkleNode();
      const color = mix(this.baseColorUniform, this.activeColorUniform, activation);
      return color.mul(this.emissiveStrengthUniform.mul(activation).mul(sparkle));
    })();
  }

  private buildActivationNode(positionNode: any) {
    if (!this.activationStorage) {
      return float(0.0);
    }
    return this.activationStorage.element(instanceIndex).clamp(0.0, 1.0);
  }

  private buildParticlePositionNode() {
    if (this.particlePositionsStorage) {
      return this.particlePositionsStorage.element(instanceIndex);
    }
    const gridX = this.gridXUniform;
    const gridY = this.gridYUniform;
    const gridZ = this.gridZUniform;

    const ix = instanceIndex.mod(gridX);
    const layer = instanceIndex.div(gridX);
    const iy = layer.mod(gridY);
    const iz = layer.div(gridY);

    const half = this.volumeSizeUniform.mul(float(0.5));
    const denomX = max(gridX, float(1.0));
    const denomY = max(gridY, float(1.0));
    const denomZ = max(gridZ, float(1.0));

    const fx = ix.toFloat().add(float(0.5)).div(denomX);
    const fy = iy.toFloat().add(float(0.5)).div(denomY);
    const fz = iz.toFloat().add(float(0.5)).div(denomZ);

    const basePos = vec3(
      fx.mul(this.volumeSizeUniform.x).sub(half.x),
      fy.mul(this.volumeSizeUniform.y).sub(half.y),
      fz.mul(this.volumeSizeUniform.z).sub(half.z)
    );

    const indexF = instanceIndex.toFloat();
    const randX = hash(indexF.mul(float(12.9898)));
    const randY = hash(indexF.mul(float(78.233)));
    const randZ = hash(indexF.mul(float(37.719)));
    const cellSize = vec3(
      this.volumeSizeUniform.x.div(denomX),
      this.volumeSizeUniform.y.div(denomY),
      this.volumeSizeUniform.z.div(denomZ)
    );
    const jitterScale = float(0.9);
    const jitter = vec3(
      randX.sub(float(0.5)),
      randY.sub(float(0.5)),
      randZ.sub(float(0.5))
    ).mul(cellSize.mul(jitterScale));

    return basePos.add(jitter).add(this.volumeCenterUniform);
  }

  private buildSparkleNode() {
    if (!this.sparkleRandomStorage) {
      return float(1.0);
    }
    const randomValue = this.sparkleRandomStorage.element(instanceIndex);
    return mix(this.sparkleMinUniform, this.sparkleMaxUniform, randomValue);
  }

  private buildParticleInitCompute(count: number) {
    return Fn(() => {
      if (!this.particlePositionsStorage || !this.particleVelocitiesStorage) {
        return;
      }
      const position = this.particlePositionsStorage.element(instanceIndex);
      const velocity = this.particleVelocitiesStorage.element(instanceIndex);

      const gridX = this.gridXUniform;
      const gridY = this.gridYUniform;
      const gridZ = this.gridZUniform;
      const ix = instanceIndex.mod(gridX);
      const layer = instanceIndex.div(gridX);
      const iy = layer.mod(gridY);
      const iz = layer.div(gridY);

      const half = this.volumeSizeUniform.mul(float(0.5));
      const denomX = max(gridX, float(1.0));
      const denomY = max(gridY, float(1.0));
      const denomZ = max(gridZ, float(1.0));

      const fx = ix.toFloat().add(float(0.5)).div(denomX);
      const fy = iy.toFloat().add(float(0.5)).div(denomY);
      const fz = iz.toFloat().add(float(0.5)).div(denomZ);

      const basePos = vec3(
        fx.mul(this.volumeSizeUniform.x).sub(half.x),
        fy.mul(this.volumeSizeUniform.y).sub(half.y),
        fz.mul(this.volumeSizeUniform.z).sub(half.z)
      );

      const indexF = instanceIndex.toFloat();
      const randX = hash(indexF.mul(float(12.9898)));
      const randY = hash(indexF.mul(float(78.233)));
      const randZ = hash(indexF.mul(float(37.719)));
      const cellSize = vec3(
        this.volumeSizeUniform.x.div(denomX),
        this.volumeSizeUniform.y.div(denomY),
        this.volumeSizeUniform.z.div(denomZ)
      );
      const jitterScale = float(0.9);
      const jitter = vec3(
        randX.sub(float(0.5)),
        randY.sub(float(0.5)),
        randZ.sub(float(0.5))
      ).mul(cellSize.mul(jitterScale));
      const spawnPos = basePos.add(jitter).add(this.volumeCenterUniform);

      position.x.assign(spawnPos.x);
      position.y.assign(spawnPos.y);
      position.z.assign(spawnPos.z);

      velocity.x.assign(float(0.0));
      velocity.y.assign(float(0.0));
      velocity.z.assign(float(0.0));
    })().compute(count);
  }

  private buildComputeUpdate(count: number) {
    return Fn(() => {
      if (
        !this.energyStorage ||
        !this.activationStorage ||
        !this.refractoryStorage ||
        !this.particlePositionsStorage ||
        !this.particleVelocitiesStorage ||
        !this.fieldVelocitySamplers ||
        !this.densityGradientSampler
      ) {
        return;
      }

      const energy = this.energyStorage.element(instanceIndex).toVar();
      const activation = this.activationStorage.element(instanceIndex).toVar();
      const refractory = this.refractoryStorage.element(instanceIndex).toVar();
      const position = this.particlePositionsStorage.element(instanceIndex).toVar();
      const velocity = this.particleVelocitiesStorage.element(instanceIndex).toVar();

      const res = vec3(this.fieldResXUniform, this.fieldResYUniform, this.fieldResZUniform);
      const invRes = vec3(float(1.0), float(1.0), float(1.0)).div(res);
      const halfTexel = invRes.mul(float(0.5));
      const maxUv = vec3(float(1.0), float(1.0), float(1.0)).sub(halfTexel);
      const local = position.sub(this.volumeCenterUniform)
        .div(this.volumeSizeUniform)
        .add(float(0.5));
      const indexF = instanceIndex.toFloat();
      const fieldJitterScale = float(0.75);
      const fieldJitter = vec3(
        hash(indexF.mul(float(19.21))).sub(float(0.5)),
        hash(indexF.mul(float(83.17))).sub(float(0.5)),
        hash(indexF.mul(float(41.53))).sub(float(0.5))
      ).mul(invRes.mul(fieldJitterScale));
      const uv = clamp(local.add(fieldJitter), halfTexel, maxUv);
      const velocityA = this.fieldVelocitySamplers[0].sample(uv).xyz;
      const velocityB = this.fieldVelocitySamplers[1].sample(uv).xyz;
      const useB = this.fieldVelocityIndexUniform.greaterThan(float(0.5));
      const fieldVelocity = select(useB, velocityB, velocityA);

      const speed = length(fieldVelocity);
      const signal = clamp(
        speed.mul(this.activationVelocityScaleUniform),
        float(0.0),
        float(1.0)
      );

      const sparkle = this.buildSparkleNode();
      const dt = this.deltaTimeUniform;

      refractory.assign(max(refractory.sub(dt), float(0.0)));

      const energyGain = signal.mul(this.energyAccumulationUniform).mul(sparkle).mul(dt);
      const energyLoss = this.energyDecayUniform.mul(dt).mul(float(1.0).sub(signal));
      energy.addAssign(energyGain);
      energy.subAssign(energyLoss);
      energy.assign(max(energy, float(0.0)));

      const threshold = this.activationThresholdUniform.div(max(sparkle, float(0.01)));
      const canActivate = refractory.equal(float(0.0)).and(energy.greaterThan(threshold));

      If(canActivate, () => {
        activation.assign(float(1.0));
        energy.assign(float(0.0));
        refractory.assign(this.refractoryPeriodUniform);
      });

      const decay = dt.div(max(this.activationDurationUniform, float(0.01)));
      activation.assign(max(activation.sub(decay), float(0.0)));

      const dtMove = this.deltaTimeUniform;
      const acceleration = fieldVelocity.sub(velocity).mul(this.particleWakeStrengthUniform);
      velocity.addAssign(acceleration.mul(dtMove));
      velocity.y.addAssign(this.particleGravityUniform.mul(dtMove));
      const twoPi = float(6.283185307179586);
      const phase1 = hash(indexF.mul(float(12.9898))).mul(twoPi);
      const phase2 = hash(indexF.mul(float(78.233))).mul(twoPi);
      const phase3 = hash(indexF.mul(float(37.719))).mul(twoPi);
      const noiseTime = this.timeUniform.mul(this.particleNoiseFrequencyUniform);
      const noise = vec3(
        sin(noiseTime.add(phase1)),
        cos(noiseTime.mul(float(1.31)).add(phase2)),
        sin(noiseTime.mul(float(0.73)).add(phase3))
      );
      velocity.addAssign(noise.mul(this.particleNoiseStrengthUniform).mul(dtMove));
      const densityRes = vec3(this.densityResXUniform, this.densityResYUniform, this.densityResZUniform);
      const densityInvRes = vec3(float(1.0), float(1.0), float(1.0)).div(densityRes);
      const densityHalfTexel = densityInvRes.mul(float(0.5));
      const densityMaxUv = vec3(float(1.0), float(1.0), float(1.0)).sub(densityHalfTexel);
      const jitterScale = float(0.75);
      const jitter = vec3(
        hash(indexF.mul(float(91.37))).sub(float(0.5)),
        hash(indexF.mul(float(47.11))).sub(float(0.5)),
        hash(indexF.mul(float(12.63))).sub(float(0.5))
      ).mul(densityInvRes.mul(jitterScale));
      const densityUv = clamp(local.add(jitter), densityHalfTexel, densityMaxUv);
      const densityGrad = this.densityGradientSampler.sample(densityUv).xyz;
      velocity.addAssign(densityGrad.negate().mul(this.particleDensityStrengthUniform).mul(dtMove));
      const dragScale = float(1.0)
        .div(float(1.0).add(this.particleDragUniform.mul(dtMove)));
      velocity.mulAssign(dragScale);
      const noiseDragScale = float(1.0)
        .div(float(1.0).add(this.particleNoiseDragUniform.mul(dtMove)));
      velocity.mulAssign(noiseDragScale);
      position.addAssign(velocity.mul(dtMove));

      const center = this.volumeCenterUniform;
      const half = this.volumeSizeUniform.mul(float(0.5));
      position.x.assign(clamp(position.x, center.x.sub(half.x), center.x.add(half.x)));
      position.y.assign(clamp(position.y, center.y.sub(half.y), center.y.add(half.y)));
      position.z.assign(clamp(position.z, center.z.sub(half.z), center.z.add(half.z)));

      this.energyStorage.element(instanceIndex).assign(energy);
      this.activationStorage.element(instanceIndex).assign(activation);
      this.refractoryStorage.element(instanceIndex).assign(refractory);
      this.particlePositionsStorage.element(instanceIndex).assign(position);
      this.particleVelocitiesStorage.element(instanceIndex).assign(velocity);
    })().compute(count);
  }

  private buildBubbleComputeUpdate() {
    return Fn(() => {
      if (!this.bubblePositionsStorage || !this.bubbleVelocityStorage || !this.bubbleDensityGradientSampler) {
        return;
      }

      const pos = this.bubblePositionsStorage.element(instanceIndex).toVar();
      const velState = this.bubbleVelocityStorage.element(instanceIndex).toVar();
      const velX = velState.x.toVar();
      const velY = velState.y.toVar();
      const velZ = velState.z.toVar();
      const seedOffset = velState.w.toVar();

      const emit = this.bubbleEmitUniform;
      const reset = this.bubbleResetUniform;
      const dt = this.deltaTimeUniform;
      const spawnRate = this.bubbleSpawnRateUniform;
      const spawnRateActive = spawnRate.greaterThan(float(0.0));
      const poolSize = float(this.bubblePoolSize);
      const spawnStart = this.timeUniform.sub(dt).mul(spawnRate);
      const spawnEnd = this.timeUniform.mul(spawnRate);
      const spawnTotal = spawnEnd.sub(spawnStart);
      const spawnCount = clamp(spawnTotal.floor(), float(0.0), poolSize);
      const spawnFrac = spawnTotal.sub(spawnCount);
      const spawnStartSlot = spawnStart.floor().mod(poolSize);
      const offset = instanceIndex.toFloat().sub(spawnStartSlot);
      const offsetWrapped = offset.add(select(offset.lessThan(float(0.0)), poolSize, float(0.0)));
      const inBurst = offsetWrapped.lessThan(spawnCount);
      const extraSlot = spawnStartSlot.add(spawnCount).mod(poolSize);
      const extraRand = hash(instanceIndex.toFloat().add(this.timeUniform.mul(float(0.73))));
      const extraSpawn = instanceIndex.toFloat().equal(extraSlot)
        .and(extraRand.lessThan(spawnFrac));

      const center = this.volumeCenterUniform;
      const half = this.volumeSizeUniform.mul(float(0.5));
      const bottom = center.y.sub(half.y);
      const top = center.y.add(half.y);

      const indexF = instanceIndex.toFloat();
      const driftRand1 = hash(indexF.mul(float(127.1)));
      const driftRand2 = hash(indexF.mul(float(311.7)));
      const driftRand3 = hash(indexF.mul(float(74.7)));
      const sizeRand = hash(indexF.mul(float(19.19)));
      const radiusMm = mix(this.bubbleSizeMinUniform, this.bubbleSizeMaxUniform, sizeRand);
      const radius = radiusMm.mul(float(0.001));
      const radiusRef = float(0.005);
      const twoPi = float(6.283185307179586);
      const driftTime = this.timeUniform.mul(this.bubbleDriftFrequencyUniform)
        .add(driftRand3.mul(float(10.0)));
      const waveX = sin(driftTime.add(driftRand1.mul(twoPi)))
        .add(sin(driftTime.mul(float(2.1)).add(driftRand2.mul(twoPi))).mul(float(0.5)));
      const waveZ = cos(driftTime.add(driftRand2.mul(twoPi)))
        .add(cos(driftTime.mul(float(1.7)).add(driftRand1.mul(twoPi))).mul(float(0.5)));
      const sizeDrift = mix(float(1.2), float(0.6), driftRand1);
      const driftScale = this.bubbleDriftStrengthUniform.mul(sizeDrift);
      const dragFactor = max(
        float(0.0),
        float(1.0).sub(this.bubbleDriftDragUniform.mul(dt))
      );

      const emitActive = emit.greaterThan(float(0.5));
      const needsSpawn = pos.w.lessThan(float(0.5))
        .or(reset.greaterThan(float(0.5)))
        .or(pos.y.greaterThan(top));

      If(reset.greaterThan(float(0.5)), () => {
        pos.w.assign(float(0.0));
        velX.assign(float(0.0));
        velY.assign(float(0.0));
        velZ.assign(float(0.0));
      });

      const spawnAllowed = emitActive
        .and(spawnRateActive)
        .and(needsSpawn)
        .and(inBurst.or(extraSpawn));

      If(spawnAllowed, () => {
        const spawnSeed = seedOffset.add(float(1.0)).toVar();
        seedOffset.assign(spawnSeed);
        const spawnBase = indexF.add(spawnSeed.mul(float(17.0)));
        const spawnRand1 = hash(spawnBase.mul(float(269.5)));
        const spawnRand2 = hash(spawnBase.mul(float(183.3)));
        const spawnArea = this.bubbleSpawnAreaUniform;
        const spawnLocalX = spawnRand1.sub(float(0.5)).mul(this.volumeSizeUniform.x);
        const spawnLocalZ = spawnRand2.sub(float(0.5)).mul(this.volumeSizeUniform.z);
        const spawnPosX = center.x.add(spawnLocalX.mul(spawnArea));
        const spawnPosZ = center.z.add(spawnLocalZ.mul(spawnArea));
        pos.x.assign(spawnPosX);
        pos.y.assign(bottom);
        pos.z.assign(spawnPosZ);
        pos.w.assign(float(1.0));
        velX.assign(float(0.0));
        velY.assign(float(0.0));
        velZ.assign(float(0.0));
      });

      If(pos.w.greaterThan(float(0.5)), () => {
        const repelStrength = this.bubbleRepelStrengthUniform
          .mul(this.bubbleRepelRadiusUniform)
          .mul(radius.div(max(radiusRef, float(1e-6))));
        const bubbleRes = vec3(this.densityResXUniform, this.densityResYUniform, this.densityResZUniform);
        const bubbleInvRes = vec3(float(1.0), float(1.0), float(1.0)).div(bubbleRes);
        const bubbleHalfTexel = bubbleInvRes.mul(float(0.5));
        const bubbleMaxUv = vec3(float(1.0), float(1.0), float(1.0)).sub(bubbleHalfTexel);
        const bubbleUv = clamp(
          pos.xyz.sub(center).div(this.volumeSizeUniform).add(float(0.5)),
          bubbleHalfTexel,
          bubbleMaxUv
        );
        const bubbleDensityGrad = this.bubbleDensityGradientSampler.sample(bubbleUv).xyz;
        velX.addAssign(bubbleDensityGrad.x.negate().mul(repelStrength).mul(dt));
        velY.addAssign(bubbleDensityGrad.y.negate().mul(repelStrength).mul(dt));
        velZ.addAssign(bubbleDensityGrad.z.negate().mul(repelStrength).mul(dt));
        velX.addAssign(waveX.mul(driftScale).mul(dt));
        velZ.addAssign(waveZ.mul(driftScale).mul(dt));
        velX.mulAssign(dragFactor);
        velZ.mulAssign(dragFactor);

        const netAccel = this.activationSpeedUniform.sub(this.bubbleGravityUniform);
        velY.addAssign(netAccel.mul(dt));

        const dragCoeff = this.bubbleViscosityUniform
          .div(max(radius.mul(radius), float(1e-6)));
        const dragScale = float(1.0).div(float(1.0).add(dragCoeff.mul(dt)));
        velX.mulAssign(dragScale);
        velY.mulAssign(dragScale);
        velZ.mulAssign(dragScale);

        pos.x.addAssign(velX.mul(dt));
        pos.z.addAssign(velZ.mul(dt));
        pos.y.addAssign(velY.mul(dt));

        pos.x.assign(clamp(pos.x, center.x.sub(half.x), center.x.add(half.x)));
        pos.z.assign(clamp(pos.z, center.z.sub(half.z), center.z.add(half.z)));

        If(pos.y.greaterThan(top), () => {
          pos.w.assign(float(0.0));
        });
      });

      this.bubblePositionsStorage.element(instanceIndex).assign(pos);
      this.bubbleVelocityStorage.element(instanceIndex)
        .assign(vec4(velX, velY, velZ, seedOffset));
    })().compute(this.bubblePoolSize);
  }

  private ensureBubbles(): void {
    if (!this.scene) return;
    if (!this.bubblePositionsArray || this.bubblePositionsArray.length !== this.bubblePoolSize * 4) {
      this.bubblePositionsArray = new Float32Array(this.bubblePoolSize * 4);
      this.bubbleVelocitiesArray = new Float32Array(this.bubblePoolSize * 4);
      for (let i = 0; i < this.bubblePoolSize; i++) {
        this.bubblePositionsArray[i * 4 + 3] = 0;
      }
    }

    if (!this.bubblePositionsStorage) {
      this.bubblePositionsStorage = attributeArray(this.bubblePositionsArray, 'vec4');
      this.bubblePositionsStorage.setPBO(true);
    }
    if (!this.bubbleVelocityStorage && this.bubbleVelocitiesArray) {
      this.bubbleVelocityStorage = attributeArray(this.bubbleVelocitiesArray, 'vec4');
      this.bubbleVelocityStorage.setPBO(true);
    }

    if (!this.bubbleGeometry) {
      this.bubbleGeometry = new THREE.SphereGeometry(1, 12, 10);
    }
    if (!this.bubbleMaterial) {
      const material = new MeshBasicNodeMaterial();
      material.color = this.bubbleColor.clone();
      material.transparent = true;
      material.opacity = 0.7;
      material.blending = THREE.AdditiveBlending;
      material.depthWrite = false;
      material.depthTest = false;
      material.toneMapped = false;
      material.positionNode = Fn(() => {
        if (!this.bubblePositionsStorage) {
          return positionLocal;
        }
        const bubble = this.bubblePositionsStorage.element(instanceIndex);
        const active = bubble.w;
        const indexF = instanceIndex.toFloat();
        const sizeRand = hash(indexF.mul(float(19.19)));
        const radiusMm = mix(this.bubbleSizeMinUniform, this.bubbleSizeMaxUniform, sizeRand);
        const radius = radiusMm.mul(float(0.001));
        const size = radius.mul(active);
        return positionLocal.mul(size).add(bubble.xyz);
      })();
      this.bubbleMaterial = material;
    }
    if (!this.bubbleMesh) {
      this.bubbleMesh = new THREE.InstancedMesh(
        this.bubbleGeometry,
        this.bubbleMaterial,
        this.bubblePoolSize
      );
      this.bubbleMesh.frustumCulled = false;
      this.scene.add(this.bubbleMesh);
    }

    if (!this.bubbleComputeUpdate && this.bubbleDensityGradientSampler) {
      this.bubbleComputeUpdate = this.buildBubbleComputeUpdate();
    }
  }

  private ensureFluidField(): void {
    if (!this.renderer) return;
    this.ensureBubbles();

    const resX = Math.max(8, Math.min(128, Math.round(this.fieldResXUniform.value as number)));
    const resY = Math.max(8, Math.min(128, Math.round(this.fieldResYUniform.value as number)));
    const resZ = Math.max(8, Math.min(128, Math.round(this.fieldResZUniform.value as number)));
    const needsRebuild =
      !this.fieldVelocityTextures ||
      resX !== Math.round(this.fieldResolution.x) ||
      resY !== Math.round(this.fieldResolution.y) ||
      resZ !== Math.round(this.fieldResolution.z);

    if (!needsRebuild) {
      return;
    }

    if (this.fieldVelocityTextures) {
      this.fieldVelocityTextures.forEach(texture => texture.dispose());
    }

    this.fieldResolution.set(resX, resY, resZ);

    const velocityA = new Storage3DTexture(resX, resY, resZ);
    const velocityB = new Storage3DTexture(resX, resY, resZ);

    this.configureFieldTexture(velocityA, 'BioluminescenceVelocityA');
    this.configureFieldTexture(velocityB, 'BioluminescenceVelocityB');

    this.fieldVelocityTextures = [velocityA, velocityB];
    this.fieldVelocitySamplers = [
      texture3D(velocityA as any, null, 0),
      texture3D(velocityB as any, null, 0)
    ];

    const totalCells = Math.max(1, Math.floor(resX * resY * resZ));

    this.fieldClearVelocityPasses = [
      this.buildFieldClearPass(velocityA, totalCells),
      this.buildFieldClearPass(velocityB, totalCells)
    ];
    this.fieldSplatPasses = [null, null];
    this.fieldVelocityReadIndex = 0;
    this.fieldVelocityIndexUniform.value = 0;
    this.fieldResetPending = true;

    if (this.particlePositionsStorage) {
      const count = Math.max(1, Math.round(this.parameters.particleCount as number));
      this.computeUpdate = this.buildComputeUpdate(count);
    }
  }

  private ensureDensityField(particleCount?: number): void {
    if (!this.renderer) return;
    if (!this.particlePositionsStorage) return;

    const count = Math.max(
      1,
      Math.round(
        typeof particleCount === 'number'
          ? particleCount
          : (this.parameters.particleCount as number)
      )
    );
    const resX = Math.max(4, Math.min(128, Math.round(this.densityResXUniform.value as number)));
    const resY = Math.max(4, Math.min(128, Math.round(this.densityResYUniform.value as number)));
    const resZ = Math.max(4, Math.min(128, Math.round(this.densityResZUniform.value as number)));
    const needsRebuild =
      !this.densityBufferAttribute ||
      !this.densityGradientTexture ||
      !this.bubbleDensityBufferAttribute ||
      !this.bubbleDensityGradientTexture ||
      !this.bubbleInjectionTexture ||
      resX !== Math.round(this.densityResolution.x) ||
      resY !== Math.round(this.densityResolution.y) ||
      resZ !== Math.round(this.densityResolution.z);

    if (needsRebuild) {
      if (this.densityBufferAttribute) {
        try {
          this.densityBufferAttribute.dispose();
        } catch (error) {
          void error;
        }
      }
      if (this.bubbleDensityBufferAttribute) {
        try {
          this.bubbleDensityBufferAttribute.dispose();
        } catch (error) {
          void error;
        }
      }
      if (this.densityGradientTexture) {
        this.densityGradientTexture.dispose();
      }
      if (this.bubbleDensityGradientTexture) {
        this.bubbleDensityGradientTexture.dispose();
      }
      if (this.bubbleInjectionTexture) {
        this.bubbleInjectionTexture.dispose();
      }
      const cellCount = Math.max(1, Math.floor(resX * resY * resZ));
      this.densityBufferAttribute = new StorageBufferAttribute(
        new Uint32Array(cellCount),
        1
      );
      this.densityBufferAttribute.needsUpdate = true;
      this.densityBufferNode = storage(this.densityBufferAttribute, 'uint', cellCount);
      this.bubbleDensityBufferAttribute = new StorageBufferAttribute(
        new Uint32Array(cellCount),
        1
      );
      this.bubbleDensityBufferAttribute.needsUpdate = true;
      this.bubbleDensityBufferNode = storage(this.bubbleDensityBufferAttribute, 'uint', cellCount);
      this.densityResolution.set(resX, resY, resZ);
      const gradientTexture = new Storage3DTexture(resX, resY, resZ);
      this.configureFieldTexture(gradientTexture, 'BioluminescenceDensityGradient');
      this.densityGradientTexture = gradientTexture;
      this.densityGradientSampler = texture3D(gradientTexture as any, null, 0);
      const bubbleGradientTexture = new Storage3DTexture(resX, resY, resZ);
      this.configureFieldTexture(bubbleGradientTexture, 'BioluminescenceBubbleDensityGradient');
      this.bubbleDensityGradientTexture = bubbleGradientTexture;
      this.bubbleDensityGradientSampler = texture3D(bubbleGradientTexture as any, null, 0);
      const injectionTexture = new Storage3DTexture(resX, resY, resZ);
      this.configureFieldTexture(injectionTexture, 'BioluminescenceBubbleInjection');
      this.bubbleInjectionTexture = injectionTexture;
      this.bubbleInjectionSampler = texture3D(injectionTexture as any, null, 0);
      this.densityClearCompute = this.buildDensityClearCompute(cellCount);
      this.densityDepositCompute = null;
      this.densityGradientCompute = this.buildDensityGradientCompute(cellCount);
      this.bubbleDensityClearCompute = this.buildBubbleDensityClearCompute(cellCount);
      this.bubbleDensityDepositCompute = null;
      this.bubbleDensityGradientCompute = this.buildBubbleDensityGradientCompute(cellCount);
      this.bubbleInjectionClearPass = this.buildTextureClearPass(
        injectionTexture,
        this.densityResXUniform,
        this.densityResYUniform,
        this.densityResZUniform,
        cellCount
      );
      this.bubbleInjectionCompute = this.buildBubbleInjectionCompute(cellCount);
      this.bubbleComputeUpdate = null;
      this.densityParticleCount = 0;
    }

    if (!this.densityGradientSampler && this.densityGradientTexture) {
      this.densityGradientSampler = texture3D(this.densityGradientTexture as any, null, 0);
    }
    if (!this.bubbleDensityGradientSampler && this.bubbleDensityGradientTexture) {
      this.bubbleDensityGradientSampler =
        texture3D(this.bubbleDensityGradientTexture as any, null, 0);
    }
    if (!this.bubbleInjectionSampler && this.bubbleInjectionTexture) {
      this.bubbleInjectionSampler = texture3D(this.bubbleInjectionTexture as any, null, 0);
    }
    if (!this.densityGradientCompute && this.densityBufferAttribute) {
      const cellCount = Math.max(1, Math.floor(resX * resY * resZ));
      this.densityGradientCompute = this.buildDensityGradientCompute(cellCount);
    }
    if (!this.bubbleDensityGradientCompute && this.bubbleDensityBufferAttribute) {
      const cellCount = Math.max(1, Math.floor(resX * resY * resZ));
      this.bubbleDensityGradientCompute = this.buildBubbleDensityGradientCompute(cellCount);
    }
    if (!this.bubbleDensityClearCompute && this.bubbleDensityBufferAttribute) {
      const cellCount = Math.max(1, Math.floor(resX * resY * resZ));
      this.bubbleDensityClearCompute = this.buildBubbleDensityClearCompute(cellCount);
    }
    if (!this.bubbleDensityDepositCompute && this.bubbleDensityBufferAttribute) {
      this.bubbleDensityDepositCompute = this.buildBubbleDensityDepositCompute(this.bubblePoolSize);
    }
    if (!this.bubbleInjectionClearPass && this.bubbleInjectionTexture) {
      const cellCount = Math.max(1, Math.floor(resX * resY * resZ));
      this.bubbleInjectionClearPass = this.buildTextureClearPass(
        this.bubbleInjectionTexture,
        this.densityResXUniform,
        this.densityResYUniform,
        this.densityResZUniform,
        cellCount
      );
    }
    if (!this.bubbleInjectionCompute && this.bubbleInjectionTexture) {
      const cellCount = Math.max(1, Math.floor(resX * resY * resZ));
      this.bubbleInjectionCompute = this.buildBubbleInjectionCompute(cellCount);
    }

    if (
      (needsRebuild || !this.fieldSplatPasses[0]) &&
      this.fieldVelocityTextures &&
      this.fieldVelocitySamplers &&
      this.bubbleInjectionSampler
    ) {
      const fieldCells = Math.max(
        1,
        Math.floor(
          Math.round(this.fieldResolution.x) *
            Math.round(this.fieldResolution.y) *
            Math.round(this.fieldResolution.z)
        )
      );
      this.fieldSplatPasses = [
        this.buildFieldSplatPass(
          this.fieldVelocitySamplers[0],
          this.bubbleInjectionSampler,
          this.fieldVelocityTextures[1],
          fieldCells
        ),
        this.buildFieldSplatPass(
          this.fieldVelocitySamplers[1],
          this.bubbleInjectionSampler,
          this.fieldVelocityTextures[0],
          fieldCells
        )
      ];
      this.fieldVelocityReadIndex = 0;
      this.fieldVelocityIndexUniform.value = 0;
      this.fieldResetPending = true;
    }

    if (
      !this.bubbleComputeUpdate &&
      this.bubblePositionsStorage &&
      this.bubbleVelocityStorage &&
      this.bubbleDensityGradientSampler
    ) {
      this.bubbleComputeUpdate = this.buildBubbleComputeUpdate();
    }

    if (this.densityParticleCount !== count || !this.densityDepositCompute) {
      this.densityDepositCompute = this.buildDensityDepositCompute(count);
      this.densityParticleCount = count;
    }
  }

  private buildDensityClearCompute(cellCount: number) {
    return Fn(() => {
      if (!this.densityBufferNode) return;
      const idx = instanceIndex.toUint();
      const densityAtomic = this.densityBufferNode.toAtomic();
      atomicStore(densityAtomic.element(idx), uint(0));
    })().compute(cellCount);
  }

  private buildDensityDepositCompute(count: number) {
    return Fn(() => {
      if (!this.densityBufferNode || !this.particlePositionsStorage) {
        return;
      }
      const densityAtomic = this.densityBufferNode.toAtomic();
      const position = this.particlePositionsStorage.element(instanceIndex);
      const resXf = max(this.densityResXUniform, float(1)).toVar();
      const resYf = max(this.densityResYUniform, float(1)).toVar();
      const resZf = max(this.densityResZUniform, float(1)).toVar();
      const maxX = max(resXf.sub(float(1)), float(0));
      const maxY = max(resYf.sub(float(1)), float(0));
      const maxZ = max(resZf.sub(float(1)), float(0));

      const local = position.sub(this.volumeCenterUniform)
        .div(this.volumeSizeUniform)
        .add(float(0.5));
      const sampleX = clamp(local.x.mul(resXf), float(0), maxX).toVar();
      const sampleY = clamp(local.y.mul(resYf), float(0), maxY).toVar();
      const sampleZ = clamp(local.z.mul(resZf), float(0), maxZ).toVar();
      const ixFloat = sampleX.floor().toVar();
      const iyFloat = sampleY.floor().toVar();
      const izFloat = sampleZ.floor().toVar();
      const ix = ixFloat.toUint();
      const iy = iyFloat.toUint();
      const iz = izFloat.toUint();
      const resXUint = resXf.floor().toUint();
      const resYUint = resYf.floor().toUint();
      const resXYUint = resXUint.mul(resYUint);
      const cellIndex = iz.mul(resXYUint).add(iy.mul(resXUint)).add(ix);
      atomicAdd(densityAtomic.element(cellIndex), uint(1));
    })().compute(count);
  }

  private buildDensityGradientCompute(cellCount: number) {
    return Fn(() => {
      if (!this.densityBufferNode || !this.densityGradientTexture) {
        return;
      }
      const resX = this.densityResXUniform;
      const resY = this.densityResYUniform;
      const resZ = this.densityResZUniform;
      const resXY = resX.mul(resY);
      const x = instanceIndex.mod(resX);
      const y = instanceIndex.div(resX).mod(resY);
      const z = instanceIndex.div(resXY);

      const resXf = max(resX, float(1)).toVar();
      const resYf = max(resY, float(1)).toVar();
      const resZf = max(resZ, float(1)).toVar();
      const resXUint = resXf.floor().toUint();
      const resYUint = resYf.floor().toUint();
      const resXYUint = resXUint.mul(resYUint);
      const cellIndex = z.toUint().mul(resXYUint).add(y.toUint().mul(resXUint)).add(x.toUint());

      const densityAtomic = this.densityBufferNode.toAtomic();
      const densityCenter = float(
        atomicAdd(densityAtomic.element(cellIndex), uint(0))
      ).toVar();
      const densityLeft = densityCenter.toVar();
      If(x.greaterThan(float(0)), () => {
        densityLeft.assign(
          float(atomicAdd(densityAtomic.element(cellIndex.sub(uint(1))), uint(0)))
        );
      });
      const densityRight = densityCenter.toVar();
      If(x.lessThan(resXf.sub(float(1))), () => {
        densityRight.assign(
          float(atomicAdd(densityAtomic.element(cellIndex.add(uint(1))), uint(0)))
        );
      });
      const densityDown = densityCenter.toVar();
      If(y.greaterThan(float(0)), () => {
        densityDown.assign(
          float(atomicAdd(densityAtomic.element(cellIndex.sub(resXUint)), uint(0)))
        );
      });
      const densityUp = densityCenter.toVar();
      If(y.lessThan(resYf.sub(float(1))), () => {
        densityUp.assign(
          float(atomicAdd(densityAtomic.element(cellIndex.add(resXUint)), uint(0)))
        );
      });
      const densityBack = densityCenter.toVar();
      If(z.greaterThan(float(0)), () => {
        densityBack.assign(
          float(atomicAdd(densityAtomic.element(cellIndex.sub(resXYUint)), uint(0)))
        );
      });
      const densityFront = densityCenter.toVar();
      If(z.lessThan(resZf.sub(float(1))), () => {
        densityFront.assign(
          float(atomicAdd(densityAtomic.element(cellIndex.add(resXYUint)), uint(0)))
        );
      });

      const densitySpacing = this.volumeSizeUniform
        .div(vec3(resXf, resYf, resZf));
      const invDx = float(1.0).div(max(densitySpacing.x, float(1e-4)));
      const invDy = float(1.0).div(max(densitySpacing.y, float(1e-4)));
      const invDz = float(1.0).div(max(densitySpacing.z, float(1e-4)));
      const densityGrad = vec3(
        densityRight.sub(densityLeft).mul(invDx),
        densityUp.sub(densityDown).mul(invDy),
        densityFront.sub(densityBack).mul(invDz)
      ).mul(float(0.5));

      const coord = uvec3(x.toUint(), y.toUint(), z.toUint());
      textureStore(this.densityGradientTexture, coord, vec4(densityGrad, float(1.0))).toWriteOnly();
    })().compute(cellCount);
  }

  private buildBubbleDensityClearCompute(cellCount: number) {
    return Fn(() => {
      if (!this.bubbleDensityBufferNode) return;
      const idx = instanceIndex.toUint();
      const densityAtomic = this.bubbleDensityBufferNode.toAtomic();
      atomicStore(densityAtomic.element(idx), uint(0));
    })().compute(cellCount);
  }

  private buildBubbleDensityDepositCompute(count: number) {
    return Fn(() => {
      if (!this.bubbleDensityBufferNode || !this.bubblePositionsStorage) {
        return;
      }
      const densityAtomic = this.bubbleDensityBufferNode.toAtomic();
      const bubble = this.bubblePositionsStorage.element(instanceIndex);
      const active = bubble.w.greaterThan(float(0.5));
      const resXf = max(this.densityResXUniform, float(1)).toVar();
      const resYf = max(this.densityResYUniform, float(1)).toVar();
      const resZf = max(this.densityResZUniform, float(1)).toVar();
      const maxX = max(resXf.sub(float(1)), float(0));
      const maxY = max(resYf.sub(float(1)), float(0));
      const maxZ = max(resZf.sub(float(1)), float(0));

      const local = bubble.xyz.sub(this.volumeCenterUniform)
        .div(this.volumeSizeUniform)
        .add(float(0.5));
      const sampleX = clamp(local.x.mul(resXf), float(0), maxX).toVar();
      const sampleY = clamp(local.y.mul(resYf), float(0), maxY).toVar();
      const sampleZ = clamp(local.z.mul(resZf), float(0), maxZ).toVar();
      const ixFloat = sampleX.floor().toVar();
      const iyFloat = sampleY.floor().toVar();
      const izFloat = sampleZ.floor().toVar();
      const ix = ixFloat.toUint();
      const iy = iyFloat.toUint();
      const iz = izFloat.toUint();
      const resXUint = resXf.floor().toUint();
      const resYUint = resYf.floor().toUint();
      const resXYUint = resXUint.mul(resYUint);
      const cellIndex = iz.mul(resXYUint).add(iy.mul(resXUint)).add(ix);
      If(active, () => {
        atomicAdd(densityAtomic.element(cellIndex), uint(1));
      });
    })().compute(count);
  }

  private buildBubbleDensityGradientCompute(cellCount: number) {
    return Fn(() => {
      if (!this.bubbleDensityBufferNode || !this.bubbleDensityGradientTexture) {
        return;
      }
      const resX = this.densityResXUniform;
      const resY = this.densityResYUniform;
      const resZ = this.densityResZUniform;
      const resXY = resX.mul(resY);
      const x = instanceIndex.mod(resX);
      const y = instanceIndex.div(resX).mod(resY);
      const z = instanceIndex.div(resXY);

      const resXf = max(resX, float(1)).toVar();
      const resYf = max(resY, float(1)).toVar();
      const resZf = max(resZ, float(1)).toVar();
      const resXUint = resXf.floor().toUint();
      const resYUint = resYf.floor().toUint();
      const resXYUint = resXUint.mul(resYUint);
      const cellIndex = z.toUint().mul(resXYUint).add(y.toUint().mul(resXUint)).add(x.toUint());

      const densityAtomic = this.bubbleDensityBufferNode.toAtomic();
      const densityCenter = float(
        atomicAdd(densityAtomic.element(cellIndex), uint(0))
      ).toVar();
      const densityLeft = densityCenter.toVar();
      If(x.greaterThan(float(0)), () => {
        densityLeft.assign(
          float(atomicAdd(densityAtomic.element(cellIndex.sub(uint(1))), uint(0)))
        );
      });
      const densityRight = densityCenter.toVar();
      If(x.lessThan(resXf.sub(float(1))), () => {
        densityRight.assign(
          float(atomicAdd(densityAtomic.element(cellIndex.add(uint(1))), uint(0)))
        );
      });
      const densityDown = densityCenter.toVar();
      If(y.greaterThan(float(0)), () => {
        densityDown.assign(
          float(atomicAdd(densityAtomic.element(cellIndex.sub(resXUint)), uint(0)))
        );
      });
      const densityUp = densityCenter.toVar();
      If(y.lessThan(resYf.sub(float(1))), () => {
        densityUp.assign(
          float(atomicAdd(densityAtomic.element(cellIndex.add(resXUint)), uint(0)))
        );
      });
      const densityBack = densityCenter.toVar();
      If(z.greaterThan(float(0)), () => {
        densityBack.assign(
          float(atomicAdd(densityAtomic.element(cellIndex.sub(resXYUint)), uint(0)))
        );
      });
      const densityFront = densityCenter.toVar();
      If(z.lessThan(resZf.sub(float(1))), () => {
        densityFront.assign(
          float(atomicAdd(densityAtomic.element(cellIndex.add(resXYUint)), uint(0)))
        );
      });

      const densitySpacing = this.volumeSizeUniform
        .div(vec3(resXf, resYf, resZf));
      const invDx = float(1.0).div(max(densitySpacing.x, float(1e-4)));
      const invDy = float(1.0).div(max(densitySpacing.y, float(1e-4)));
      const invDz = float(1.0).div(max(densitySpacing.z, float(1e-4)));
      const densityGrad = vec3(
        densityRight.sub(densityLeft).mul(invDx),
        densityUp.sub(densityDown).mul(invDy),
        densityFront.sub(densityBack).mul(invDz)
      ).mul(float(0.5));

      const coord = uvec3(x.toUint(), y.toUint(), z.toUint());
      textureStore(this.bubbleDensityGradientTexture, coord, vec4(densityGrad, float(1.0))).toWriteOnly();
    })().compute(cellCount);
  }

  private buildTextureClearPass(
    storageTexture: Storage3DTexture,
    resX: any,
    resY: any,
    resZ: any,
    totalCells: number
  ) {
    return Fn(() => {
      const resXY = resX.mul(resY);
      const x = instanceIndex.mod(resX);
      const y = instanceIndex.div(resX).mod(resY);
      const z = instanceIndex.div(resXY);
      const coord = uvec3(x.toUint(), y.toUint(), z.toUint());
      textureStore(storageTexture, coord, vec4(0.0, 0.0, 0.0, 0.0)).toWriteOnly();
    })().compute(totalCells);
  }

  private buildBubbleInjectionCompute(cellCount: number) {
    return Fn(() => {
      if (!this.bubblePositionsStorage || !this.bubbleVelocityStorage || !this.bubbleInjectionTexture) {
        return;
      }

      const resX = this.densityResXUniform;
      const resY = this.densityResYUniform;
      const resZ = this.densityResZUniform;
      const resXY = resX.mul(resY);
      const x = instanceIndex.mod(resX);
      const y = instanceIndex.div(resX).mod(resY);
      const z = instanceIndex.div(resXY);

      const uv = vec3(
        x.toFloat().add(float(0.5)).div(resX),
        y.toFloat().add(float(0.5)).div(resY),
        z.toFloat().add(float(0.5)).div(resZ)
      ).toVar();

      const worldPos = uv.sub(float(0.5)).mul(this.volumeSizeUniform).add(this.volumeCenterUniform);
      const injected = vec3(float(0.0), float(0.0), float(0.0)).toVar();
      const wakeRadiusBase = this.bubbleWakeAngleUniform;
      const wakeLength = this.bubbleWakeLengthUniform;
      const radiusRef = float(0.005);

      for (let i = 0; i < this.bubblePoolSize; i++) {
        const bubble = this.bubblePositionsStorage.element(float(i));
        const bubbleVel = this.bubbleVelocityStorage.element(float(i));
        const active = bubble.w;
        const toCell = worldPos.sub(bubble.xyz);
        const distSq = dot(toCell, toCell);
        const sizeRand = hash(float(i).mul(float(19.19)));
        const bubbleRadiusMm = mix(this.bubbleSizeMinUniform, this.bubbleSizeMaxUniform, sizeRand);
        const bubbleRadius = bubbleRadiusMm.mul(float(0.001));
        const baseRadius = max(this.fieldSplatRadiusUniform, bubbleRadius);
        const radiusSq = max(baseRadius.mul(baseRadius), float(1e-6));
        const influence = exp(distSq.negate().div(radiusSq)).mul(active);

        const velocityVec = vec3(bubbleVel.x, bubbleVel.y, bubbleVel.z);
        injected.addAssign(velocityVec.mul(influence));

        const speed = length(velocityVec);
        const speedSafe = max(speed, float(1e-4));
        const useUp = speed.lessThan(float(1e-4));
        const dir = vec3(
          select(useUp, float(0.0), velocityVec.x.div(speedSafe)),
          select(useUp, float(1.0), velocityVec.y.div(speedSafe)),
          select(useUp, float(0.0), velocityVec.z.div(speedSafe))
        );
        const along = dot(toCell, dir);
        const behind = along.lessThan(float(0.0));
        const d = along.negate();
        const radialVec = toCell.sub(dir.mul(along));
        const rho = length(radialVec);
        const radialDir = radialVec.div(max(rho, float(1e-4)));
        const sizeFactor = bubbleRadius.div(max(radiusRef, float(1e-6)));
        const axialSigma = max(wakeLength.mul(float(0.5)), bubbleRadius.mul(float(2.0)));
        const radialSigma = max(wakeRadiusBase, bubbleRadius.mul(float(1.5)));
        const axialWeight = exp(d.mul(d).negate().div(max(axialSigma.mul(axialSigma), float(1e-6))));
        const radialWeight = exp(rho.mul(rho).negate().div(max(radialSigma.mul(radialSigma), float(1e-6))));
        const wakeWeight = axialWeight.mul(radialWeight).mul(select(behind, float(1.0), float(0.0))).mul(active);
        const carry = float(0.6);
        const wakeDir = dir.mul(carry).add(radialDir.negate().mul(float(1.0).sub(carry)));
        const wakeStrength = this.bubbleWakeStrengthUniform.mul(sizeFactor);
        const wakeVel = wakeDir.mul(wakeStrength).mul(speed).mul(wakeWeight);
        injected.addAssign(wakeVel);
      }

      const coord = uvec3(x.toUint(), y.toUint(), z.toUint());
      textureStore(this.bubbleInjectionTexture, coord, vec4(injected, float(1.0))).toWriteOnly();
    })().compute(cellCount);
  }

  private clearFluidField(): void {
    if (!this.renderer) return;
    if (this.fieldClearVelocityPasses[0]) {
      this.renderer.compute(this.fieldClearVelocityPasses[0]);
    }
    if (this.fieldClearVelocityPasses[1]) {
      this.renderer.compute(this.fieldClearVelocityPasses[1]);
    }
    this.fieldVelocityReadIndex = 0;
    this.fieldVelocityIndexUniform.value = 0;
  }

  private stepField(): void {
    if (!this.renderer || !this.fieldSplatPasses[0]) {
      return;
    }
    const pass = this.fieldSplatPasses[this.fieldVelocityReadIndex];
    if (pass) {
      this.renderer.compute(pass);
    }
    this.fieldVelocityReadIndex = 1 - this.fieldVelocityReadIndex;
    this.fieldVelocityIndexUniform.value = this.fieldVelocityReadIndex;
  }

  private configureFieldTexture(texture: Storage3DTexture, name: string): void {
    texture.format = THREE.RGBAFormat;
    texture.type = THREE.FloatType;
    texture.generateMipmaps = false;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.wrapR = THREE.ClampToEdgeWrapping;
    texture.name = name;
  }

  private buildFieldClearPass(storageTexture: Storage3DTexture, totalCells: number) {
    return Fn(() => {
      const resX = this.fieldResXUniform;
      const resY = this.fieldResYUniform;
      const resXY = resX.mul(resY);
      const x = instanceIndex.mod(resX);
      const y = instanceIndex.div(resX).mod(resY);
      const z = instanceIndex.div(resXY);
      const coord = uvec3(x.toUint(), y.toUint(), z.toUint());
      textureStore(storageTexture, coord, vec4(0.0, 0.0, 0.0, 0.0)).toWriteOnly();
    })().compute(totalCells);
  }

  private buildFieldSplatPass(
    velocitySampler: ReturnType<typeof texture3D>,
    injectionSampler: ReturnType<typeof texture3D>,
    targetTexture: Storage3DTexture,
    totalCells: number
  ) {
    return Fn(() => {
      const resX = this.fieldResXUniform;
      const resY = this.fieldResYUniform;
      const resZ = this.fieldResZUniform;
      const resXY = resX.mul(resY);
      const x = instanceIndex.mod(resX);
      const y = instanceIndex.div(resX).mod(resY);
      const z = instanceIndex.div(resXY);

      const uv = vec3(
        x.toFloat().add(float(0.5)).div(resX),
        y.toFloat().add(float(0.5)).div(resY),
        z.toFloat().add(float(0.5)).div(resZ)
      );

      const prevVelocity = velocitySampler.sample(uv).xyz;
      const decay = clamp(
        float(1.0).sub(this.fieldDissipationUniform.mul(this.deltaTimeUniform)),
        float(0.0),
        float(1.0)
      );
      const injection = injectionSampler.sample(uv).xyz;
      const nextVelocity = prevVelocity
        .add(injection.mul(this.fieldSplatStrengthUniform).mul(this.deltaTimeUniform))
        .mul(decay);

      const coord = uvec3(x.toUint(), y.toUint(), z.toUint());
      textureStore(targetTexture, coord, vec4(nextVelocity, float(1.0))).toWriteOnly();
    })().compute(totalCells);
  }

  private buildFieldDivergencePass(
    velocitySampler: ReturnType<typeof texture3D>,
    targetTexture: Storage3DTexture,
    totalCells: number
  ) {
    return Fn(() => {
      const resX = this.fieldResXUniform;
      const resY = this.fieldResYUniform;
      const resZ = this.fieldResZUniform;
      const resXY = resX.mul(resY);
      const x = instanceIndex.mod(resX);
      const y = instanceIndex.div(resX).mod(resY);
      const z = instanceIndex.div(resXY);

      const one = float(1.0);
      const invRes = vec3(one, one, one).div(vec3(resX, resY, resZ));
      const halfTexel = invRes.mul(float(0.5));
      const maxUv = vec3(one, one, one).sub(halfTexel);

      const uv = vec3(
        x.toFloat().add(float(0.5)).div(resX),
        y.toFloat().add(float(0.5)).div(resY),
        z.toFloat().add(float(0.5)).div(resZ)
      );

      const leftUv = clamp(uv.sub(vec3(invRes.x, float(0.0), float(0.0))), halfTexel, maxUv);
      const rightUv = clamp(uv.add(vec3(invRes.x, float(0.0), float(0.0))), halfTexel, maxUv);
      const downUv = clamp(uv.sub(vec3(float(0.0), invRes.y, float(0.0))), halfTexel, maxUv);
      const upUv = clamp(uv.add(vec3(float(0.0), invRes.y, float(0.0))), halfTexel, maxUv);
      const backUv = clamp(uv.sub(vec3(float(0.0), float(0.0), invRes.z)), halfTexel, maxUv);
      const frontUv = clamp(uv.add(vec3(float(0.0), float(0.0), invRes.z)), halfTexel, maxUv);

      const velL = velocitySampler.sample(leftUv).xyz;
      const velR = velocitySampler.sample(rightUv).xyz;
      const velD = velocitySampler.sample(downUv).xyz;
      const velU = velocitySampler.sample(upUv).xyz;
      const velB = velocitySampler.sample(backUv).xyz;
      const velF = velocitySampler.sample(frontUv).xyz;

      const spacing = vec3(
        resX.div(this.volumeSizeUniform.x),
        resY.div(this.volumeSizeUniform.y),
        resZ.div(this.volumeSizeUniform.z)
      );

      const div = velR.x.sub(velL.x).mul(spacing.x)
        .add(velU.y.sub(velD.y).mul(spacing.y))
        .add(velF.z.sub(velB.z).mul(spacing.z))
        .mul(float(0.5));

      const coord = uvec3(x.toUint(), y.toUint(), z.toUint());
      textureStore(targetTexture, coord, vec4(div, div, div, one)).toWriteOnly();
    })().compute(totalCells);
  }

  private buildFieldPressurePass(
    pressureSampler: ReturnType<typeof texture3D>,
    divergenceSampler: ReturnType<typeof texture3D> | null,
    targetTexture: Storage3DTexture,
    totalCells: number
  ) {
    return Fn(() => {
      if (!divergenceSampler) {
        return;
      }

      const resX = this.fieldResXUniform;
      const resY = this.fieldResYUniform;
      const resZ = this.fieldResZUniform;
      const resXY = resX.mul(resY);
      const x = instanceIndex.mod(resX);
      const y = instanceIndex.div(resX).mod(resY);
      const z = instanceIndex.div(resXY);

      const one = float(1.0);
      const invRes = vec3(one, one, one).div(vec3(resX, resY, resZ));
      const halfTexel = invRes.mul(float(0.5));
      const maxUv = vec3(one, one, one).sub(halfTexel);

      const uv = vec3(
        x.toFloat().add(float(0.5)).div(resX),
        y.toFloat().add(float(0.5)).div(resY),
        z.toFloat().add(float(0.5)).div(resZ)
      );

      const leftUv = clamp(uv.sub(vec3(invRes.x, float(0.0), float(0.0))), halfTexel, maxUv);
      const rightUv = clamp(uv.add(vec3(invRes.x, float(0.0), float(0.0))), halfTexel, maxUv);
      const downUv = clamp(uv.sub(vec3(float(0.0), invRes.y, float(0.0))), halfTexel, maxUv);
      const upUv = clamp(uv.add(vec3(float(0.0), invRes.y, float(0.0))), halfTexel, maxUv);
      const backUv = clamp(uv.sub(vec3(float(0.0), float(0.0), invRes.z)), halfTexel, maxUv);
      const frontUv = clamp(uv.add(vec3(float(0.0), float(0.0), invRes.z)), halfTexel, maxUv);

      const pL = pressureSampler.sample(leftUv).x;
      const pR = pressureSampler.sample(rightUv).x;
      const pD = pressureSampler.sample(downUv).x;
      const pU = pressureSampler.sample(upUv).x;
      const pB = pressureSampler.sample(backUv).x;
      const pF = pressureSampler.sample(frontUv).x;
      const div = divergenceSampler.sample(uv).x;

      const cellSize = this.volumeSizeUniform.div(vec3(resX, resY, resZ));
      const cellSizeAvg = cellSize.x.add(cellSize.y).add(cellSize.z).div(float(3.0));
      const cellSizeSq = max(cellSizeAvg.mul(cellSizeAvg), float(1e-6));

      const pressure = pL.add(pR).add(pD).add(pU).add(pB).add(pF)
        .sub(div.mul(cellSizeSq))
        .div(float(6.0));

      const coord = uvec3(x.toUint(), y.toUint(), z.toUint());
      textureStore(targetTexture, coord, vec4(pressure, pressure, pressure, one)).toWriteOnly();
    })().compute(totalCells);
  }

  private buildFieldGradientPass(
    velocitySampler: ReturnType<typeof texture3D>,
    pressureSampler: ReturnType<typeof texture3D>,
    targetTexture: Storage3DTexture,
    totalCells: number
  ) {
    return Fn(() => {
      const resX = this.fieldResXUniform;
      const resY = this.fieldResYUniform;
      const resZ = this.fieldResZUniform;
      const resXY = resX.mul(resY);
      const x = instanceIndex.mod(resX);
      const y = instanceIndex.div(resX).mod(resY);
      const z = instanceIndex.div(resXY);

      const one = float(1.0);
      const invRes = vec3(one, one, one).div(vec3(resX, resY, resZ));
      const halfTexel = invRes.mul(float(0.5));
      const maxUv = vec3(one, one, one).sub(halfTexel);

      const uv = vec3(
        x.toFloat().add(float(0.5)).div(resX),
        y.toFloat().add(float(0.5)).div(resY),
        z.toFloat().add(float(0.5)).div(resZ)
      );

      const leftUv = clamp(uv.sub(vec3(invRes.x, float(0.0), float(0.0))), halfTexel, maxUv);
      const rightUv = clamp(uv.add(vec3(invRes.x, float(0.0), float(0.0))), halfTexel, maxUv);
      const downUv = clamp(uv.sub(vec3(float(0.0), invRes.y, float(0.0))), halfTexel, maxUv);
      const upUv = clamp(uv.add(vec3(float(0.0), invRes.y, float(0.0))), halfTexel, maxUv);
      const backUv = clamp(uv.sub(vec3(float(0.0), float(0.0), invRes.z)), halfTexel, maxUv);
      const frontUv = clamp(uv.add(vec3(float(0.0), float(0.0), invRes.z)), halfTexel, maxUv);

      const pL = pressureSampler.sample(leftUv).x;
      const pR = pressureSampler.sample(rightUv).x;
      const pD = pressureSampler.sample(downUv).x;
      const pU = pressureSampler.sample(upUv).x;
      const pB = pressureSampler.sample(backUv).x;
      const pF = pressureSampler.sample(frontUv).x;

      const spacing = vec3(
        resX.div(this.volumeSizeUniform.x),
        resY.div(this.volumeSizeUniform.y),
        resZ.div(this.volumeSizeUniform.z)
      );

      const gradient = vec3(
        pR.sub(pL).mul(spacing.x),
        pU.sub(pD).mul(spacing.y),
        pF.sub(pB).mul(spacing.z)
      ).mul(float(0.5));

      const velocity = velocitySampler.sample(uv).xyz.sub(gradient);

      const coord = uvec3(x.toUint(), y.toUint(), z.toUint());
      textureStore(targetTexture, coord, vec4(velocity, one)).toWriteOnly();
    })().compute(totalCells);
  }

  private attachKeyHandlers(): void {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'b' || event.key === 'B') {
        if (!this.bubbleKeyEmit) {
          this.bubbleResetPending = true;
        }
        this.bubbleKeyEmit = true;
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.key === 'b' || event.key === 'B') {
        this.bubbleKeyEmit = false;
      }
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    this.disposeCallbacks.push(() => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    });
  }

  private createVolumeWireframe(): void {
    if (!this.scene) return;
    const geometry = new THREE.BoxGeometry(
      this.parameters.volumeWidth as number,
      this.parameters.volumeHeight as number,
      this.parameters.volumeDepth as number
    );
    const edges = new THREE.EdgesGeometry(geometry);
    const material = new THREE.LineBasicMaterial({
      color: 0x113344,
      transparent: true,
      opacity: 0.35
    });
    this.volumeWireframe = new THREE.LineSegments(edges, material);
    this.volumeWireframe.visible = this.parameters.showVolume as boolean;
    this.scene.add(this.volumeWireframe);
  }

  private updateVolumeWireframe(): void {
    if (!this.scene) return;
    if (!this.volumeWireframe) {
      this.createVolumeWireframe();
      return;
    }
    const geometry = new THREE.BoxGeometry(
      this.parameters.volumeWidth as number,
      this.parameters.volumeHeight as number,
      this.parameters.volumeDepth as number
    );
    const edges = new THREE.EdgesGeometry(geometry);
    this.volumeWireframe.geometry.dispose();
    this.volumeWireframe.geometry = edges;
    this.volumeWireframe.visible = this.parameters.showVolume as boolean;
  }

  private setupPostProcessing(): void {
    if (!this.renderer || !this.scene || !this.camera) return;
    if (this.postProcessing) {
      this.postProcessing.dispose();
      this.postProcessing = null;
    }

    this.postProcessing = new PostProcessing(this.renderer);
    const scenePass = pass(this.scene, this.camera);
    scenePass.setMRT(mrt({ output, emissive }));

    const sceneColor = scenePass.getTextureNode('output');
    const emissivePass = scenePass.getTextureNode('emissive');

    this.bloomNode = bloom(
      emissivePass,
      this.parameters.bloomStrength as number,
      this.parameters.bloomRadius as number,
      this.parameters.bloomThreshold as number
    );

    this.postProcessing.outputNode = sceneColor.add(this.bloomNode);
  }

  private updateBloom(): void {
    if (!this.bloomNode) {
      this.setupPostProcessing();
      return;
    }
    if (this.bloomNode.strength) this.bloomNode.strength.value = this.parameters.bloomStrength;
    if (this.bloomNode.radius) this.bloomNode.radius.value = this.parameters.bloomRadius;
    if (this.bloomNode.threshold) this.bloomNode.threshold.value = this.parameters.bloomThreshold;
  }
}
