import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  Fn,
  Loop,
  clamp,
  color,
  dot,
  exp,
  float,
  int,
  getViewPosition,
  interleavedGradientNoise,
  length,
  max,
  min,
  mix,
  normalize,
  shadow,
  pass,
  pow,
  screenCoordinate,
  uv,
  select,
  sqrt,
  uniform,
  vec3,
  vec4
} from 'three/tsl';
import { PLANET_SCENE_DEFAULTS, type PlanetSceneParameters } from './PlanetSceneParameters';

const MAX_ATMOSPHERES = 10;
const ACTIVE_ATMOSPHERES = 2;
const EXTRA_ATMOSPHERES = Math.max(MAX_ATMOSPHERES - ACTIVE_ATMOSPHERES, 0);

const createEmptyAtmosphere = () => ({
  center: uniform(new THREE.Vector3()),
  fogDensity: uniform(0),
  atmosphereAltitude: uniform(0),
  falloffPower: uniform(1),
  multiScatterBoost: uniform(0),
  phaseG: uniform(0),
  rayleighStrength: uniform(0),
  mieStrength: uniform(0),
  rayleighColor: uniform(color('#000000')),
  mieColor: uniform(color('#000000'))
});

export class PlanetScene {
  private readonly planetRadiusValue = 1.0;
  private canvas: HTMLCanvasElement | null = null;
  private renderer: THREE.WebGPURenderer | null = null;
  private postProcessing: THREE.PostProcessing | null = null;
  private scene: THREE.Scene | null = null;
  private camera: THREE.PerspectiveCamera | null = null;
  private controls: OrbitControls | null = null;
  private parameters: PlanetSceneParameters = { ...PLANET_SCENE_DEFAULTS };
  private ambientStrengthUniform = uniform(PLANET_SCENE_DEFAULTS.ambientStrength);
  private sunColorUniform = uniform(color(PLANET_SCENE_DEFAULTS.sunColor));
  private sphere1FogDensityUniform = uniform(PLANET_SCENE_DEFAULTS.sphere1FogDensity);
  private sphere1AtmosphereAltitudeUniform = uniform(PLANET_SCENE_DEFAULTS.sphere1AtmosphereAltitude);
  private sphere1FalloffPowerUniform = uniform(PLANET_SCENE_DEFAULTS.sphere1FalloffPower);
  private sphere1MultiScatterBoostUniform = uniform(PLANET_SCENE_DEFAULTS.sphere1MultiScatterBoost);
  private sphere1PhaseGUniform = uniform(PLANET_SCENE_DEFAULTS.sphere1PhaseG);
  private sphere1RayleighStrengthUniform = uniform(PLANET_SCENE_DEFAULTS.sphere1RayleighStrength);
  private sphere1MieStrengthUniform = uniform(PLANET_SCENE_DEFAULTS.sphere1MieStrength);
  private sphere1RayleighColorUniform = uniform(color(PLANET_SCENE_DEFAULTS.sphere1RayleighColor));
  private sphere1MieColorUniform = uniform(color(PLANET_SCENE_DEFAULTS.sphere1MieColor));
  private sphere2FogDensityUniform = uniform(PLANET_SCENE_DEFAULTS.sphere2FogDensity);
  private sphere2AtmosphereAltitudeUniform = uniform(PLANET_SCENE_DEFAULTS.sphere2AtmosphereAltitude);
  private sphere2FalloffPowerUniform = uniform(PLANET_SCENE_DEFAULTS.sphere2FalloffPower);
  private sphere2MultiScatterBoostUniform = uniform(PLANET_SCENE_DEFAULTS.sphere2MultiScatterBoost);
  private sphere2PhaseGUniform = uniform(PLANET_SCENE_DEFAULTS.sphere2PhaseG);
  private sphere2RayleighStrengthUniform = uniform(PLANET_SCENE_DEFAULTS.sphere2RayleighStrength);
  private sphere2MieStrengthUniform = uniform(PLANET_SCENE_DEFAULTS.sphere2MieStrength);
  private sphere2RayleighColorUniform = uniform(color(PLANET_SCENE_DEFAULTS.sphere2RayleighColor));
  private sphere2MieColorUniform = uniform(color(PLANET_SCENE_DEFAULTS.sphere2MieColor));
  private cameraPositionUniform = uniform(new THREE.Vector3());
  private cameraWorldMatrixUniform = uniform(new THREE.Matrix4());
  private cameraProjectionMatrixInverseUniform = uniform(new THREE.Matrix4());
  private primaryCenterUniform = uniform(new THREE.Vector3());
  private secondaryCenterUniform = uniform(new THREE.Vector3());
  private sunDirectionUniform = uniform(new THREE.Vector3(0, 0, 1));
  private sunIntensityUniform = uniform(1.0);
  private extraAtmospheres = Array.from({ length: EXTRA_ATMOSPHERES }, () => createEmptyAtmosphere());
  private sunDirectionScratch = new THREE.Vector3();
  private sunOffsetScratch = new THREE.Vector3();
  private readonly sunOffsetDistance = this.planetRadiusValue * 2.0 * 10.0;
  private keyLight: THREE.DirectionalLight | null = null;
  private shadowCasterGeometry: THREE.SphereGeometry | null = null;
  private shadowCasterMaterial: THREE.MeshBasicMaterial | null = null;
  private primaryShadowCaster: THREE.Mesh | null = null;
  private secondaryShadowCaster: THREE.Mesh | null = null;
  private width = 1;
  private height = 1;
  private handleControlChange: (() => void) | null = null;
  private ready = false;
  private pendingResize: { width: number; height: number } | null = null;

  async init(canvas: HTMLCanvasElement): Promise<void> {
    this.canvas = canvas;
    this.width = canvas.clientWidth || canvas.width || window.innerWidth || 1;
    this.height = canvas.clientHeight || canvas.height || window.innerHeight || 1;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000000);

    this.camera = new THREE.PerspectiveCamera(50, this.width / this.height, 0.1, 100);
    this.camera.position.set(0, 0.6, 3);

    this.renderer = new THREE.WebGPURenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(this.width, this.height, false);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.shadowMap.autoUpdate = true;
    await this.renderer.init();
    this.ready = true;

    const ambient = new THREE.AmbientLight(0xffffff, 0.35);
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
    keyLight.position.set(3, 0, 5);
    keyLight.target.position.set(0, 0, 0);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(2048, 2048);
    keyLight.shadow.bias = -0.0005;
    keyLight.shadow.normalBias = 0.0;
    keyLight.shadow.camera.near = 0.1;
    keyLight.shadow.camera.far = 100;
    const shadowRange = this.sunOffsetDistance + this.planetRadiusValue * 4.0;
    keyLight.shadow.camera.left = -shadowRange;
    keyLight.shadow.camera.right = shadowRange;
    keyLight.shadow.camera.top = shadowRange;
    keyLight.shadow.camera.bottom = -shadowRange;
    this.keyLight = keyLight;
    const fillLight = new THREE.DirectionalLight(0x4f83ff, 0.35);
    fillLight.position.set(-4, -2, -3);

    this.scene.add(ambient, keyLight, keyLight.target, fillLight);

    const planetRadiusValue = this.planetRadiusValue;
    this.shadowCasterGeometry = new THREE.SphereGeometry(planetRadiusValue, 64, 64);
    this.shadowCasterMaterial = new THREE.MeshBasicMaterial({ color: 0x000000 });
    this.shadowCasterMaterial.colorWrite = false;
    this.primaryShadowCaster = new THREE.Mesh(this.shadowCasterGeometry, this.shadowCasterMaterial);
    this.primaryShadowCaster.castShadow = true;
    this.primaryShadowCaster.receiveShadow = false;
    this.scene.add(this.primaryShadowCaster);
    this.secondaryShadowCaster = new THREE.Mesh(this.shadowCasterGeometry, this.shadowCasterMaterial);
    this.secondaryShadowCaster.castShadow = true;
    this.secondaryShadowCaster.receiveShadow = false;
    this.scene.add(this.secondaryShadowCaster);
    this.ambientStrengthUniform.value = this.parameters.ambientStrength;
    this.sphere1FogDensityUniform.value = this.parameters.sphere1FogDensity;
    this.sphere1AtmosphereAltitudeUniform.value = this.parameters.sphere1AtmosphereAltitude;
    this.sphere1FalloffPowerUniform.value = this.parameters.sphere1FalloffPower;
    this.sphere1MultiScatterBoostUniform.value = this.parameters.sphere1MultiScatterBoost;
    this.sphere1PhaseGUniform.value = this.parameters.sphere1PhaseG;
    this.sphere1RayleighStrengthUniform.value = this.parameters.sphere1RayleighStrength;
    this.sphere1MieStrengthUniform.value = this.parameters.sphere1MieStrength;
    this.sphere1RayleighColorUniform.value.set(this.parameters.sphere1RayleighColor);
    this.sphere1MieColorUniform.value.set(this.parameters.sphere1MieColor);
    this.sphere2FogDensityUniform.value = this.parameters.sphere2FogDensity;
    this.sphere2AtmosphereAltitudeUniform.value = this.parameters.sphere2AtmosphereAltitude;
    this.sphere2FalloffPowerUniform.value = this.parameters.sphere2FalloffPower;
    this.sphere2MultiScatterBoostUniform.value = this.parameters.sphere2MultiScatterBoost;
    this.sphere2PhaseGUniform.value = this.parameters.sphere2PhaseG;
    this.sphere2RayleighStrengthUniform.value = this.parameters.sphere2RayleighStrength;
    this.sphere2MieStrengthUniform.value = this.parameters.sphere2MieStrength;
    this.sphere2RayleighColorUniform.value.set(this.parameters.sphere2RayleighColor);
    this.sphere2MieColorUniform.value.set(this.parameters.sphere2MieColor);
    this.sunColorUniform.value.set(this.parameters.sunColor);
    this.primaryCenterUniform.value.set(0, 0, 0);
    this.updateSunUniforms();
    this.updateCameraUniforms();

    const planetRadius = uniform(planetRadiusValue);
    const sunDirection = this.sunDirectionUniform;
    const sunIntensity = this.sunIntensityUniform;
    const ambientStrength = this.ambientStrengthUniform;
    const sunColor = this.sunColorUniform;
    const cameraPositionNode = this.cameraPositionUniform;
    const cameraWorldMatrixNode = this.cameraWorldMatrixUniform;
    const cameraProjectionMatrixInverseNode = this.cameraProjectionMatrixInverseUniform;
    const primaryCenter = this.primaryCenterUniform;
    const secondaryCenter = this.secondaryCenterUniform;
    const sphere1 = {
      center: primaryCenter,
      fogDensity: this.sphere1FogDensityUniform,
      atmosphereAltitude: this.sphere1AtmosphereAltitudeUniform,
      falloffPower: this.sphere1FalloffPowerUniform,
      multiScatterBoost: this.sphere1MultiScatterBoostUniform,
      phaseG: this.sphere1PhaseGUniform,
      rayleighStrength: this.sphere1RayleighStrengthUniform,
      mieStrength: this.sphere1MieStrengthUniform,
      rayleighColor: this.sphere1RayleighColorUniform,
      mieColor: this.sphere1MieColorUniform
    };
    const sphere2 = {
      center: secondaryCenter,
      fogDensity: this.sphere2FogDensityUniform,
      atmosphereAltitude: this.sphere2AtmosphereAltitudeUniform,
      falloffPower: this.sphere2FalloffPowerUniform,
      multiScatterBoost: this.sphere2MultiScatterBoostUniform,
      phaseG: this.sphere2PhaseGUniform,
      rayleighStrength: this.sphere2RayleighStrengthUniform,
      mieStrength: this.sphere2MieStrengthUniform,
      rayleighColor: this.sphere2RayleighColorUniform,
      mieColor: this.sphere2MieColorUniform
    };
    const atmosphereConfigs = [sphere1, sphere2, ...this.extraAtmospheres].slice(
      0,
      ACTIVE_ATMOSPHERES
    );
    const shadowNode = shadow(keyLight);

    const scenePass = pass(this.scene, this.camera);
    const sceneColor = scenePass.getTextureNode('output');
    const atmosphereNode = Fn(() => {
      const maxRayDistance = float(100000.0);
      const uvNode = uv();
      const rayOrigin = cameraPositionNode;
      const viewPos = getViewPosition(uvNode, float(1.0), cameraProjectionMatrixInverseNode);
      const rayDirView = normalize(viewPos);
      const worldPos = cameraWorldMatrixNode.mul(vec4(viewPos, float(1.0))).xyz;
      const rayDir = normalize(worldPos.sub(rayOrigin));
      const sunDir = normalize(sunDirection);
      const jitter = clamp(interleavedGradientNoise(screenCoordinate), float(0.0), float(1.0));
      const sceneStop = maxRayDistance;

      const buildAtmosphere = (settings: typeof sphere1) => {
        const {
          center,
          fogDensity,
          atmosphereAltitude,
          falloffPower,
          multiScatterBoost,
          phaseG,
          rayleighStrength,
          mieStrength,
          rayleighColor,
          mieColor
        } = settings;
        const rayOriginLocal = rayOrigin.sub(center);
        const outerRadius = planetRadius.add(atmosphereAltitude);
        const oc = rayOriginLocal;
        const b = dot(oc, rayDir);
        const ocLenSq = dot(oc, oc);
        const outerC = ocLenSq.sub(outerRadius.mul(outerRadius));
        const outerH = b.mul(b).sub(outerC);
        const sqrtOuter = sqrt(max(outerH, float(0.0)));

        const outerEntry = b.negate().sub(sqrtOuter);
        const outerExit = b.negate().add(sqrtOuter);
        const hitOuter = outerH.greaterThan(float(0.0)).and(outerExit.greaterThan(float(0.0)));
        const tEntry = max(outerEntry, float(0.0));
        const tEntrySort = select(hitOuter, tEntry, maxRayDistance);

        const innerC = ocLenSq.sub(planetRadius.mul(planetRadius));
        const innerH = b.mul(b).sub(innerC);
        const sqrtInner = sqrt(max(innerH, float(0.0)));
        const innerEntry = b.negate().sub(sqrtInner);
        const innerExit = b.negate().add(sqrtInner);
        const hitsInner = innerH.greaterThan(float(0.0)).and(innerExit.greaterThan(float(0.0)));
        const innerHitT = select(hitsInner, max(innerEntry, float(0.0)), maxRayDistance);

        const thicknessToPlanet = max(innerEntry.sub(tEntry), float(0.0));
        const thicknessNoPlanet = max(outerExit.sub(tEntry), float(0.0));
        const thickness = select(hitOuter, select(hitsInner, thicknessToPlanet, thicknessNoPlanet), float(0.0));

        const safeAltitude = max(atmosphereAltitude, float(0.0001));
        const marchSteps = int(12);
        const marchStepsF = float(12.0);
        const stepSize = thickness.div(marchStepsF);
        const rayStart = rayOriginLocal.add(rayDir.mul(tEntry));
        const rayEnd = rayStart.add(rayDir.mul(thickness));
        const rayMid = rayStart.add(rayDir.mul(thickness.mul(float(0.5))));

        const densityAt = (samplePos: typeof rayOriginLocal) => {
          const altitude = length(samplePos).sub(planetRadius);
          const heightT = clamp(altitude.div(safeAltitude), float(0.0), float(1.0));
          return pow(float(1.0).sub(heightT), falloffPower);
        };

        const sunTransmittanceAt = (samplePos: typeof rayOriginLocal) => {
          const bSun = dot(samplePos, sunDir);
          const sampleLenSq = dot(samplePos, samplePos);

          const outerC = sampleLenSq.sub(outerRadius.mul(outerRadius));
          const outerH = bSun.mul(bSun).sub(outerC);
          const sqrtOuterSun = sqrt(max(outerH, float(0.0)));
          const outerExit = bSun.negate().add(sqrtOuterSun);
          const sunRayLength = max(outerExit, float(0.0));

          const innerC = sampleLenSq.sub(planetRadius.mul(planetRadius));
          const innerH = bSun.mul(bSun).sub(innerC);
          const sqrtInnerSun = sqrt(max(innerH, float(0.0)));
          const innerEntry = bSun.negate().sub(sqrtInnerSun);
          const occluded = innerH.greaterThan(float(0.0)).and(innerEntry.greaterThan(float(0.0)));

          const sunEnd = samplePos.add(sunDir.mul(sunRayLength));
          const sunMid = samplePos.add(sunDir.mul(sunRayLength.mul(float(0.5))));
          const sunDensityStart = densityAt(samplePos);
          const sunDensityMid = densityAt(sunMid);
          const sunDensityEnd = densityAt(sunEnd);
          const sunIntegral = sunDensityStart
            .add(sunDensityEnd)
            .add(sunDensityMid.mul(float(4.0)))
            .mul(sunRayLength)
            .div(float(6.0));
          const sunOpticalDepth = sunIntegral.mul(fogDensity);
          const sunTransmittance = exp(sunOpticalDepth.negate());
          const samplePosWorld = samplePos.add(center);
          const shadowSample = shadowNode.context({ shadowPositionWorld: samplePosWorld });
          const shadowFactor = clamp(shadowSample.x, float(0.0), float(1.0));
          const litTransmittance = sunTransmittance.mul(shadowFactor);
          return select(occluded, float(0.0), litTransmittance);
        };

        const densityStart = densityAt(rayStart);
        const densityMid = densityAt(rayMid);
        const densityEnd = densityAt(rayEnd);
        const smoothDensityIntegral = densityStart
          .add(densityEnd)
          .add(densityMid.mul(float(4.0)))
          .mul(thickness)
          .div(float(6.0));

        const densitySum = float(0.0).toVar();
        const litDensitySum = float(0.0).toVar();

        Loop(marchSteps, ({ i }) => {
          const t0 = stepSize.mul(float(i));
          const t1 = min(t0.add(stepSize), thickness);
          const segmentLength = max(t1.sub(t0), float(0.0));
          const sampleT = t0.add(segmentLength.mul(jitter));
          const samplePos = rayStart.add(rayDir.mul(sampleT));
          const density = densityAt(samplePos);
          const densityStep = density.mul(segmentLength);
          const sunTransmittance = sunTransmittanceAt(samplePos);

          densitySum.addAssign(densityStep);
          litDensitySum.addAssign(densityStep.mul(sunTransmittance));
        });

        const opticalDepth = smoothDensityIntegral.mul(fogDensity);
        const fogFactor = select(
          hitOuter,
          float(1.0).sub(exp(opticalDepth.negate())),
          float(0.0)
        );

        const densityNorm = max(densitySum, float(0.0001));
        const baseLight = clamp(litDensitySum.div(densityNorm), float(0.0), float(1.0));
        const mu = clamp(dot(rayDir.negate(), sunDir), float(-1.0), float(1.0));
        const g2 = phaseG.mul(phaseG);
        const denom = float(1.0).add(g2).sub(float(2.0).mul(phaseG).mul(mu));
        const miePhase = float(1.0).sub(g2).div(denom.mul(sqrt(denom)));
        const rayleighPhase = float(0.75).mul(float(1.0).add(mu.mul(mu)));
        const scatterColor = rayleighColor.mul(rayleighStrength).mul(rayleighPhase)
          .add(mieColor.mul(mieStrength).mul(miePhase))
          .mul(float(0.5));
        const directScatter = scatterColor.mul(baseLight).mul(sunColor).mul(sunIntensity);
        const multiScatter = scatterColor
          .mul(multiScatterBoost)
          .mul(float(1.0).sub(baseLight))
          .mul(sunColor)
          .mul(sunIntensity);
        const ambientScatter = scatterColor.mul(ambientStrength);
        const atmosphereLit = ambientScatter.add(directScatter).add(multiScatter);

        return { fogFactor, atmosphereLit, tEntry: tEntrySort, innerHitT };
      };

      const atmosphereResults = atmosphereConfigs.map((settings) => buildAtmosphere(settings));

      let closestInnerT = maxRayDistance;
      for (const atmosphere of atmosphereResults) {
        closestInnerT = min(closestInnerT, atmosphere.innerHitT);
      }

      const rayStop = min(sceneStop, closestInnerT);
      const sceneSample = sceneColor.sample(uvNode);
      const baseColor = select(closestInnerT.lessThan(sceneStop), vec3(0.0), sceneSample.rgb);

      const entries = atmosphereResults.map((atmosphere) => atmosphere.tEntry);
      const fogs = atmosphereResults.map((atmosphere) =>
        select(atmosphere.tEntry.lessThan(rayStop), atmosphere.fogFactor, float(0.0))
      );
      const colors = atmosphereResults.map((atmosphere) => atmosphere.atmosphereLit);

      for (let i = 0; i < entries.length - 1; i += 1) {
        for (let j = 0; j < entries.length - 1 - i; j += 1) {
          const swap = entries[j].greaterThan(entries[j + 1]);
          const entryA = entries[j];
          const entryB = entries[j + 1];
          entries[j] = select(swap, entryB, entryA);
          entries[j + 1] = select(swap, entryA, entryB);

          const fogA = fogs[j];
          const fogB = fogs[j + 1];
          fogs[j] = select(swap, fogB, fogA);
          fogs[j + 1] = select(swap, fogA, fogB);

          const colorA = colors[j];
          const colorB = colors[j + 1];
          colors[j] = select(swap, colorB, colorA);
          colors[j + 1] = select(swap, colorA, colorB);
        }
      }

      let transmittance = float(1.0);
      let accumulated = vec3(0.0);
      for (let i = 0; i < entries.length; i += 1) {
        const fog = fogs[i];
        accumulated = accumulated.add(colors[i].mul(fog).mul(transmittance));
        transmittance = transmittance.mul(float(1.0).sub(fog));
      }

      const finalColor = accumulated.add(baseColor.mul(transmittance));
      return vec4(finalColor, float(1.0));
    })();

    this.postProcessing = new THREE.PostProcessing(this.renderer);
    this.postProcessing.outputNode = atmosphereNode;

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = false;
    this.controls.target.set(0, 0, 0);
    this.controls.update();

    this.handleControlChange = () => this.render();
    this.controls.addEventListener('change', this.handleControlChange);

    if (this.pendingResize) {
      this.applyResize(this.pendingResize.width, this.pendingResize.height);
      this.pendingResize = null;
    } else {
      this.applyResize(this.width, this.height);
    }
  }

  update(): void {
    // Intentionally empty: no animation.
  }

  updateParameters(params: Partial<PlanetSceneParameters>): void {
    let needsRender = false;
    const setNumber = (key: keyof PlanetSceneParameters, uniform: { value: number }) => {
      const value = params[key];
      if (typeof value !== 'number') return;
      (this.parameters as Record<string, number>)[key] = value;
      uniform.value = value;
      needsRender = true;
    };
    const setColor = (key: keyof PlanetSceneParameters, uniform: { value: THREE.Color }) => {
      const value = params[key];
      if (typeof value !== 'string') return;
      (this.parameters as Record<string, string>)[key] = value;
      try {
        uniform.value.set(value);
      } catch (error) {
        void error;
      }
      needsRender = true;
    };

    setNumber('ambientStrength', this.ambientStrengthUniform);
    setColor('sunColor', this.sunColorUniform);
    setNumber('sphere1FogDensity', this.sphere1FogDensityUniform);
    setNumber('sphere1AtmosphereAltitude', this.sphere1AtmosphereAltitudeUniform);
    setNumber('sphere1FalloffPower', this.sphere1FalloffPowerUniform);
    setNumber('sphere1MultiScatterBoost', this.sphere1MultiScatterBoostUniform);
    setNumber('sphere1RayleighStrength', this.sphere1RayleighStrengthUniform);
    setNumber('sphere1MieStrength', this.sphere1MieStrengthUniform);
    setNumber('sphere1PhaseG', this.sphere1PhaseGUniform);
    setColor('sphere1RayleighColor', this.sphere1RayleighColorUniform);
    setColor('sphere1MieColor', this.sphere1MieColorUniform);
    setNumber('sphere2FogDensity', this.sphere2FogDensityUniform);
    setNumber('sphere2AtmosphereAltitude', this.sphere2AtmosphereAltitudeUniform);
    setNumber('sphere2FalloffPower', this.sphere2FalloffPowerUniform);
    setNumber('sphere2MultiScatterBoost', this.sphere2MultiScatterBoostUniform);
    setNumber('sphere2RayleighStrength', this.sphere2RayleighStrengthUniform);
    setNumber('sphere2MieStrength', this.sphere2MieStrengthUniform);
    setNumber('sphere2PhaseG', this.sphere2PhaseGUniform);
    setColor('sphere2RayleighColor', this.sphere2RayleighColorUniform);
    setColor('sphere2MieColor', this.sphere2MieColorUniform);

    if (needsRender) {
      this.render();
    }
  }

  render(): void {
    if (!this.ready || !this.renderer || !this.scene || !this.camera) return;
    this.updateSunUniforms();
    this.updateCameraUniforms();
    if (this.postProcessing) {
      this.postProcessing.render();
    } else {
      this.renderer.render(this.scene, this.camera);
    }
  }

  onResize(width: number, height: number): void {
    if (!this.ready) {
      this.pendingResize = { width, height };
      return;
    }

    this.applyResize(width, height);
  }

  cleanup(): void {
    if (this.controls && this.handleControlChange) {
      this.controls.removeEventListener('change', this.handleControlChange);
    }

    this.controls?.dispose();

    this.postProcessing?.dispose();
    this.renderer?.dispose();
    this.shadowCasterGeometry?.dispose();
    this.shadowCasterMaterial?.dispose();

    this.ready = false;
    this.pendingResize = null;
    this.canvas = null;
    this.renderer = null;
    this.postProcessing = null;
    this.scene = null;
    this.camera = null;
    this.controls = null;
    this.keyLight = null;
    this.shadowCasterGeometry = null;
    this.shadowCasterMaterial = null;
    this.primaryShadowCaster = null;
    this.secondaryShadowCaster = null;
    this.handleControlChange = null;
  }

  private applyResize(width: number, height: number): void {
    if (!this.renderer || !this.camera) return;
    this.width = Math.max(width, 1);
    this.height = Math.max(height, 1);
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.width, this.height, false);
    this.render();
  }

  private updateSunUniforms(): void {
    if (!this.keyLight) return;
    this.sunDirectionScratch.copy(this.keyLight.position).sub(this.keyLight.target.position).normalize();
    this.sunDirectionUniform.value.copy(this.sunDirectionScratch);
    this.sunIntensityUniform.value = this.keyLight.intensity;
    this.sunOffsetScratch.set(this.sunDirectionScratch.x, 0, this.sunDirectionScratch.z);
    if (this.sunOffsetScratch.lengthSq() < 1e-6) {
      this.sunOffsetScratch.set(1, 0, 0);
    }
    this.sunOffsetScratch.normalize().multiplyScalar(this.sunOffsetDistance);
    this.secondaryCenterUniform.value.copy(this.sunOffsetScratch);
    if (this.secondaryShadowCaster) {
      this.secondaryShadowCaster.position.copy(this.sunOffsetScratch);
    }
  }

  private updateCameraUniforms(): void {
    if (!this.camera) return;
    this.camera.updateMatrixWorld();
    this.cameraPositionUniform.value.setFromMatrixPosition(this.camera.matrixWorld);
    this.cameraWorldMatrixUniform.value.copy(this.camera.matrixWorld);
    this.cameraProjectionMatrixInverseUniform.value.copy(this.camera.projectionMatrixInverse);
  }
}
