import * as THREE from 'three';
import { Storage3DTexture, WebGPURenderer } from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  Loop,
  abs,
  clamp,
  exp,
  float,
  instanceIndex,
  length,
  max,
  min,
  mix,
  normalize,
  select,
  texture,
  texture3D,
  textureStore,
  uvec2,
  uvec3,
  uniform,
  vec2,
  vec3,
  vec4
} from 'three/tsl';
import { FROXEL_PIXEL_SIZE, FROXEL_SLICE_COUNT, MAX_FROXEL_SLICES } from '../oneill-cylinder/constants';

export const MAX_FOG_VOLUMES = 4;

export type FogVolumeConfig = {
  type: 'sphere';
  center: THREE.Vector3;
  radius: number;
  density: number;
  falloff?: number;
  enabled?: boolean;
};

type BoxFroxelPipelineOptions = {
  froxelResolutionUniform: ReturnType<typeof uniform>;
  froxelStepUniform: ReturnType<typeof uniform>;
  cameraProjectionInverseUniform: ReturnType<typeof uniform>;
  cameraWorldMatrixUniform: ReturnType<typeof uniform>;
  cameraNearUniform: ReturnType<typeof uniform>;
  cameraFarUniform: ReturnType<typeof uniform>;
  fogDensityUniform: ReturnType<typeof uniform>;
  cameraIsOrthographicUniform: ReturnType<typeof uniform>;
  shadowDepthTextureNode: ReturnType<typeof texture>;
  shadowReadyUniform: ReturnType<typeof uniform>;
  shadowBiasUniform: ReturnType<typeof uniform>;
  shadowMapSizeUniform: ReturnType<typeof uniform>;
  shadowMatrixUniform: ReturnType<typeof uniform>;
  shadowWebGPUUniform: ReturnType<typeof uniform>;
};

export class BoxFroxelPipeline {
  private readonly froxelResolutionUniform: ReturnType<typeof uniform>;
  private readonly froxelStepUniform: ReturnType<typeof uniform>;
  private readonly cameraProjectionInverseUniform: ReturnType<typeof uniform>;
  private readonly cameraWorldMatrixUniform: ReturnType<typeof uniform>;
  private readonly cameraNearUniform: ReturnType<typeof uniform>;
  private readonly cameraFarUniform: ReturnType<typeof uniform>;
  private readonly fogDensityUniform: ReturnType<typeof uniform>;
  private readonly cameraIsOrthographicUniform: ReturnType<typeof uniform>;
  private readonly shadowDepthTextureNode: ReturnType<typeof texture>;
  private readonly shadowReadyUniform: ReturnType<typeof uniform>;
  private readonly shadowBiasUniform: ReturnType<typeof uniform>;
  private readonly shadowMapSizeUniform: ReturnType<typeof uniform>;
  private readonly shadowMatrixUniform: ReturnType<typeof uniform>;
  private readonly shadowWebGPUUniform: ReturnType<typeof uniform>;
  private readonly fogVolumeEnabledUniforms: Array<ReturnType<typeof uniform>>;
  private readonly fogVolumeCenterUniforms: Array<ReturnType<typeof uniform>>;
  private readonly fogVolumeParamsUniforms: Array<ReturnType<typeof uniform>>;

  private viewTexture: Storage3DTexture | null = null;
  private viewSampler: ReturnType<typeof texture3D> | null = null;
  private viewCompute: any = null;

  constructor(options: BoxFroxelPipelineOptions) {
    this.froxelResolutionUniform = options.froxelResolutionUniform;
    this.froxelStepUniform = options.froxelStepUniform;
    this.cameraProjectionInverseUniform = options.cameraProjectionInverseUniform;
    this.cameraWorldMatrixUniform = options.cameraWorldMatrixUniform;
    this.cameraNearUniform = options.cameraNearUniform;
    this.cameraFarUniform = options.cameraFarUniform;
    this.fogDensityUniform = options.fogDensityUniform;
    this.cameraIsOrthographicUniform = options.cameraIsOrthographicUniform;
    this.shadowDepthTextureNode = options.shadowDepthTextureNode;
    this.shadowReadyUniform = options.shadowReadyUniform;
    this.shadowBiasUniform = options.shadowBiasUniform;
    this.shadowMapSizeUniform = options.shadowMapSizeUniform;
    this.shadowMatrixUniform = options.shadowMatrixUniform;
    this.shadowWebGPUUniform = options.shadowWebGPUUniform;
    this.fogVolumeEnabledUniforms = Array.from(
      { length: MAX_FOG_VOLUMES },
      () => uniform(0, 'float')
    );
    this.fogVolumeCenterUniforms = Array.from(
      { length: MAX_FOG_VOLUMES },
      () => uniform(new THREE.Vector3(), 'vec3')
    );
    this.fogVolumeParamsUniforms = Array.from(
      { length: MAX_FOG_VOLUMES },
      () => uniform(new THREE.Vector4(), 'vec4')
    );
  }

  init(width: number, height: number, camera: THREE.PerspectiveCamera | THREE.OrthographicCamera): void {
    this.dispose();

    const resX = Math.max(1, Math.floor(width / FROXEL_PIXEL_SIZE));
    const resY = Math.max(1, Math.floor(height / FROXEL_PIXEL_SIZE));
    const resZ = FROXEL_SLICE_COUNT;

    this.froxelResolutionUniform.value.set(resX, resY, resZ);
    const depthRange = camera.far - camera.near;
    this.froxelStepUniform.value = depthRange / resZ;

    const viewTexture = new Storage3DTexture(resX, resY, resZ);
    viewTexture.format = THREE.RGBAFormat;
    viewTexture.type = THREE.FloatType;
    viewTexture.generateMipmaps = false;
    viewTexture.minFilter = THREE.LinearFilter;
    viewTexture.magFilter = THREE.LinearFilter;
    viewTexture.wrapS = THREE.ClampToEdgeWrapping;
    viewTexture.wrapT = THREE.ClampToEdgeWrapping;
    viewTexture.wrapR = THREE.ClampToEdgeWrapping;
    viewTexture.name = 'CodexBoxFogView';
    this.viewTexture = viewTexture;
    this.viewSampler = texture3D(viewTexture as any, null, 0);

    this.createViewCompute();
  }

  compute(renderer: WebGPURenderer): void {
    if (this.viewCompute) {
      renderer.compute(this.viewCompute);
    }
  }

  dispose(): void {
    if (this.viewTexture) {
      this.viewTexture.dispose();
      this.viewTexture = null;
    }
    this.viewSampler = null;
    this.viewCompute = null;
  }

  getViewSampler(): ReturnType<typeof texture3D> | null {
    return this.viewSampler;
  }

  setFogVolumes(volumes: FogVolumeConfig[]): void {
    const maxCount = Math.min(volumes.length, MAX_FOG_VOLUMES);
    for (let i = 0; i < MAX_FOG_VOLUMES; i += 1) {
      const volume = i < maxCount ? volumes[i] : null;
      const enabled = volume?.enabled !== false ? 1 : 0;
      this.fogVolumeEnabledUniforms[i].value = enabled;
      if (volume) {
        this.fogVolumeCenterUniforms[i].value.copy(volume.center);
        this.fogVolumeParamsUniforms[i].value.set(
          volume.radius,
          volume.density,
          volume.falloff ?? 0,
          0
        );
      } else {
        this.fogVolumeCenterUniforms[i].value.set(0, 0, 0);
        this.fogVolumeParamsUniforms[i].value.set(0, 0, 0, 0);
      }
    }
  }

  private createViewCompute(): void {
    if (!this.viewTexture) return;

    const res = this.froxelResolutionUniform;
    const cameraProjInv = this.cameraProjectionInverseUniform;
    const cameraWorld = this.cameraWorldMatrixUniform;
    const stepSize = this.froxelStepUniform;
    const fogDensity = this.fogDensityUniform;
    const isOrtho = this.cameraIsOrthographicUniform;
    const near = this.cameraNearUniform;
    const far = this.cameraFarUniform;
    const stepRange = far.sub(near);
    const shadowMap = this.shadowDepthTextureNode;
    const shadowReady = this.shadowReadyUniform;
    const shadowBias = this.shadowBiasUniform;
    const shadowMapSize = this.shadowMapSizeUniform;
    const shadowMatrix = this.shadowMatrixUniform;
    const shadowWebGPU = this.shadowWebGPUUniform;
    const fogVolumeEnabled = this.fogVolumeEnabledUniforms;
    const fogVolumeCenters = this.fogVolumeCenterUniforms;
    const fogVolumeParams = this.fogVolumeParamsUniforms;

    const computeFn = Fn(({ storageTexture }: { storageTexture: any }) => {
      const resX = res.x;
      const resY = res.y;
      const resZ = res.z;
      const idx = instanceIndex;
      const ix = idx.mod(resX);
      const iy = idx.div(resX).mod(resY);

      const fx = ix.toFloat().add(float(0.5)).div(resX);
      const fy = iy.toFloat().add(float(0.5)).div(resY);

      const transmittance = float(1.0).toVar();
      const accumulated = vec3(float(0.0)).toVar();

      Loop({ start: 0, end: MAX_FROXEL_SLICES, type: 'int', condition: '<' }, ({ i }) => {
        const sliceCount = resZ.toInt();
        If(i.greaterThanEqual(sliceCount), () => {
          Break();
        });

        const slice = i;
        const fz = slice.toFloat().add(float(0.5)).div(resZ);
        const screenY = float(1.0).sub(fy);
        const ndc = vec4(
          fx.mul(float(2)).sub(float(1)),
          screenY.mul(float(2)).sub(float(1)),
          float(1),
          float(1)
        );
        const viewFar = cameraProjInv.mul(ndc);
        const viewDir = normalize(viewFar.xyz.div(viewFar.w));
        const viewZ = near.add(stepRange.mul(fz)).negate();
        const safeViewZ = min(viewDir.z, float(-0.0001));
        const viewPos = vec3(float(0.0)).toVar();
        If(isOrtho.greaterThan(float(0.5)), () => {
          viewPos.assign(vec3(viewFar.x, viewFar.y, viewZ));
        }).Else(() => {
          viewPos.assign(viewDir.mul(viewZ.div(safeViewZ)));
        });

        const worldPos = cameraWorld.mul(vec4(viewPos, float(1))).xyz;
        const density = fogDensity.toVar();
        for (let i = 0; i < MAX_FOG_VOLUMES; i += 1) {
          const enabled = fogVolumeEnabled[i];
          const center = fogVolumeCenters[i];
          const params = fogVolumeParams[i];
          If(enabled.greaterThan(float(0.5)), () => {
            const dist = length(worldPos.sub(center));
            const radius = params.x;
            const baseDensity = params.y;
            const falloff = params.z;
            const sd = dist.sub(radius);
            const falloffSafe = max(falloff, float(0.0001));
            const falloffT = clamp(
              float(1.0).sub(sd.div(falloffSafe)),
              float(0.0),
              float(1.0)
            );
            const hasFalloff = falloff.greaterThan(float(0.0001));
            const densityWithFalloff = baseDensity.mul(falloffT);
            const densityNoFalloff = select(sd.lessThanEqual(float(0.0)), baseDensity, float(0.0));
            const localDensity = select(hasFalloff, densityWithFalloff, densityNoFalloff);
            density.assign(max(density, localDensity));
          });
        }
        const extinction = density.mul(float(0.06));
        const attenuation = exp(extinction.mul(stepSize).negate());
        const stepWeight = float(1.0).sub(attenuation);
        const lightFactor = float(1.0).toVar();

        If(shadowReady.greaterThan(float(0.5)), () => {
          const shadowPos = shadowMatrix.mul(vec4(worldPos, float(1.0)));
          const invW = float(1.0).div(max(abs(shadowPos.w), float(0.0001)));
          const shadowCoord = shadowPos.xyz.mul(invW);
          const coordZ = select(
            shadowWebGPU.greaterThan(float(0.5)),
            shadowCoord.z.mul(float(2.0)).sub(float(1.0)),
            shadowCoord.z
          );
          const shadowUV = vec2(shadowCoord.x, shadowCoord.y.oneMinus());
          const shadowDepth = coordZ.add(shadowBias);
          const inBounds = shadowUV.x.greaterThanEqual(float(0.0))
            .and(shadowUV.x.lessThanEqual(float(1.0)))
            .and(shadowUV.y.greaterThanEqual(float(0.0)))
            .and(shadowUV.y.lessThanEqual(float(1.0)));
          If(inBounds, () => {
            const maxUv = shadowMapSize.sub(float(1.0)).div(shadowMapSize);
            const shadowUvClamped = clamp(shadowUV, vec2(float(0.0)), maxUv);
            const shadowTexel = uvec2(shadowUvClamped.mul(shadowMapSize));
            const shadowSample = shadowMap.load(shadowTexel);
            const occluderPresent = shadowSample.lessThan(float(0.999));
            If(occluderPresent, () => {
              const lit = shadowDepth.lessThanEqual(shadowSample);
              const visibilitySoft = mix(float(0.2), float(1.0), lit);
              lightFactor.assign(visibilitySoft);
            });
          });
        });

        accumulated.addAssign(vec3(stepWeight.mul(transmittance).mul(lightFactor)));
        transmittance.mulAssign(attenuation);

        const storeCoord = uvec3(ix, iy, slice.toUint());
        textureStore(storageTexture, storeCoord, vec4(accumulated, transmittance)).toWriteOnly();
      });
    });

    const resolution = this.froxelResolutionUniform.value;
    const totalCells = Math.max(1, Math.floor(resolution.x * resolution.y));
    this.viewCompute = computeFn({ storageTexture: this.viewTexture }).compute(totalCells);
  }
}
