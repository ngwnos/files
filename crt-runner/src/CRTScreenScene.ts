/**
 * CRT Screen Scene Class
 * 
 * A WebGPU scene that renders a virtual CRT screen with RGB subpixels
 * and various distortion effects using vertex and fragment shaders.
 */

import * as THREE from 'three/webgpu';
import { WebGPURenderer, MeshBasicNodeMaterial, MeshStandardNodeMaterial, PostProcessing, StorageTexture } from 'three/webgpu';
import { storage, uniform, instanceIndex, Fn, Loop, If, Break, float, vec3, vec4, vec2, uint, floor, clamp, color, positionLocal, positionWorld, normalWorld, cameraPosition, uv, texture, select, pass, max, min, mrt, output, emissive, smoothstep, pow, sqrt, inversesqrt, mix, normalize, abs, add, sub, mul, div, instancedArray, sin, cos, fract, dot, hash, dFdx, dFdy, attribute, cross, textureStore, ivec2, log } from 'three/tsl';
import { bloom } from 'three/examples/jsm/tsl/display/BloomNode.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

const CP437_TO_UNICODE: number[] = [
    0x0000, 0x263A, 0x263B, 0x2665, 0x2666, 0x2663, 0x2660, 0x2022,
    0x25D8, 0x25CB, 0x25D9, 0x2642, 0x2640, 0x266A, 0x266B, 0x263C,
    0x25BA, 0x25C4, 0x2195, 0x203C, 0x00B6, 0x00A7, 0x25AC, 0x21A8,
    0x2191, 0x2193, 0x2192, 0x2190, 0x221F, 0x2194, 0x25B2, 0x25BC,
    0x0020, 0x0021, 0x0022, 0x0023, 0x0024, 0x0025, 0x0026, 0x0027,
    0x0028, 0x0029, 0x002A, 0x002B, 0x002C, 0x002D, 0x002E, 0x002F,
    0x0030, 0x0031, 0x0032, 0x0033, 0x0034, 0x0035, 0x0036, 0x0037,
    0x0038, 0x0039, 0x003A, 0x003B, 0x003C, 0x003D, 0x003E, 0x003F,
    0x0040, 0x0041, 0x0042, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047,
    0x0048, 0x0049, 0x004A, 0x004B, 0x004C, 0x004D, 0x004E, 0x004F,
    0x0050, 0x0051, 0x0052, 0x0053, 0x0054, 0x0055, 0x0056, 0x0057,
    0x0058, 0x0059, 0x005A, 0x005B, 0x005C, 0x005D, 0x005E, 0x005F,
    0x0060, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065, 0x0066, 0x0067,
    0x0068, 0x0069, 0x006A, 0x006B, 0x006C, 0x006D, 0x006E, 0x006F,
    0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077,
    0x0078, 0x0079, 0x007A, 0x007B, 0x007C, 0x007D, 0x007E, 0x2302,
    0x00C7, 0x00FC, 0x00E9, 0x00E2, 0x00E4, 0x00E0, 0x00E5, 0x00E7,
    0x00EA, 0x00EB, 0x00E8, 0x00EF, 0x00EE, 0x00EC, 0x00C4, 0x00C5,
    0x00C9, 0x00E6, 0x00C6, 0x00F4, 0x00F6, 0x00F2, 0x00FB, 0x00F9,
    0x00FF, 0x00D6, 0x00DC, 0x00A2, 0x00A3, 0x00A5, 0x20A7, 0x0192,
    0x00E1, 0x00ED, 0x00F3, 0x00FA, 0x00F1, 0x00D1, 0x00AA, 0x00BA,
    0x00BF, 0x2310, 0x00AC, 0x00BD, 0x00BC, 0x00A1, 0x00AB, 0x00BB,
    0x2591, 0x2592, 0x2593, 0x2502, 0x2524, 0x2561, 0x2562, 0x2556,
    0x2555, 0x2563, 0x2551, 0x2557, 0x255D, 0x255C, 0x255B, 0x2510,
    0x2514, 0x2534, 0x252C, 0x251C, 0x2500, 0x253C, 0x255E, 0x255F,
    0x255A, 0x2554, 0x2569, 0x2566, 0x2560, 0x2550, 0x256C, 0x2567,
    0x2568, 0x2564, 0x2565, 0x2559, 0x2558, 0x2552, 0x2553, 0x256B,
    0x256A, 0x2518, 0x250C, 0x2588, 0x2584, 0x258C, 0x2590, 0x2580,
    0x03B1, 0x00DF, 0x0393, 0x03C0, 0x03A3, 0x03C3, 0x00B5, 0x03C4,
    0x03A6, 0x0398, 0x03A9, 0x03B4, 0x221E, 0x03C6, 0x03B5, 0x2229,
    0x2261, 0x00B1, 0x2265, 0x2264, 0x2320, 0x2321, 0x00F7, 0x2248,
    0x00B0, 0x2219, 0x00B7, 0x221A, 0x207F, 0x00B2, 0x25A0, 0x00A0
];

const CP437_TO_UNICODE_STR: string[] = CP437_TO_UNICODE.map(code => String.fromCodePoint(code));
const UNICODE_TO_CP437: Map<number, number> = (() => {
    const map = new Map<number, number>();
    CP437_TO_UNICODE.forEach((code, idx) => {
        if (!map.has(code)) {
            map.set(code, idx);
        }
    });
    return map;
})();

type KeyboardLabel = {
    top?: string;
    bottom?: string;
    center?: string;
    align?: 'left' | 'center';
    topRotate?: number;
    bottomRotate?: number;
    centerRotate?: number;
    ids?: string[];
    suppressDefaultIds?: boolean;
};

export interface CRTScreenSceneParameters {
    displayMode?: 'video' | 'emulator' | 'static' | 'terminal' | 'xterm' | 'shader';
    emulatorSource?: 'snes' | 'dos';
    emulatorRom?: string;
    emulatorLoadRom?: boolean;
    dosBundle?: string;
    dosLoadBundle?: boolean;
    screenResolution?: string;
    shaderType?: 'mandelbrot' | 'julia';
    shaderPanX?: number;
    shaderPanY?: number;
    shaderZoom?: number;
    mandelbrotPanX?: number;
    mandelbrotPanY?: number;
    mandelbrotZoom?: number;
    juliaPanX?: number;
    juliaPanY?: number;
    juliaZoom?: number;
    juliaCReal?: number;
    juliaCImag?: number;
    terminalFontScale?: number;
    terminalFontColor?: string;
    moireStrength?: number;
    moireChroma?: number;
    moireFeather?: number;
    moireThreshold?: number;
    // Screen physical layout
    screenWidth?: number;        // physical width in world units
    screenHeight?: number;       // physical height in world units
    minBrightness?: number;      // minimum brightness for pixels
    brightness?: number;         // Overall brightness multiplier (0-2)
    
    // Static pattern
    staticSpeed?: number;        // How fast the static updates (multiplier for time)
    staticContrast?: number;     // Amount of gray vs pure black/white (0-1)
    
    // Phosphor appearance
    slotDutyX?: number;          // horizontal subpixel fill (0.1-1.0)
    slotDutyY?: number;          // vertical subpixel fill (0.1-1.0)
    subpixelFeather?: number;    // edge softness for anti-aliasing
    phosphorTint?: number;       // secondary channel brightness (0-0.5)
    
    // Beam physics
    beamGamma?: number;          // beam falloff exponent
    beamSpread?: number;         // beam spread factor
    vignetteStrength?: number;   // corner darkening amount
    phaseShearAmount?: number;   // edge phase distortion
    scanFramerate?: number;      // How many times per second to scan all pixels (e.g., 30)
    beamPixelDuration?: number;  // Effective time the beam dwells on each pixel (multiplier)
    
    // Color interpolation
    colorAttack?: number;        // Attack rate for color changes (higher = faster)
    colorDecay?: number;         // Decay rate for color changes (higher = faster)
    
    // CRT distortion
    crtAmount?: number;
    crtBarrel?: number;
    crtKeystoneX?: number;
    crtKeystoneY?: number;
    crtZoom?: number;
    screenCurvature?: number;
    
    // Bloom
    bloomStrength?: number;
    bloomRadius?: number;
    bloomThreshold?: number;

    // Keyboard
    keyboardEnabled?: boolean;
    keyboardSolenoidVolume?: number;
    keyboardWidthScale?: number;
    keyboardDepthRatio?: number;
    keyboardTopInset?: number;
    keyboardBumpScale?: number;
    keyboardBumpStrength?: number;
    keyboardBaseColor?: string;
    keyboardRoughnessMin?: number;
    keyboardRoughnessMax?: number;
    keyboardMetalness?: number;
    keyboardLabelOpacity?: number;
    keyboardScreenLightIntensity?: number;
    keyboardLightSaturation?: number;
    keyboardLightWrap?: number;
    keyboardLightSampleGrid?: number;
    keyboardCornerRadiusBottom?: number;
    keyboardCornerRadiusTop?: number;
    keyboardTopEdgeRadius?: number;
    keyboardOffsetY?: number;
    keyboardOffsetZ?: number;

    // Power on/off effect
    powerOn?: boolean;
    powerOnDuration?: number;
    powerWarmupDuration?: number;
    powerOffDuration?: number;
    powerOffEndDuration?: number;
    powerFlash?: number;
}

export class CRTScreenScene {
    private static readonly BLOOM_LAYER = 1;
    private static readonly KEYBOARD_LIGHT_GRID_X = 4;
    private static readonly KEYBOARD_LIGHT_GRID_Y = 3;

    private renderer: WebGPURenderer | null = null;
    private scene: THREE.Scene | null = null;
    private camera: THREE.PerspectiveCamera | null = null;
    private bloomCamera: THREE.PerspectiveCamera | null = null;
    private controls: OrbitControls | null = null;
    private canvas: HTMLCanvasElement | null = null;
    
    private screenMesh: THREE.Mesh | null = null;
    private keyboardGroup: THREE.Group | null = null;
    private keyboardOuterMesh: THREE.Mesh | null = null;
    private keyboardInnerMesh: THREE.Mesh | null = null;
    private keyboardKeyMeshes: Map<string, THREE.Mesh[]> = new Map();
    private keyboardKeyMeshList: THREE.Mesh[] = [];
    private keyboardLoading = false;
    private keyboardLights: THREE.Object3D[] = [];
    private keyboardLabelTexture: THREE.CanvasTexture | null = null;
    private keyboardScreenLight: THREE.DirectionalLight | null = null;
    private keyboardLightIntensityUniform = uniform(1.0, 'float');
    private keyboardLightSaturationUniform = uniform(1.0, 'float');
    private keyboardLightWrapUniform = uniform(0.3, 'float');
    private keyboardLightOriginUniform = uniform(new THREE.Vector3(0, 0, 0));
    private keyboardLightScreenRightUniform = uniform(new THREE.Vector3(1, 0, 0));
    private keyboardLightScreenUpUniform = uniform(new THREE.Vector3(0, 1, 0));
    private keyboardLightScreenNormalUniform = uniform(new THREE.Vector3(0, 0, 1));
    private keyboardLightScreenSizeUniform = uniform(new THREE.Vector2(1, 1));
    private keyboardBaseColorUniform: any = null;
    private keyboardBumpScaleUniform: any = null;
    private keyboardBumpStrengthUniform: any = null;
    private keyboardRoughnessMinUniform: any = null;
    private keyboardRoughnessMaxUniform: any = null;
    private keyboardMetalnessUniform: any = null;
    private keyboardLabelOpacityUniform: any = null;
    private keyboardBaseKeyWidth = 0;
    private keyboardRebuildHandle: number | null = null;
    private keyboardRebuildUsesIdle = false;
    private screenLightTarget: THREE.RenderTarget | null = null;
    private screenLightTexture: StorageTexture | null = null;
    private screenLightComputeNode: any = null;
    private screenLightTextureNode: any = null;
    private screenLightSampleGrid = 4;
    private screenLightPostProcessing: PostProcessing | null = null;
    private screenLightScenePass: any = null;
    private screenLightBloomPass: any = null;
    private screenLightBloomNode: any = null;
    private screenLightLayers: THREE.Layers | null = null;
    private screenLightCamera: THREE.OrthographicCamera | null = null;
    private screenLightOriginScratch = new THREE.Vector3();
    private screenLightRightScratch = new THREE.Vector3();
    private screenLightUpScratch = new THREE.Vector3();
    private screenLightNormalScratch = new THREE.Vector3();
    private screenLightQuatScratch = new THREE.Quaternion();
    private screenLightCameraPosScratch = new THREE.Vector3();
    private keyboardLayout: {
        keys: Array<{
            outer: { x: number; y: number; width: number; height: number; rx: number };
            inner: { x: number; y: number; width: number; height: number; rx: number };
        }>;
        bounds: { minX: number; minY: number; maxX: number; maxY: number };
    } | null = null;
    
    // Post-processing
    private postProcessing: PostProcessing | null = null;
    private bloomNode: any = null;
    
    // Pixel screen uniforms
    // CRT distortion uniforms
    private crtAmountUniform: any = null;
    private crtBarrelUniform: any = null;
    private crtKeystoneXUniform: any = null;
    private crtKeystoneYUniform: any = null;
    private crtZoomUniform: any = null;
    private screenCurvatureUniform: any = null;
    
    // Brightness and color interpolation uniforms
    private minBrightnessUniform: any = null;
    private brightnessUniform: any = null;
    private colorAttackUniform: any = null;
    private colorDecayUniform: any = null;

    // Power effect uniforms
    private powerOnUniform: any = null;
    private powerTransitionUniform: any = null;
    private powerDirectionUniform: any = null;
    private powerFlashUniform: any = null;
    private powerWarmupUniform: any = null;
    private powerCollapseRatioUniform: any = null;
    private useExternalContentUniform: any = null;
    private useExternalTextureUniform: any = null;
    
    // Static pattern uniforms
    private staticSpeedUniform: any = null;
    private staticContrastUniform: any = null;
    
    // Phosphor uniforms
    private slotDutyXUniform: any = null;
    private slotDutyYUniform: any = null;
    private subpixelFeatherUniform: any = null;
    private phosphorTintUniform: any = null;
    private moireStrengthUniform: any = null;
    private moireChromaUniform: any = null;
    private moireFeatherUniform: any = null;
    private moireThresholdUniform: any = null;
    private screenLightModeUniform: any = null;
    
    // Beam physics uniforms
    private beamGammaUniform: any = null;
    private beamSpreadUniform: any = null;
    private vignetteStrengthUniform: any = null;
    private phaseShearAmountUniform: any = null;
    
    // Beam scan uniforms
    private scanFramerateUniform: any = null;
    private scanHeadUniform: any = null;
    private beamPixelDurationUniform: any = null;
    
    // GPU compute shader color system
    private currentColors: any = null;  // GPU storage buffer for current colors
    private targetColors: any = null;   // GPU storage buffer for target colors
    private colorComputeNode: any = null; // Compute shader for interpolation
    private shaderComputeNode: any = null; // Compute shader for per-pixel shader content
    private targetColorArray: Float32Array | null = null; // CPU-side target buffer (only updated when needed)
    private targetColorsNeedUpdate = false;

    // External content source (images/video)
    private contentCanvas: HTMLCanvasElement | null = null;
    private contentContext: CanvasRenderingContext2D | null = null;
    private contentTexture: THREE.CanvasTexture | null = null;
    private contentTextureNode: any = null;
    private contentSource: CanvasImageSource | null = null;
    private contentIsVideo = false;
    private contentDirty = false;
    private useExternalContent = false;
    private videoFrameHandle: number | null = null;
    private videoFrameSource: HTMLVideoElement | null = null;

    // Terminal rendering
    private terminalDirty = false;
    private terminalCols = 0;
    private terminalRows = 0;
    private terminalCursorX = 0;
    private terminalCursorY = 0;
    private terminalBuffer: Uint16Array | null = null;
    private terminalColorBuffer: Uint32Array | null = null;
    private terminalCurrentColor = 0x33ff66;
    private terminalColorCache = new Map<number, string>();
    private terminalColorScratch = new THREE.Color();
    private terminalFontLoaded = false;
    private terminalFontLoading = false;
    private readonly terminalFontFamily = 'IBM VGA 8x16';
    private shaderOverlayFontLoaded = false;
    private shaderOverlayFontLoading = false;
    private readonly shaderOverlayFontFamily = 'Compaq Thin 8x16';
    private readonly terminalCharWidth = 8;
    private readonly terminalCharHeight = 16;
    private readonly terminalPaddingChars = 1;
    private readonly terminalForeground = '#33ff66';
    private readonly terminalBackground = '#000000';
    private terminalCursorBlinkTime = 0;
    private readonly terminalCursorBlinkInterval = 0.5;
    private terminalCursorVisible = true;
    
    // Animation timing
    private lastUpdateTime: number = 0;
    private timeUniform = uniform(0, 'float');
    private deltaTimeUniform = uniform(0, 'float');
    
    // Beam scan state
    private scanHead: number = 0;  // Current scan position (fractional)

    // Power transition state
    private powerTransition = 1;
    private powerDirection = 0;
    private powerWarmup = 1;
    
    // Default resolution and scale (matches previous VGA sizing)
    private static readonly DEFAULT_LOGICAL_WIDTH = 640;
    private static readonly DEFAULT_LOGICAL_HEIGHT = 480;
    private static readonly DEFAULT_SCREEN_WIDTH = 64.0;
    private static readonly DEFAULT_SCREEN_HEIGHT = 48.0;
    private static readonly DEFAULT_KEYBOARD_WIDTH = 32.0;
    private static readonly DEFAULT_KEYBOARD_BASE_Y = -21.6;
    private static readonly DEFAULT_KEYBOARD_BASE_Z = 12.8;
    private static readonly DEFAULT_RESOLUTION_PRESET =
        `${CRTScreenScene.DEFAULT_LOGICAL_WIDTH}x${CRTScreenScene.DEFAULT_LOGICAL_HEIGHT}`;
    private static readonly SHADER_MAX_ITER = 80;

    private logicalWidth = CRTScreenScene.DEFAULT_LOGICAL_WIDTH;
    private logicalHeight = CRTScreenScene.DEFAULT_LOGICAL_HEIGHT;
    private shaderPanXUniform: any = null;
    private shaderPanYUniform: any = null;
    private shaderZoomUniform: any = null;
    private shaderEnabledUniform: any = null;
    private shaderTypeUniform: any = null;
    private shaderJuliaCUniform: any = null;

    private getActiveShaderParams(): { panX: number; panY: number; zoom: number } {
        const isJulia = (this.parameters.shaderType ?? 'mandelbrot') === 'julia';
        if (isJulia) {
            return {
                panX: this.parameters.juliaPanX ?? 0.0,
                panY: this.parameters.juliaPanY ?? 0.0,
                zoom: this.parameters.juliaZoom ?? 1.0
            };
        }
        return {
            panX: this.parameters.mandelbrotPanX ?? this.parameters.shaderPanX ?? -0.75,
            panY: this.parameters.mandelbrotPanY ?? this.parameters.shaderPanY ?? 0.0,
            zoom: this.parameters.mandelbrotZoom ?? this.parameters.shaderZoom ?? 1.0
        };
    }
    
    // Scene parameters
    private parameters: CRTScreenSceneParameters = {
        displayMode: 'shader',
        emulatorSource: 'snes',
        emulatorRom: 'roms/jetpilotrising.sfc',
        emulatorLoadRom: false,
        dosBundle: 'dos/digger.jsdos',
        dosLoadBundle: false,
        screenResolution: CRTScreenScene.DEFAULT_RESOLUTION_PRESET,
        shaderType: 'julia',
        mandelbrotPanX: -0.75,
        mandelbrotPanY: 0.0,
        mandelbrotZoom: 1.0,
        juliaPanX: 0.0,
        juliaPanY: 0.0,
        juliaZoom: 1.0,
        juliaCReal: -0.8,
        juliaCImag: 0.156,
        terminalFontScale: 3,
        terminalFontColor: '#33ff66',
        moireStrength: 2.0,
        moireChroma: 1.0,
        moireFeather: 0.0,
        moireThreshold: 0.5,
        screenWidth: CRTScreenScene.DEFAULT_SCREEN_WIDTH,
        screenHeight: CRTScreenScene.DEFAULT_SCREEN_HEIGHT,
        minBrightness: 0.01,
        brightness: 1.19,             // Slightly boosted baseline
        powerOn: true,
        powerOnDuration: 0.8,
        powerWarmupDuration: 1.6,
        powerOffDuration: 0.45,
        powerOffEndDuration: 0.0,
        powerFlash: 0.6,
        staticSpeed: 15.0,           // Fast updates for realistic TV static
        staticContrast: 0.2,         // 20% gray values, 80% black/white
        slotDutyX: 0.65,             // 65% horizontal fill
        slotDutyY: 0.68,             // 68% vertical fill
        subpixelFeather: 0.08,
        phosphorTint: 0.15,
        colorAttack: 20.0,    // Fast attack for responsive color changes
        colorDecay: 15.0,     // Slower decay for smooth fading
        beamGamma: 1.6,
        beamSpread: 1.3,
        vignetteStrength: 0.1,
        phaseShearAmount: 0.0,
        crtAmount: 0.0,
        crtBarrel: 0.0,
        crtKeystoneX: 0.0,
        crtKeystoneY: 0.0,
        crtZoom: 1.0,
        screenCurvature: 0.2,
        bloomStrength: 1.88,
        bloomRadius: 0.0,
        bloomThreshold: 0.0,
        scanFramerate: 30,           // Scan all pixels 30 times per second
        beamPixelDuration: 5.0,      // Effective dwell time multiplier
        keyboardEnabled: true,
        keyboardSolenoidVolume: 1.0,
        keyboardWidthScale: 1.0,
        keyboardDepthRatio: 0.4,
        keyboardTopInset: 0.21,
        keyboardBumpScale: 6.0,
        keyboardBumpStrength: 0.005,
        keyboardBaseColor: '#DAD8CE',
        keyboardRoughnessMin: 0.78,
        keyboardRoughnessMax: 0.92,
        keyboardMetalness: 0.0,
        keyboardLabelOpacity: 1.0,
        keyboardScreenLightIntensity: 2.05,
        keyboardLightSaturation: 1.0,
        keyboardLightWrap: 0.0,
        keyboardLightSampleGrid: 8,
        keyboardCornerRadiusBottom: 0.015,
        keyboardCornerRadiusTop: 0.15,
        keyboardTopEdgeRadius: 0.02,
        keyboardOffsetY: 0.0,
        keyboardOffsetZ: 0.0
    };
    
    private time = 0;

    constructor() {
        console.log('CRTScreenScene constructor');
    }

    private parseResolutionPreset(value?: string): { width: number; height: number } | null {
        if (!value) {
            return null;
        }
        const match = value.trim().match(/^(\d+)\s*x\s*(\d+)$/i);
        if (!match) {
            return null;
        }
        const width = Number(match[1]);
        const height = Number(match[2]);
        if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
            return null;
        }
        return { width, height };
    }

    private setLogicalResolution(value?: string): void {
        const parsed = this.parseResolutionPreset(value);
        if (!parsed) {
            return;
        }
        if (parsed.width === this.logicalWidth && parsed.height === this.logicalHeight) {
            return;
        }
        this.logicalWidth = parsed.width;
        this.logicalHeight = parsed.height;
        this.rebuildResolution();
    }

    private getScreenPhysicalWidth(): number {
        return this.parameters.screenWidth ?? CRTScreenScene.DEFAULT_SCREEN_WIDTH;
    }

    private getScreenPhysicalHeight(): number {
        return this.parameters.screenHeight ?? CRTScreenScene.DEFAULT_SCREEN_HEIGHT;
    }

    private getKeyboardLightSampleGrid(): number {
        const raw = this.parameters.keyboardLightSampleGrid ?? 4;
        return Math.max(1, Math.min(8, Math.floor(raw)));
    }

    private rebuildResolution(): void {
        if (!this.renderer || !this.scene) {
            return;
        }

        this.scanHead = 0;
        this.targetColorsNeedUpdate = false;
        if (this.contentTexture) {
            this.contentTexture.dispose();
        }
        this.contentTexture = null;
        this.contentTextureNode = null;
        this.ensureContentCanvas();
        this.contentDirty = true;
        this.resetTerminalBuffer();
        this.clearVideoFrameCallback();
        this.currentColors = null;
        this.targetColors = null;
        this.colorComputeNode = null;
        this.shaderComputeNode = null;
        this.targetColorArray = null;
        this.screenLightComputeNode = null;
        this.screenLightSampleGrid = this.getKeyboardLightSampleGrid();

        this.initializeGPUComputeShaders();
        this.createCRTScreen();
        this.applyDisplayMode();
        this.refreshKeyboardLayout();
    }

    async init(canvas: HTMLCanvasElement, renderer: WebGPURenderer): Promise<void> {
        this.canvas = canvas;
        this.renderer = renderer;
        
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
        
        // Camera setup
        this.camera = new THREE.PerspectiveCamera(
            75,
            canvas.width / canvas.height,
            0.1,
            1000
        );
        this.camera.position.set(0, 0, 50);  // Good viewing distance for default screen
        this.camera.lookAt(0, 0, 0);
        this.camera.layers.enable(CRTScreenScene.BLOOM_LAYER);

        this.bloomCamera = this.camera.clone();
        this.bloomCamera.layers.set(CRTScreenScene.BLOOM_LAYER);
        
        // Controls
        this.controls = new OrbitControls(this.camera, canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        // Initialize GPU compute shaders FIRST
        this.initializeGPUComputeShaders();
        
        // Create the CRT screen visualization
        this.createCRTScreen();
        
        // Initialize post-processing
        this.initPostProcessing();
        this.applyDisplayMode();

        // Add keyboard layout (async)
        if (this.parameters.keyboardEnabled ?? true) {
            this.loadKeyboardLayout();
        }
    }

    update(deltaTime: number): void {
        const clampedDelta = Math.min(deltaTime, 1 / 60);
        this.time += clampedDelta;
        
        // Update time uniforms for GPU compute shader
        this.timeUniform.value = this.time;
        this.deltaTimeUniform.value = clampedDelta;

        // Update power transition
        this.updatePowerTransition(clampedDelta);
        this.updatePowerWarmup(clampedDelta);

        if ((this.parameters.displayMode ?? 'video') === 'terminal') {
            this.updateTerminalBlink(clampedDelta);
        }
        
        // Update beam scan position
        if (this.scanHeadUniform) {
            const totalSubpixels = this.logicalWidth * this.logicalHeight * 3;
            const scanFramerate = this.parameters.scanFramerate ?? 30;
            const pixelsPerSecond = totalSubpixels * scanFramerate;
            const pixelsThisFrame = pixelsPerSecond * clampedDelta;
            
            // Update scan head position
            this.scanHead = (this.scanHead + pixelsThisFrame) % totalSubpixels;
            
            // Update uniforms for GPU
            this.scanHeadUniform.value = this.scanHead;
        }
        
        // Update target colors on GPU if needed (rare)
        this.updateContentFrame();
        const isShaderMode = (this.parameters.displayMode ?? 'video') === 'shader';
        if (!isShaderMode && this.targetColorsNeedUpdate) {
            this.updateGPUTargetColors();
            this.targetColorsNeedUpdate = false;
        }

        if (isShaderMode) {
            if (this.shaderComputeNode && this.renderer) {
                this.renderer.computeAsync(this.shaderComputeNode);
            }
        }
        
        // Execute GPU compute shader for color interpolation
        if (this.colorComputeNode && this.renderer) {
            this.renderer.computeAsync(this.colorComputeNode);
        }
        
        // Update controls
        if (this.controls) {
            this.controls.update();
        }
        this.syncBloomCamera();
        this.updateKeyboardScreenLight(clampedDelta);
    }

    render(): void {
        if (!this.renderer || !this.scene || !this.camera) return;
        
        if (this.postProcessing) {
            this.postProcessing.render();
        } else {
            this.renderer.render(this.scene, this.camera);
        }
    }

    onResize(width: number, height: number): void {
        if (!this.renderer || !this.camera) return;

        const safeWidth = Math.max(1, Math.floor(width));
        const safeHeight = Math.max(1, Math.floor(height));

        this.renderer.setSize(safeWidth, safeHeight, false);
        this.camera.aspect = safeWidth / safeHeight;
        this.camera.updateProjectionMatrix();
        if (this.bloomCamera) {
            this.bloomCamera.aspect = safeWidth / safeHeight;
            this.bloomCamera.updateProjectionMatrix();
        }
        this.updateScreenLightResolutionScale();
    }

    cleanup(): void {
        if (this.controls) {
            this.controls.dispose();
        }

        this.clearVideoFrameCallback();
        this.cancelKeyboardRebuild();

        if (this.screenMesh) {
            this.scene?.remove(this.screenMesh);
            if (this.screenMesh.geometry) this.screenMesh.geometry.dispose();
            if (this.screenMesh.material) {
                (this.screenMesh.material as THREE.Material).dispose();
            }
        }

        if (this.keyboardGroup && this.scene) {
            this.scene.remove(this.keyboardGroup);
        }
        if (this.keyboardOuterMesh) {
            (this.keyboardOuterMesh.material as THREE.Material).dispose();
        }
        if (this.keyboardInnerMesh) {
            this.keyboardInnerMesh.geometry.dispose();
            (this.keyboardInnerMesh.material as THREE.Material).dispose();
        }
        if (this.keyboardLights.length > 0 && this.scene) {
            this.keyboardLights.forEach(light => this.scene?.remove(light));
        }
        if (this.keyboardKeyMeshList.length > 0) {
            this.keyboardKeyMeshList.forEach(mesh => mesh.geometry.dispose());
        }
        this.keyboardKeyMeshList = [];
        this.keyboardKeyMeshes.clear();
        
        // GPU resources are automatically cleaned up
        this.currentColors = null;
        this.targetColors = null;
        this.targetColorArray = null;
        this.colorComputeNode = null;
        this.shaderComputeNode = null;
        this.powerOnUniform = null;
        this.powerTransitionUniform = null;
        this.powerDirectionUniform = null;
        this.powerFlashUniform = null;
        this.powerWarmupUniform = null;
        this.powerCollapseRatioUniform = null;
        this.useExternalContentUniform = null;
        this.useExternalTextureUniform = null;
        this.moireStrengthUniform = null;
        this.moireChromaUniform = null;
        this.moireFeatherUniform = null;
        this.moireThresholdUniform = null;
        this.screenLightModeUniform = null;
        this.contentCanvas = null;
        this.contentContext = null;
        if (this.contentTexture) {
            this.contentTexture.dispose();
        }
        this.contentTexture = null;
        this.contentTextureNode = null;
        this.contentSource = null;
        this.contentDirty = false;
        this.contentIsVideo = false;
        this.useExternalContent = false;
        this.videoFrameHandle = null;
        this.videoFrameSource = null;
        this.terminalDirty = false;
        this.terminalCols = 0;
        this.terminalRows = 0;
        this.terminalCursorX = 0;
        this.terminalCursorY = 0;
        this.terminalBuffer = null;
        this.terminalColorBuffer = null;
        this.terminalColorCache.clear();
        this.terminalFontLoaded = false;
        this.terminalFontLoading = false;
        this.terminalCursorBlinkTime = 0;
        this.terminalCursorVisible = true;
        this.keyboardGroup = null;
        this.keyboardOuterMesh = null;
        this.keyboardInnerMesh = null;
        this.keyboardLoading = false;
        this.keyboardLights = [];
        if (this.keyboardLabelTexture) {
            this.keyboardLabelTexture.dispose();
        }
        this.keyboardLabelTexture = null;
        this.keyboardLayout = null;
        this.bloomCamera = null;
        this.keyboardScreenLight = null;
        this.screenLightTextureNode = null;
        if (this.screenLightPostProcessing) {
            this.screenLightPostProcessing.dispose();
        }
        this.screenLightPostProcessing = null;
        this.screenLightScenePass = null;
        this.screenLightBloomPass = null;
        this.screenLightBloomNode = null;
        this.screenLightLayers = null;
        this.screenLightCamera = null;
        if (this.screenLightTarget) {
            this.screenLightTarget.dispose();
        }
        this.screenLightTarget = null;
        if (this.screenLightTexture) {
            this.screenLightTexture.dispose();
        }
        this.screenLightTexture = null;
        this.screenLightComputeNode = null;
    }

    private syncBloomCamera(): void {
        if (!this.camera || !this.bloomCamera) {
            return;
        }
        this.bloomCamera.position.copy(this.camera.position);
        this.bloomCamera.quaternion.copy(this.camera.quaternion);
        this.bloomCamera.scale.copy(this.camera.scale);
        this.bloomCamera.updateMatrixWorld(true);
        this.bloomCamera.projectionMatrix.copy(this.camera.projectionMatrix);
        this.bloomCamera.projectionMatrixInverse.copy(this.camera.projectionMatrixInverse);
    }

    private ensureScreenLightCamera(): void {
        if (this.screenLightCamera) {
            return;
        }
        this.screenLightCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 200);
        this.screenLightCamera.layers.set(CRTScreenScene.BLOOM_LAYER);
    }

    private updateScreenLightCamera(): void {
        if (!this.screenMesh || !this.screenLightCamera) {
            return;
        }

        const screenWidth = this.getScreenPhysicalWidth();
        const screenHeight = this.getScreenPhysicalHeight();
        const halfWidth = screenWidth * 0.5;
        const halfHeight = screenHeight * 0.5;

        this.screenMesh.getWorldPosition(this.screenLightOriginScratch);
        this.screenMesh.getWorldQuaternion(this.screenLightQuatScratch);
        this.screenLightNormalScratch.set(0, 0, 1).applyQuaternion(this.screenLightQuatScratch).normalize();
        this.screenLightUpScratch.set(0, 1, 0).applyQuaternion(this.screenLightQuatScratch).normalize();

        const camera = this.screenLightCamera;
        camera.left = -halfWidth;
        camera.right = halfWidth;
        camera.top = halfHeight;
        camera.bottom = -halfHeight;

        const distance = Math.max(screenWidth, screenHeight) * 2 + 5;
        this.screenLightCameraPosScratch
            .copy(this.screenLightNormalScratch)
            .multiplyScalar(distance)
            .add(this.screenLightOriginScratch);
        camera.position.copy(this.screenLightCameraPosScratch);
        camera.up.copy(this.screenLightUpScratch);
        camera.near = 0.1;
        camera.far = distance + 50;
        camera.lookAt(this.screenLightOriginScratch);
        camera.updateProjectionMatrix();
        camera.updateMatrixWorld(true);
    }

    private ensureScreenLightDownsample(): void {
        if (!this.renderer || !this.currentColors) {
            return;
        }
        const sampleGrid = this.getKeyboardLightSampleGrid();
        if (sampleGrid !== this.screenLightSampleGrid) {
            this.screenLightSampleGrid = sampleGrid;
            this.screenLightComputeNode = null;
        }
        if (!this.screenLightTexture) {
            const width = CRTScreenScene.KEYBOARD_LIGHT_GRID_X;
            const height = CRTScreenScene.KEYBOARD_LIGHT_GRID_Y;
            const textureTarget = new StorageTexture(width, height);
            textureTarget.format = THREE.RGBAFormat;
            textureTarget.type = THREE.FloatType;
            textureTarget.generateMipmaps = false;
            textureTarget.name = 'CRTKeyboardLight';
            this.screenLightTexture = textureTarget;
            this.screenLightTextureNode = texture(textureTarget, null, 0);
        } else if (!this.screenLightTextureNode) {
            this.screenLightTextureNode = texture(this.screenLightTexture, null, 0);
        }
        if (!this.screenLightComputeNode && this.screenLightTexture) {
            const gridX = CRTScreenScene.KEYBOARD_LIGHT_GRID_X;
            const gridY = CRTScreenScene.KEYBOARD_LIGHT_GRID_Y;
            const totalCells = gridX * gridY;
            const screenWidth = this.logicalWidth;
            const screenHeight = this.logicalHeight;
            const screenWidthF = float(screenWidth);
            const screenHeightF = float(screenHeight);
            const totalColumnsU = uint(screenWidth * 3);
            const invGridX = 1 / gridX;
            const invGridY = 1 / gridY;
            const sampleStep = 1 / sampleGrid;
            const sampleWeight = 1 / (sampleGrid * sampleGrid);

            const computeFn = Fn(({ storageTexture }: { storageTexture: any }) => {
                const idx = instanceIndex;
                const cellX = idx.mod(uint(gridX));
                const cellY = idx.div(uint(gridX));
                const cellXf = cellX.toFloat();
                const cellYf = cellY.toFloat();
                const u0 = cellXf.mul(float(invGridX));
                const v0 = cellYf.mul(float(invGridY));
                const u1 = u0.add(float(invGridX));
                const v1 = v0.add(float(invGridY));

                let accum = vec3(0.0, 0.0, 0.0);
                for (let sy = 0; sy < sampleGrid; sy += 1) {
                    for (let sx = 0; sx < sampleGrid; sx += 1) {
                        const fracX = float((sx + 0.5) * sampleStep);
                        const fracY = float((sy + 0.5) * sampleStep);
                        const sampleU = u0.add(fracX.mul(u1.sub(u0)));
                        const sampleV = v0.add(fracY.mul(v1.sub(v0)));
                        const pixelXf = clamp(
                            floor(sampleU.mul(screenWidthF)),
                            float(0.0),
                            screenWidthF.sub(float(1.0))
                        );
                        const pixelYf = clamp(
                            floor(float(1.0).sub(sampleV).mul(screenHeightF)),
                            float(0.0),
                            screenHeightF.sub(float(1.0))
                        );
                        const pixelX = pixelXf.toUint();
                        const pixelY = pixelYf.toUint();
                        const baseSubpixelIndex = pixelY.mul(totalColumnsU).add(pixelX.mul(uint(3)));
                        const red = this.currentColors.element(baseSubpixelIndex).r;
                        const green = this.currentColors.element(baseSubpixelIndex.add(uint(1))).r;
                        const blue = this.currentColors.element(baseSubpixelIndex.add(uint(2))).r;
                        accum = accum.add(vec3(red, green, blue));
                    }
                }

                const averaged = accum.mul(float(sampleWeight));
                const storeCoord = ivec2(cellX.toInt(), cellY.toInt());
                textureStore(storageTexture, storeCoord, vec4(averaged, float(1.0))).toWriteOnly();
            });

            this.screenLightComputeNode = computeFn({ storageTexture: this.screenLightTexture }).compute(totalCells);
        }
    }

    private updateScreenLightTexture(): void {
        if (!this.renderer || !this.screenLightComputeNode) {
            return;
        }
        this.renderer.compute(this.screenLightComputeNode);
    }

    private getScreenLightResolutionScale(): number {
        if (!this.renderer) {
            return 0.01;
        }
        const size = new THREE.Vector2();
        this.renderer.getSize(size);
        const scaleX = CRTScreenScene.KEYBOARD_LIGHT_GRID_X / Math.max(1, size.x);
        const scaleY = CRTScreenScene.KEYBOARD_LIGHT_GRID_Y / Math.max(1, size.y);
        const scale = Math.min(scaleX, scaleY);
        return Math.max(0.001, Math.min(1.0, scale));
    }

    private updateScreenLightResolutionScale(): void {
        const scale = this.getScreenLightResolutionScale();
        if (this.screenLightScenePass?.setResolutionScale) {
            this.screenLightScenePass.setResolutionScale(scale);
        }
        if (this.screenLightBloomPass?.setResolutionScale) {
            this.screenLightBloomPass.setResolutionScale(scale);
        }
    }

    private updateKeyboardLightBasis(): void {
        if (!this.screenMesh) {
            return;
        }

        this.screenMesh.getWorldPosition(this.screenLightOriginScratch);
        this.keyboardLightOriginUniform.value.copy(this.screenLightOriginScratch);

        this.screenMesh.getWorldQuaternion(this.screenLightQuatScratch);
        this.screenLightRightScratch.set(1, 0, 0).applyQuaternion(this.screenLightQuatScratch).normalize();
        this.screenLightUpScratch.set(0, 1, 0).applyQuaternion(this.screenLightQuatScratch).normalize();
        this.screenLightNormalScratch.set(0, 0, 1).applyQuaternion(this.screenLightQuatScratch).normalize();
        this.keyboardLightScreenRightUniform.value.copy(this.screenLightRightScratch);
        this.keyboardLightScreenUpUniform.value.copy(this.screenLightUpScratch);
        this.keyboardLightScreenNormalUniform.value.copy(this.screenLightNormalScratch);

        const screenWidth = this.getScreenPhysicalWidth();
        const screenHeight = this.getScreenPhysicalHeight();
        this.keyboardLightScreenSizeUniform.value.set(screenWidth, screenHeight);
    }

    private updateKeyboardScreenLight(_deltaTime: number): void {
        if (!(this.parameters.keyboardEnabled ?? true)) {
            return;
        }

        this.ensureScreenLightDownsample();
        this.updateScreenLightTexture();
        this.updateKeyboardLightBasis();

        const powerFactor = (this.parameters.powerOn ?? true) ? this.powerWarmup : 0.0;
        const lightStrength = this.parameters.keyboardScreenLightIntensity ?? 1.0;
        const intensity = 2.5 * powerFactor * (this.parameters.brightness ?? 1.0) * lightStrength;
        this.keyboardLightIntensityUniform.value = intensity;
    }

    updateParameters(params: Partial<CRTScreenSceneParameters>): void {
        const prevPowerOn = this.parameters.powerOn ?? true;
        const prevDisplayMode = this.parameters.displayMode ?? 'video';
        const prevTerminalScale = this.parameters.terminalFontScale ?? 1;
        const prevResolution = this.parameters.screenResolution ?? CRTScreenScene.DEFAULT_RESOLUTION_PRESET;
        const prevShaderType = this.parameters.shaderType ?? 'mandelbrot';
        Object.assign(this.parameters, params);

        if (params.displayMode !== undefined && params.displayMode !== prevDisplayMode) {
            this.applyDisplayMode();
        }
        if (params.screenResolution !== undefined && params.screenResolution !== prevResolution) {
            this.setLogicalResolution(this.parameters.screenResolution);
        }
        if (params.terminalFontScale !== undefined && params.terminalFontScale !== prevTerminalScale) {
            this.resetTerminalBuffer();
        }
        if (params.shaderPanX !== undefined) {
            this.parameters.mandelbrotPanX = params.shaderPanX;
        }
        if (params.shaderPanY !== undefined) {
            this.parameters.mandelbrotPanY = params.shaderPanY;
        }
        if (params.shaderZoom !== undefined) {
            this.parameters.mandelbrotZoom = params.shaderZoom;
        }
        const isJulia = (this.parameters.shaderType ?? 'mandelbrot') === 'julia';
        if (params.shaderType !== undefined && params.shaderType !== prevShaderType) {
            if (this.shaderTypeUniform) {
                this.shaderTypeUniform.value = params.shaderType === 'julia' ? 1.0 : 0.0;
            }
        }
        if (
            params.shaderType !== undefined ||
            params.mandelbrotPanX !== undefined ||
            params.mandelbrotPanY !== undefined ||
            params.mandelbrotZoom !== undefined ||
            params.juliaPanX !== undefined ||
            params.juliaPanY !== undefined ||
            params.juliaZoom !== undefined ||
            params.juliaCReal !== undefined ||
            params.juliaCImag !== undefined ||
            params.shaderPanX !== undefined ||
            params.shaderPanY !== undefined ||
            params.shaderZoom !== undefined
        ) {
            const active = this.getActiveShaderParams();
            if (this.shaderPanXUniform) {
                this.shaderPanXUniform.value = active.panX;
            }
            if (this.shaderPanYUniform) {
                this.shaderPanYUniform.value = active.panY;
            }
            if (this.shaderZoomUniform) {
                this.shaderZoomUniform.value = Math.max(0.0001, active.zoom);
            }
        } else if (isJulia) {
            if (this.shaderPanXUniform && params.juliaPanX !== undefined) {
                this.shaderPanXUniform.value = params.juliaPanX;
            }
            if (this.shaderPanYUniform && params.juliaPanY !== undefined) {
                this.shaderPanYUniform.value = params.juliaPanY;
            }
            if (this.shaderZoomUniform && params.juliaZoom !== undefined) {
                this.shaderZoomUniform.value = Math.max(0.0001, params.juliaZoom);
            }
        } else {
            if (this.shaderPanXUniform && params.mandelbrotPanX !== undefined) {
                this.shaderPanXUniform.value = params.mandelbrotPanX;
            }
            if (this.shaderPanYUniform && params.mandelbrotPanY !== undefined) {
                this.shaderPanYUniform.value = params.mandelbrotPanY;
            }
            if (this.shaderZoomUniform && params.mandelbrotZoom !== undefined) {
                this.shaderZoomUniform.value = Math.max(0.0001, params.mandelbrotZoom);
            }
        }
        if (this.shaderJuliaCUniform) {
            const cReal = params.juliaCReal ?? this.parameters.juliaCReal;
            const cImag = params.juliaCImag ?? this.parameters.juliaCImag;
            if (cReal !== undefined && cImag !== undefined) {
                this.shaderJuliaCUniform.value.set(cReal, cImag);
            }
        }
        if (params.powerOn !== undefined) {
            if (params.powerOn !== prevPowerOn) {
                this.startPowerTransition(params.powerOn);
            } else if (this.powerOnUniform) {
                this.powerOnUniform.value = params.powerOn ? 1.0 : 0.0;
            }
        }
        if (params.powerWarmupDuration !== undefined) {
            this.updateWarmupUniform();
        }
        if (params.powerOffDuration !== undefined || params.powerOffEndDuration !== undefined) {
            this.updatePowerCollapseRatio();
        }
        const keyboardEnabledChanged = params.keyboardEnabled !== undefined;
        const keyboardGeometryChanged = (
            params.keyboardWidthScale !== undefined ||
            params.keyboardDepthRatio !== undefined ||
            params.keyboardTopInset !== undefined ||
            params.keyboardCornerRadiusBottom !== undefined ||
            params.keyboardCornerRadiusTop !== undefined ||
            params.keyboardTopEdgeRadius !== undefined
        );
        const keyboardMaterialChanged = (
            params.keyboardBumpScale !== undefined ||
            params.keyboardBumpStrength !== undefined ||
            params.keyboardBaseColor !== undefined ||
            params.keyboardRoughnessMin !== undefined ||
            params.keyboardRoughnessMax !== undefined ||
            params.keyboardMetalness !== undefined ||
            params.keyboardLabelOpacity !== undefined
        );
        const keyboardPlacementChanged = (
            params.keyboardOffsetY !== undefined ||
            params.keyboardOffsetZ !== undefined
        );
        if (keyboardEnabledChanged || keyboardGeometryChanged) {
            this.refreshKeyboardLayout();
        }
        if (keyboardMaterialChanged) {
            this.updateKeyboardMaterialParams();
        }
        if (keyboardPlacementChanged) {
            this.updateKeyboardPlacement();
        }
        if (params.screenWidth !== undefined || params.screenHeight !== undefined) {
            this.createCRTScreen();
        }

        if (this.keyboardLightWrapUniform && params.keyboardLightWrap !== undefined) {
            this.keyboardLightWrapUniform.value = params.keyboardLightWrap;
        }
        if (this.keyboardLightSaturationUniform && params.keyboardLightSaturation !== undefined) {
            this.keyboardLightSaturationUniform.value = params.keyboardLightSaturation;
        }
        if (params.keyboardLightSampleGrid !== undefined) {
            this.screenLightSampleGrid = this.getKeyboardLightSampleGrid();
            this.screenLightComputeNode = null;
        }
        if (params.terminalFontColor !== undefined) {
            this.terminalCurrentColor = this.parseTerminalColor(params.terminalFontColor);
            this.terminalDirty = true;
        }

        if (this.minBrightnessUniform && params.minBrightness !== undefined) {
            this.minBrightnessUniform.value = params.minBrightness;
        }
        if (this.brightnessUniform && params.brightness !== undefined) {
            this.brightnessUniform.value = params.brightness;
        }
        if (this.staticSpeedUniform && params.staticSpeed !== undefined) {
            this.staticSpeedUniform.value = params.staticSpeed;
        }
        if (this.staticContrastUniform && params.staticContrast !== undefined) {
            this.staticContrastUniform.value = params.staticContrast;
        }
        if (this.colorAttackUniform && params.colorAttack !== undefined) {
            this.colorAttackUniform.value = params.colorAttack;
        }
        if (this.colorDecayUniform && params.colorDecay !== undefined) {
            this.colorDecayUniform.value = params.colorDecay;
        }
        if (this.powerFlashUniform && params.powerFlash !== undefined) {
            this.powerFlashUniform.value = params.powerFlash;
        }
        
        // Phosphor uniforms
        if (this.slotDutyXUniform && params.slotDutyX !== undefined) {
            this.slotDutyXUniform.value = params.slotDutyX;
        }
        if (this.slotDutyYUniform && params.slotDutyY !== undefined) {
            this.slotDutyYUniform.value = params.slotDutyY;
        }
        if (this.subpixelFeatherUniform && params.subpixelFeather !== undefined) {
            this.subpixelFeatherUniform.value = params.subpixelFeather;
        }
        if (this.phosphorTintUniform && params.phosphorTint !== undefined) {
            this.phosphorTintUniform.value = params.phosphorTint;
        }
        if (this.moireStrengthUniform && params.moireStrength !== undefined) {
            this.moireStrengthUniform.value = params.moireStrength;
        }
        if (this.moireChromaUniform && params.moireChroma !== undefined) {
            this.moireChromaUniform.value = params.moireChroma;
        }
        if (this.moireFeatherUniform && params.moireFeather !== undefined) {
            this.moireFeatherUniform.value = params.moireFeather;
        }
        if (this.moireThresholdUniform && params.moireThreshold !== undefined) {
            this.moireThresholdUniform.value = params.moireThreshold;
        }
        
        // Beam physics uniforms
        if (this.beamGammaUniform && params.beamGamma !== undefined) {
            this.beamGammaUniform.value = params.beamGamma;
        }
        if (this.beamSpreadUniform && params.beamSpread !== undefined) {
            this.beamSpreadUniform.value = params.beamSpread;
        }
        if (this.vignetteStrengthUniform && params.vignetteStrength !== undefined) {
            this.vignetteStrengthUniform.value = params.vignetteStrength;
        }
        if (this.phaseShearAmountUniform && params.phaseShearAmount !== undefined) {
            this.phaseShearAmountUniform.value = params.phaseShearAmount;
        }
        
        // CRT uniforms
        if (this.crtAmountUniform && params.crtAmount !== undefined) {
            this.crtAmountUniform.value = params.crtAmount;
        }
        if (this.crtBarrelUniform && params.crtBarrel !== undefined) {
            this.crtBarrelUniform.value = params.crtBarrel;
        }
        if (this.crtKeystoneXUniform && params.crtKeystoneX !== undefined) {
            this.crtKeystoneXUniform.value = params.crtKeystoneX;
        }
        if (this.crtKeystoneYUniform && params.crtKeystoneY !== undefined) {
            this.crtKeystoneYUniform.value = params.crtKeystoneY;
        }
        if (this.crtZoomUniform && params.crtZoom !== undefined) {
            this.crtZoomUniform.value = params.crtZoom;
        }
        if (this.screenCurvatureUniform && params.screenCurvature !== undefined) {
            this.screenCurvatureUniform.value = Math.max(0, params.screenCurvature);
        }
        
        // Bloom parameters
        if (this.bloomNode) {
            if (params.bloomStrength !== undefined) {
                this.bloomNode.strength.value = params.bloomStrength;
            }
            if (params.bloomRadius !== undefined) {
                this.bloomNode.radius.value = params.bloomRadius;
            }
            if (params.bloomThreshold !== undefined) {
                this.bloomNode.threshold.value = params.bloomThreshold;
            }
        }
        if (this.screenLightBloomNode) {
            if (params.bloomStrength !== undefined) {
                this.screenLightBloomNode.strength.value = params.bloomStrength;
            }
            if (params.bloomRadius !== undefined) {
                this.screenLightBloomNode.radius.value = params.bloomRadius;
            }
            if (params.bloomThreshold !== undefined) {
                this.screenLightBloomNode.threshold.value = params.bloomThreshold;
            }
        }
        
        // Beam scan parameters
        if (this.scanFramerateUniform && params.scanFramerate !== undefined) {
            this.scanFramerateUniform.value = params.scanFramerate;
        }
        if (this.beamPixelDurationUniform && params.beamPixelDuration !== undefined) {
            this.beamPixelDurationUniform.value = params.beamPixelDuration;
        }
    }

    setVideoElement(videoElement: HTMLVideoElement | null): void {
        this.setContentSource(videoElement);
    }

    private applyDisplayMode(): void {
        const mode = this.parameters.displayMode ?? 'video';
        if (mode === 'static') {
            this.useExternalContent = false;
            if (this.useExternalContentUniform) {
                this.useExternalContentUniform.value = 0.0;
            }
            if (this.useExternalTextureUniform) {
                this.useExternalTextureUniform.value = 0.0;
            }
            if (this.shaderEnabledUniform) {
                this.shaderEnabledUniform.value = 0.0;
            }
            this.clearVideoFrameCallback();
            return;
        }

        if (mode === 'terminal') {
            this.ensureContentCanvas();
            this.ensureTerminalBuffer();
            this.ensureTerminalFont();
            this.useExternalContent = true;
            if (this.useExternalContentUniform) {
                this.useExternalContentUniform.value = 1.0;
            }
            if (this.useExternalTextureUniform) {
                this.useExternalTextureUniform.value = 1.0;
            }
            if (this.shaderEnabledUniform) {
                this.shaderEnabledUniform.value = 0.0;
            }
            this.terminalDirty = true;
            this.terminalCursorVisible = true;
            this.terminalCursorBlinkTime = 0;
            this.clearVideoFrameCallback();
            return;
        }

        if (mode === 'shader') {
            this.ensureContentCanvas();
            this.useExternalContent = true;
            if (this.useExternalContentUniform) {
                this.useExternalContentUniform.value = 1.0;
            }
            if (this.useExternalTextureUniform) {
                this.useExternalTextureUniform.value = 0.0;
            }
            if (this.shaderEnabledUniform) {
                this.shaderEnabledUniform.value = 1.0;
            }
            this.clearVideoFrameCallback();
            return;
        }

        if (mode === 'xterm') {
            this.ensureContentCanvas();
            this.useExternalContent = true;
            if (this.useExternalContentUniform) {
                this.useExternalContentUniform.value = 1.0;
            }
            if (this.useExternalTextureUniform) {
                this.useExternalTextureUniform.value = 1.0;
            }
            if (this.shaderEnabledUniform) {
                this.shaderEnabledUniform.value = 0.0;
            }
            if (!this.contentSource && this.contentContext && this.contentCanvas) {
                this.contentContext.fillStyle = '#000000';
                this.contentContext.fillRect(0, 0, this.contentCanvas.width, this.contentCanvas.height);
                if (this.contentTexture) {
                    this.contentTexture.needsUpdate = true;
                }
            }
            this.contentDirty = true;
            this.clearVideoFrameCallback();
            return;
        }

        const useExternal = !!this.contentSource;
        this.useExternalContent = useExternal;
        if (this.useExternalContentUniform) {
            this.useExternalContentUniform.value = useExternal ? 1.0 : 0.0;
        }
        if (this.useExternalTextureUniform) {
            this.useExternalTextureUniform.value = useExternal ? 1.0 : 0.0;
        }
        if (this.shaderEnabledUniform) {
            this.shaderEnabledUniform.value = 0.0;
        }
        if (useExternal) {
            this.contentDirty = true;
        }
        if (this.contentIsVideo && this.contentSource) {
            this.setupVideoFrameCallback(this.contentSource as HTMLVideoElement);
        }
    }

    setContentSource(source: CanvasImageSource | null): void {
        if (source === this.contentSource) {
            return;
        }

        this.contentSource = source;
        this.contentIsVideo = !!source && source instanceof HTMLVideoElement;
        this.contentDirty = true;
        this.targetColorsNeedUpdate = false;

        this.clearVideoFrameCallback();

        if (!source) {
            this.applyDisplayMode();
            return;
        }

        this.ensureContentCanvas();

        if (!this.contentIsVideo) {
            this.refreshContentFromSource();
        } else {
            this.setupVideoFrameCallback(source as HTMLVideoElement);
        }

        this.applyDisplayMode();
    }

    setPixelBuffer(buffer: Float32Array): void {
        if (!this.targetColorArray) return;
        if (buffer.length !== this.targetColorArray.length) {
            console.warn('Invalid buffer size for CRT content');
            return;
        }

        this.targetColorArray.set(buffer);
        this.targetColorsNeedUpdate = true;
        this.contentSource = null;
        this.contentIsVideo = false;
        this.contentDirty = false;
        this.useExternalContent = true;
        if (this.useExternalContentUniform) {
            this.useExternalContentUniform.value = 1.0;
        }
        if (this.useExternalTextureUniform) {
            this.useExternalTextureUniform.value = 0.0;
        }
    }

    disableExternalContent(): void {
        this.contentSource = null;
        this.contentIsVideo = false;
        this.contentDirty = false;
        this.useExternalContent = false;
        if (this.useExternalContentUniform) {
            this.useExternalContentUniform.value = 0.0;
        }
        if (this.useExternalTextureUniform) {
            this.useExternalTextureUniform.value = 0.0;
        }
        this.clearVideoFrameCallback();
    }

    private ensureContentCanvas(): void {
        const width = this.logicalWidth;
        const height = this.logicalHeight;
        const hadCanvas = !!this.contentCanvas;

        if (!this.contentCanvas) {
            this.contentCanvas = document.createElement('canvas');
        }

        if (this.contentCanvas.width !== width || this.contentCanvas.height !== height) {
            this.contentCanvas.width = width;
            this.contentCanvas.height = height;
        }

        if (!this.contentContext || !hadCanvas) {
            this.contentContext = this.contentCanvas.getContext('2d');
        }

        if (!this.contentTexture) {
            this.contentTexture = new THREE.CanvasTexture(this.contentCanvas);
            this.contentTexture.colorSpace = THREE.SRGBColorSpace;
            this.contentTexture.minFilter = THREE.LinearFilter;
            this.contentTexture.magFilter = THREE.LinearFilter;
            this.contentTexture.wrapS = THREE.ClampToEdgeWrapping;
            this.contentTexture.wrapT = THREE.ClampToEdgeWrapping;
            this.contentTexture.generateMipmaps = false;
            this.contentTexture.needsUpdate = true;
        } else {
            this.contentTexture.image = this.contentCanvas;
            this.contentTexture.needsUpdate = true;
        }
    }

    private updateContentFrame(): void {
        const mode = this.parameters.displayMode ?? 'video';
        if (mode === 'terminal') {
            this.ensureContentCanvas();
            this.ensureTerminalBuffer();
            this.ensureTerminalFont();
            if (this.terminalDirty) {
                this.renderTerminal();
            }
            return;
        }

        if (mode === 'shader') {
            this.ensureContentCanvas();
            this.renderShaderOverlay();
            return;
        }

        const isVideoMode = mode === 'video' || mode === 'emulator';
        const isXtermMode = mode === 'xterm';
        if ((!isVideoMode && !isXtermMode) || !this.useExternalContent || !this.contentSource) {
            return;
        }

        this.ensureContentCanvas();
        if (!this.contentContext || !this.contentCanvas) {
            return;
        }

        if (this.contentIsVideo) {
            const video = this.contentSource as HTMLVideoElement;
            if (video.readyState < 2) {
                return;
            }
            if (!this.contentDirty) {
                return;
            }
        } else {
            if (isXtermMode) {
                this.contentDirty = true;
            }
            if (!this.contentDirty) {
                return;
            }
        }

        this.refreshContentFromSource();
    }


    private getTerminalScale(): number {
        const rawScale = this.parameters.terminalFontScale ?? 1;
        const safeScale = Math.max(1, Math.min(10, Math.floor(rawScale)));
        return safeScale;
    }

    private getTerminalCharWidth(): number {
        return this.terminalCharWidth * this.getTerminalScale();
    }

    private getTerminalCharHeight(): number {
        return this.terminalCharHeight * this.getTerminalScale();
    }

    private getTerminalContentCols(): number {
        return Math.max(1, this.terminalCols - (this.terminalPaddingChars * 2));
    }

    private getTerminalContentRows(): number {
        return Math.max(1, this.terminalRows - (this.terminalPaddingChars * 2));
    }

    private resetTerminalBuffer(): void {
        this.terminalBuffer = null;
        this.terminalColorBuffer = null;
        this.terminalDirty = true;
        this.terminalCursorX = 0;
        this.terminalCursorY = 0;
        if ((this.parameters.displayMode ?? 'video') === 'terminal') {
            this.ensureTerminalBuffer();
        }
    }

    private ensureTerminalBuffer(): void {
        const charWidth = this.getTerminalCharWidth();
        const charHeight = this.getTerminalCharHeight();
        const cols = Math.max(1, Math.floor(this.logicalWidth / charWidth));
        const rows = Math.max(1, Math.floor(this.logicalHeight / charHeight));

        if (this.terminalBuffer && this.terminalCols === cols && this.terminalRows === rows) {
            if (!this.terminalColorBuffer || this.terminalColorBuffer.length !== this.terminalBuffer.length) {
                this.terminalCurrentColor = this.parseTerminalColor(this.parameters.terminalFontColor);
                this.terminalColorBuffer = new Uint32Array(this.terminalBuffer.length);
                this.terminalColorBuffer.fill(this.terminalCurrentColor);
                this.terminalDirty = true;
            }
            return;
        }

        this.terminalCols = cols;
        this.terminalRows = rows;
        this.terminalCursorX = 0;
        this.terminalCursorY = 0;
        this.terminalBuffer = new Uint16Array(cols * rows);
        this.terminalBuffer.fill(32);
        this.terminalCurrentColor = this.parseTerminalColor(this.parameters.terminalFontColor);
        this.terminalColorBuffer = new Uint32Array(cols * rows);
        this.terminalColorBuffer.fill(this.terminalCurrentColor);
        this.terminalDirty = true;
    }

    private ensureTerminalFont(): void {
        if (this.terminalFontLoaded || this.terminalFontLoading) {
            return;
        }
        if (typeof FontFace === 'undefined' || !document?.fonts) {
            return;
        }

        this.terminalFontLoading = true;
        const font = new FontFace(
            this.terminalFontFamily,
            'url(fonts/oldschool_pc/Web437_IBM_VGA_8x16.woff)'
        );

        font.load()
            .then(loaded => {
                document.fonts.add(loaded);
                this.terminalFontLoaded = true;
                this.terminalFontLoading = false;
                this.terminalDirty = true;
                if (this.keyboardLabelTexture) {
                    this.keyboardLabelTexture.dispose();
                    this.keyboardLabelTexture = null;
                }
                this.refreshKeyboardLayout();
            })
            .catch(() => {
                this.terminalFontLoading = false;
            });
    }

    private ensureShaderOverlayFont(): void {
        if (this.shaderOverlayFontLoaded || this.shaderOverlayFontLoading) {
            return;
        }
        if (typeof FontFace === 'undefined' || !document?.fonts) {
            return;
        }

        this.shaderOverlayFontLoading = true;
        const font = new FontFace(
            this.shaderOverlayFontFamily,
            'url(fonts/oldschool_pc/Web437_CompaqThin_8x16.woff)'
        );

        font.load()
            .then(loaded => {
                document.fonts.add(loaded);
                this.shaderOverlayFontLoaded = true;
                this.shaderOverlayFontLoading = false;
                this.contentDirty = true;
            })
            .catch(() => {
                this.shaderOverlayFontLoading = false;
            });
    }

    private renderTerminal(): void {
        if (!this.contentCanvas || !this.contentContext || !this.contentTexture || !this.terminalBuffer) {
            return;
        }

        const charWidth = this.getTerminalCharWidth();
        const charHeight = this.getTerminalCharHeight();
        const terminalColor = this.parameters.terminalFontColor ?? this.terminalForeground;

        this.contentContext.fillStyle = this.terminalBackground;
        this.contentContext.fillRect(0, 0, this.contentCanvas.width, this.contentCanvas.height);
        this.contentContext.textBaseline = 'top';
        this.contentContext.textAlign = 'left';
        this.contentContext.imageSmoothingEnabled = false;
        this.contentContext.font = `${charHeight}px '${this.terminalFontFamily}', monospace`;

        const pad = this.terminalPaddingChars;
        const contentCols = this.getTerminalContentCols();
        const contentRows = this.getTerminalContentRows();
        const cursorX = this.terminalCursorVisible ? Math.min(this.terminalCursorX, contentCols - 1) : -1;
        const cursorY = this.terminalCursorVisible ? Math.min(this.terminalCursorY, contentRows - 1) : -1;
        const cursorIndex = cursorX >= 0 && cursorY >= 0
            ? (cursorY + pad) * this.terminalCols + (cursorX + pad)
            : -1;

        for (let row = 0; row < contentRows; row++) {
            const start = (row + pad) * this.terminalCols + pad;
            const y = (row + pad) * charHeight;
            for (let col = 0; col < contentCols; col++) {
                const idx = start + col;
                if (idx === cursorIndex) {
                    continue;
                }
                const code = this.terminalBuffer[idx] ?? 32;
                if (code === 32) {
                    continue;
                }
                const glyph = CP437_TO_UNICODE_STR[code] ?? ' ';
                if (glyph === ' ') {
                    continue;
                }
                const colorValue = this.terminalColorBuffer
                    ? this.terminalColorBuffer[idx]
                    : this.terminalCurrentColor;
                this.contentContext.fillStyle = this.getTerminalColorCss(colorValue);
                const x = (col + pad) * charWidth;
                this.contentContext.fillText(glyph, x, y);
            }
        }

        if (this.terminalCursorVisible) {
            const cursorIdx = cursorIndex;
            if (cursorIdx >= 0) {
                const code = this.terminalBuffer[cursorIdx] ?? 32;
                const glyph = CP437_TO_UNICODE_STR[code] ?? ' ';
                const x = (cursorX + pad) * charWidth;
                const y = (cursorY + pad) * charHeight;
                this.contentContext.fillStyle = this.getTerminalColorCss(this.terminalCurrentColor) ?? terminalColor;
                this.contentContext.fillRect(x, y, charWidth, charHeight);
                this.contentContext.fillStyle = this.terminalBackground;
                this.contentContext.fillText(glyph, x, y);
            }
        }

        this.contentTexture.needsUpdate = true;
        this.terminalDirty = false;
    }

    private updateTerminalBlink(deltaTime: number): void {
        this.terminalCursorBlinkTime += deltaTime;
        if (this.terminalCursorBlinkTime >= this.terminalCursorBlinkInterval) {
            this.terminalCursorBlinkTime -= this.terminalCursorBlinkInterval;
            this.terminalCursorVisible = !this.terminalCursorVisible;
            this.terminalDirty = true;
        }
    }

    private parseTerminalColor(value?: string): number {
        const fallback = this.terminalForeground;
        const color = this.terminalColorScratch;
        try {
            color.set(value ?? fallback);
        } catch {
            color.set(fallback);
        }
        const r = Math.max(0, Math.min(255, Math.round(color.r * 255)));
        const g = Math.max(0, Math.min(255, Math.round(color.g * 255)));
        const b = Math.max(0, Math.min(255, Math.round(color.b * 255)));
        return (r << 16) | (g << 8) | b;
    }

    private getTerminalColorCss(value: number): string {
        const cached = this.terminalColorCache.get(value);
        if (cached) {
            return cached;
        }
        const hex = value.toString(16).padStart(6, '0');
        const css = `#${hex}`;
        this.terminalColorCache.set(value, css);
        return css;
    }

    private writeTerminalChar(charCode: number): void {
        if (!this.terminalBuffer) {
            return;
        }
        if (!this.terminalColorBuffer || this.terminalColorBuffer.length !== this.terminalBuffer.length) {
            this.terminalCurrentColor = this.parseTerminalColor(this.parameters.terminalFontColor);
            this.terminalColorBuffer = new Uint32Array(this.terminalBuffer.length);
            this.terminalColorBuffer.fill(this.terminalCurrentColor);
        }

        const pad = this.terminalPaddingChars;
        const contentCols = this.getTerminalContentCols();
        const contentRows = this.getTerminalContentRows();
        const idx = (this.terminalCursorY + pad) * this.terminalCols + (this.terminalCursorX + pad);
        this.terminalBuffer[idx] = charCode;
        if (this.terminalColorBuffer) {
            this.terminalColorBuffer[idx] = this.terminalCurrentColor;
        }
        this.terminalCursorX += 1;

        if (this.terminalCursorX >= contentCols) {
            this.terminalCursorX = 0;
            this.terminalCursorY += 1;
        }

        if (this.terminalCursorY >= contentRows) {
            this.scrollTerminal();
        }
        this.terminalDirty = true;
        this.terminalCursorVisible = true;
        this.terminalCursorBlinkTime = 0;
    }

    private newlineTerminal(): void {
        this.terminalCursorX = 0;
        this.terminalCursorY += 1;
        if (this.terminalCursorY >= this.getTerminalContentRows()) {
            this.scrollTerminal();
        }
        this.terminalDirty = true;
        this.terminalCursorVisible = true;
        this.terminalCursorBlinkTime = 0;
    }

    private scrollTerminal(): void {
        if (!this.terminalBuffer) {
            return;
        }

        const pad = this.terminalPaddingChars;
        const contentCols = this.getTerminalContentCols();
        const contentRows = this.getTerminalContentRows();
        if (!this.terminalColorBuffer) {
            this.terminalColorBuffer = new Uint32Array(this.terminalBuffer.length);
            this.terminalColorBuffer.fill(this.terminalCurrentColor);
        }

        for (let row = 0; row < contentRows - 1; row++) {
            const srcStart = (row + 1 + pad) * this.terminalCols + pad;
            const dstStart = (row + pad) * this.terminalCols + pad;
            this.terminalBuffer.copyWithin(dstStart, srcStart, srcStart + contentCols);
            this.terminalColorBuffer.copyWithin(dstStart, srcStart, srcStart + contentCols);
        }

        const lastRowStart = (contentRows - 1 + pad) * this.terminalCols + pad;
        this.terminalBuffer.fill(32, lastRowStart, lastRowStart + contentCols);
        this.terminalColorBuffer.fill(this.terminalCurrentColor, lastRowStart, lastRowStart + contentCols);
        this.terminalCursorY = contentRows - 1;
    }

    private backspaceTerminal(): void {
        if (!this.terminalBuffer) {
            return;
        }

        const contentCols = this.getTerminalContentCols();
        if (this.terminalCursorX > 0) {
            this.terminalCursorX -= 1;
        } else if (this.terminalCursorY > 0) {
            this.terminalCursorY -= 1;
            this.terminalCursorX = contentCols - 1;
        } else {
            return;
        }

        const pad = this.terminalPaddingChars;
        const idx = (this.terminalCursorY + pad) * this.terminalCols + (this.terminalCursorX + pad);
        this.terminalBuffer[idx] = 32;
        if (this.terminalColorBuffer) {
            this.terminalColorBuffer[idx] = this.terminalCurrentColor;
        }
        this.terminalDirty = true;
        this.terminalCursorVisible = true;
        this.terminalCursorBlinkTime = 0;
    }

    handleTerminalKey(event: KeyboardEvent): void {
        if ((this.parameters.displayMode ?? 'video') !== 'terminal') {
            return;
        }

        this.ensureTerminalBuffer();
        this.ensureShaderOverlayFont();

        if (event.key === 'Backspace') {
            event.preventDefault();
            this.backspaceTerminal();
            return;
        }
        if (event.key === 'Enter') {
            event.preventDefault();
            this.newlineTerminal();
            return;
        }
        if (event.key === 'Tab') {
            event.preventDefault();
            for (let i = 0; i < 4; i++) {
                this.writeTerminalChar(32);
            }
            return;
        }
        if (event.key.length === 1) {
            const codePoint = event.key.codePointAt(0);
            if (codePoint !== undefined) {
                const mapped = UNICODE_TO_CP437.get(codePoint);
                if (mapped !== undefined) {
                    event.preventDefault();
                    this.writeTerminalChar(mapped);
                }
            }
        }
    }

    private refreshContentFromSource(): void {
        if (!this.contentSource || !this.contentContext || !this.contentCanvas || !this.contentTexture) {
            return;
        }

        const sourceSize = this.getContentSourceSize(this.contentSource);
        if (!sourceSize || sourceSize.width <= 0 || sourceSize.height <= 0) {
            return;
        }

        const destWidth = this.contentCanvas.width;
        const destHeight = this.contentCanvas.height;

        this.contentContext.drawImage(this.contentSource, 0, 0, destWidth, destHeight);

        this.contentTexture.needsUpdate = true;
        this.contentDirty = false;
    }

    private setupVideoFrameCallback(video: HTMLVideoElement): void {
        this.videoFrameSource = video;
        if ('requestVideoFrameCallback' in video) {
            const onFrame = () => {
                this.contentDirty = true;
                if (this.videoFrameSource === video) {
                    this.videoFrameHandle = video.requestVideoFrameCallback(onFrame);
                }
            };
            this.videoFrameHandle = video.requestVideoFrameCallback(onFrame);
        }
    }

    private clearVideoFrameCallback(): void {
        if (this.videoFrameSource && this.videoFrameHandle !== null) {
            if ('cancelVideoFrameCallback' in this.videoFrameSource) {
                this.videoFrameSource.cancelVideoFrameCallback(this.videoFrameHandle);
            }
        }
        this.videoFrameHandle = null;
        this.videoFrameSource = null;
    }

    private getContentSourceSize(source: CanvasImageSource): { width: number; height: number } | null {
        if (source instanceof HTMLVideoElement) {
            return { width: source.videoWidth, height: source.videoHeight };
        }
        if (source instanceof HTMLImageElement) {
            return { width: source.naturalWidth || source.width, height: source.naturalHeight || source.height };
        }
        if (source instanceof HTMLCanvasElement) {
            return { width: source.width, height: source.height };
        }
        if (typeof OffscreenCanvas !== 'undefined' && source instanceof OffscreenCanvas) {
            return { width: source.width, height: source.height };
        }
        if (typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap) {
            return { width: source.width, height: source.height };
        }
        return null;
    }

    private startPowerTransition(nextOn: boolean): void {
        if (!this.powerOnUniform || !this.powerTransitionUniform || !this.powerDirectionUniform) {
            return;
        }

        const duration = nextOn
            ? (this.parameters.powerOnDuration ?? 0.8)
            : this.getPowerOffTotalDuration();
        const clampedDuration = Math.max(0.0, duration);

        this.powerOnUniform.value = nextOn ? 1.0 : 0.0;
        if (nextOn) {
            const warmupDuration = this.parameters.powerWarmupDuration ?? 1.6;
            this.powerWarmup = warmupDuration <= 0.0 ? 1.0 : 0.0;
            this.updateWarmupUniform();
        }

        if (clampedDuration <= 0.0) {
            this.powerTransition = 1.0;
            this.powerDirection = 0.0;
            this.powerTransitionUniform.value = 1.0;
            this.powerDirectionUniform.value = 0.0;
            return;
        }

        this.powerTransition = 0.0;
        this.powerDirection = nextOn ? 1.0 : -1.0;
        this.powerTransitionUniform.value = 0.0;
        this.powerDirectionUniform.value = this.powerDirection;
        if (!nextOn) {
            this.updatePowerCollapseRatio();
        }
    }

    private updatePowerTransition(deltaTime: number): void {
        if (!this.powerTransitionUniform || !this.powerDirectionUniform) {
            return;
        }

        if (this.powerDirection === 0) {
            return;
        }

        const duration = this.powerDirection > 0
            ? (this.parameters.powerOnDuration ?? 0.8)
            : this.getPowerOffTotalDuration();
        const safeDuration = Math.max(0.001, duration);

        this.powerTransition = Math.min(this.powerTransition + (deltaTime / safeDuration), 1.0);
        this.powerTransitionUniform.value = this.powerTransition;

        if (this.powerTransition >= 1.0) {
            this.powerDirection = 0.0;
            this.powerDirectionUniform.value = 0.0;
        }
    }

    private getPowerOffTotalDuration(): number {
        const offDuration = Math.max(0.0, this.parameters.powerOffDuration ?? 0.45);
        const endDuration = Math.max(0.0, this.parameters.powerOffEndDuration ?? 0.0);
        return offDuration + endDuration;
    }

    private updatePowerCollapseRatio(): void {
        if (!this.powerCollapseRatioUniform) {
            return;
        }

        const offDuration = Math.max(0.0, this.parameters.powerOffDuration ?? 0.45);
        const endDuration = Math.max(0.0, this.parameters.powerOffEndDuration ?? 0.0);
        const total = offDuration + endDuration;
        const ratio = total > 0.0 ? offDuration / total : 1.0;
        this.powerCollapseRatioUniform.value = ratio;
    }

    private updatePowerWarmup(deltaTime: number): void {
        if (!this.powerWarmupUniform) {
            return;
        }

        if (!(this.parameters.powerOn ?? true)) {
            return;
        }

        if (this.powerWarmup >= 1.0) {
            return;
        }

        const duration = this.parameters.powerWarmupDuration ?? 1.6;
        if (duration <= 0.0) {
            this.powerWarmup = 1.0;
            this.updateWarmupUniform();
            return;
        }

        this.powerWarmup = Math.min(this.powerWarmup + (deltaTime / duration), 1.0);
        this.updateWarmupUniform();
    }

    private updateWarmupUniform(): void {
        if (!this.powerWarmupUniform) {
            return;
        }

        this.powerWarmupUniform.value = this.powerWarmup;
    }

    private addKeyboardLights(): void {
        if (!this.scene) {
            return;
        }

        const screenPosition = new THREE.Vector3(0, 0, 0);
        if (this.screenMesh) {
            this.screenMesh.getWorldPosition(screenPosition);
        }
        this.keyboardLightOriginUniform.value.copy(screenPosition);
        this.keyboardScreenLight = null;
        this.keyboardLights = [];
    }

    private clearKeyboard(): void {
        if (this.keyboardGroup && this.scene) {
            this.scene.remove(this.keyboardGroup);
        }
        if (this.keyboardOuterMesh) {
            (this.keyboardOuterMesh.material as THREE.Material).dispose();
        }
        if (this.keyboardInnerMesh) {
            this.keyboardInnerMesh.geometry.dispose();
            (this.keyboardInnerMesh.material as THREE.Material).dispose();
        }
        if (this.keyboardLights.length > 0 && this.scene) {
            this.keyboardLights.forEach(light => this.scene?.remove(light));
        }

        this.keyboardGroup = null;
        this.keyboardOuterMesh = null;
        this.keyboardInnerMesh = null;
        if (this.keyboardKeyMeshList.length > 0) {
            this.keyboardKeyMeshList.forEach(mesh => mesh.geometry.dispose());
        }
        this.keyboardKeyMeshList = [];
        this.keyboardKeyMeshes.clear();
        this.keyboardLights = [];
        this.keyboardScreenLight = null;
        this.keyboardBaseKeyWidth = 0;
        if (this.keyboardLabelTexture) {
            this.keyboardLabelTexture.dispose();
            this.keyboardLabelTexture = null;
        }
    }

    private cancelKeyboardRebuild(): void {
        if (this.keyboardRebuildHandle === null) {
            return;
        }
        if (this.keyboardRebuildUsesIdle) {
            const cancelIdle = (globalThis as any).cancelIdleCallback;
            if (typeof cancelIdle === 'function') {
                cancelIdle(this.keyboardRebuildHandle);
            }
        } else {
            clearTimeout(this.keyboardRebuildHandle);
        }
        this.keyboardRebuildHandle = null;
        this.keyboardRebuildUsesIdle = false;
    }

    private scheduleKeyboardRebuild(): void {
        this.cancelKeyboardRebuild();
        const rebuild = (): void => {
            this.keyboardRebuildHandle = null;
            this.keyboardRebuildUsesIdle = false;
            if (!(this.parameters.keyboardEnabled ?? true) || !this.keyboardLayout) {
                return;
            }
            this.buildKeyboardLayout(this.keyboardLayout);
        };
        const requestIdle = (globalThis as any).requestIdleCallback;
        if (typeof requestIdle === 'function') {
            this.keyboardRebuildUsesIdle = true;
            this.keyboardRebuildHandle = requestIdle(() => rebuild(), { timeout: 200 });
        } else {
            this.keyboardRebuildHandle = globalThis.setTimeout(rebuild, 0);
        }
    }

    private updateKeyboardPlacement(): void {
        if (!this.keyboardGroup) {
            return;
        }
        const keyboardBaseY = CRTScreenScene.DEFAULT_KEYBOARD_BASE_Y;
        const keyboardBaseZ = CRTScreenScene.DEFAULT_KEYBOARD_BASE_Z;
        const offsetY = this.parameters.keyboardOffsetY ?? 0.0;
        const offsetZ = this.parameters.keyboardOffsetZ ?? 0.0;
        this.keyboardGroup.position.set(0, keyboardBaseY + offsetY, keyboardBaseZ + offsetZ);
    }

    private updateKeyboardMaterialParams(): void {
        if (!this.keyboardOuterMesh) {
            return;
        }
        const baseColorHex = this.parameters.keyboardBaseColor ?? '#DAD8CE';
        if (this.keyboardBaseColorUniform) {
            this.keyboardBaseColorUniform.value.set(baseColorHex);
        }
        const bumpScale = Math.max(0.1, this.parameters.keyboardBumpScale ?? 2.5);
        if (this.keyboardBumpScaleUniform) {
            this.keyboardBumpScaleUniform.value = bumpScale;
        }
        const bumpStrength = Math.max(0, this.parameters.keyboardBumpStrength ?? 0.01);
        if (this.keyboardBumpStrengthUniform) {
            this.keyboardBumpStrengthUniform.value = this.keyboardBaseKeyWidth * bumpStrength;
        }
        const roughnessMin = this.parameters.keyboardRoughnessMin ?? 0.78;
        if (this.keyboardRoughnessMinUniform) {
            this.keyboardRoughnessMinUniform.value = roughnessMin;
        }
        const roughnessMax = this.parameters.keyboardRoughnessMax ?? 0.92;
        if (this.keyboardRoughnessMaxUniform) {
            this.keyboardRoughnessMaxUniform.value = roughnessMax;
        }
        const metalness = this.parameters.keyboardMetalness ?? 0.0;
        if (this.keyboardMetalnessUniform) {
            this.keyboardMetalnessUniform.value = metalness;
        }
        const labelOpacity = this.parameters.keyboardLabelOpacity ?? 1.0;
        if (this.keyboardLabelOpacityUniform) {
            this.keyboardLabelOpacityUniform.value = labelOpacity;
        }
    }

    private refreshKeyboardLayout(): void {
        if (!(this.parameters.keyboardEnabled ?? true)) {
            this.cancelKeyboardRebuild();
            this.clearKeyboard();
            return;
        }

        this.ensureScreenLightDownsample();

        if (this.keyboardLayout) {
            this.scheduleKeyboardRebuild();
            return;
        }

        this.loadKeyboardLayout();
    }

    private applyKeyboardLabelUVs(
        geometry: THREE.BufferGeometry,
        mapping: { centerX: number; centerY: number; scale: number; bounds: { minX: number; minY: number; maxX: number; maxY: number } }
    ): void {
        const positions = geometry.getAttribute('position');
        if (!positions) {
            return;
        }
        const { centerX, centerY, scale, bounds } = mapping;
        const svgWidth = bounds.maxX - bounds.minX;
        const svgHeight = bounds.maxY - bounds.minY;
        if (svgWidth <= 0 || svgHeight <= 0) {
            return;
        }
        const uvs = new Float32Array(positions.count * 2);
        for (let i = 0; i < positions.count; i += 1) {
            const x = positions.getX(i);
            const y = positions.getY(i);
            const svgX = x / scale + centerX;
            const svgY = centerY - y / scale;
            const u = (svgX - bounds.minX) / svgWidth;
            const v = (svgY - bounds.minY) / svgHeight;
            uvs[i * 2] = u;
            uvs[i * 2 + 1] = v;
        }
        geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    }

    private getKeyboardLabelRows(): KeyboardLabel[][] {
        return [
            [
                { center: 'Esc', ids: ['Escape'] },
                { center: 'F1', ids: ['F1'] },
                { center: 'F2', ids: ['F2'] },
                { center: 'F3', ids: ['F3'] },
                { center: 'F4', ids: ['F4'] },
                { center: 'F5', ids: ['F5'] },
                { center: 'F6', ids: ['F6'] },
                { center: 'F7', ids: ['F7'] },
                { center: 'F8', ids: ['F8'] },
                { center: 'F9', ids: ['F9'] },
                { center: 'F10', ids: ['F10'] },
                { center: 'F11', ids: ['F11'] },
                { center: 'F12', ids: ['F12'] },
                { center: 'PrtSc', ids: ['PrintScreen'] },
                { center: 'Scroll Lock', ids: ['ScrollLock'] },
                { center: 'Pause Break', ids: ['Pause'] }
            ],
            [
                { top: '~', bottom: '`' },
                { top: '!', bottom: '1' },
                { top: '@', bottom: '2' },
                { top: '#', bottom: '3' },
                { top: '$', bottom: '4' },
                { top: '%', bottom: '5' },
                { top: '^', bottom: '6' },
                { top: '&', bottom: '7' },
                { top: '*', bottom: '8' },
                { top: '(', bottom: '9' },
                { top: ')', bottom: '0' },
                { top: '_', bottom: '-' },
                { top: '+', bottom: '=' },
                { center: 'Backspace', align: 'left' },
                { center: 'Insert', ids: ['Insert'] },
                { center: 'Home', ids: ['Home'] },
                { center: 'PgUp', ids: ['PageUp'] },
                { center: 'Num Lock', ids: ['NumLock'] },
                { center: '/', ids: ['NumpadDivide'] },
                { center: '*', ids: ['NumpadMultiply'] },
                { center: '-', ids: ['NumpadSubtract'] }
            ],
            [
                { center: 'Tab', align: 'left' },
                { center: 'Q' },
                { center: 'W' },
                { center: 'E' },
                { center: 'R' },
                { center: 'T' },
                { center: 'Y' },
                { center: 'U' },
                { center: 'I' },
                { center: 'O' },
                { center: 'P' },
                { top: '{', bottom: '[' },
                { top: '}', bottom: ']' },
                { top: '|', bottom: '\\\\' },
                { center: 'Delete', ids: ['Delete'] },
                { center: 'End', ids: ['End'] },
                { center: 'PgDn', ids: ['PageDown'] },
                { top: 'Home', bottom: '7', ids: ['Numpad7'] },
                { top: '↑', bottom: '8', ids: ['Numpad8'] },
                { top: 'PgUp', bottom: '9', ids: ['Numpad9'] },
                { center: '+', ids: ['NumpadAdd'] }
            ],
            [
                { center: 'Caps Lock', align: 'left' },
                { center: 'A' },
                { center: 'S' },
                { center: 'D' },
                { center: 'F' },
                { center: 'G' },
                { center: 'H' },
                { center: 'J' },
                { center: 'K' },
                { center: 'L' },
                { top: ':', bottom: ';' },
                { top: '\"', bottom: '\'' },
                { center: 'Enter', align: 'left' },
                { top: '↑', bottom: '4', ids: ['Numpad4'], topRotate: -1.5707963267948966 },
                { center: '5', ids: ['Numpad5'] },
                { top: '↑', bottom: '6', ids: ['Numpad6'], topRotate: 1.5707963267948966 }
            ],
            [
                { center: 'Shift', align: 'left' },
                { center: 'Z' },
                { center: 'X' },
                { center: 'C' },
                { center: 'V' },
                { center: 'B' },
                { center: 'N' },
                { center: 'M' },
                { top: '<', bottom: ',' },
                { top: '>', bottom: '.' },
                { top: '?', bottom: '/' },
                { center: 'Shift', align: 'left' },
                { center: '↑', ids: ['ArrowUp'] },
                { top: 'End', bottom: '1', ids: ['Numpad1'] },
                { top: '↓', bottom: '2', ids: ['Numpad2'] },
                { top: 'PgDn', bottom: '3', ids: ['Numpad3'] },
                { center: 'Enter', align: 'left', ids: ['NumpadEnter'] }
            ],
            [
                { center: 'Ctrl', align: 'left' },
                { center: 'Win', align: 'left' },
                { center: 'Alt', align: 'left' },
                { center: '' },
                { center: 'Alt', align: 'left' },
                { center: 'Win', align: 'left' },
                { center: 'Menu', align: 'left' },
                { center: 'Ctrl', align: 'left' },
                { center: '↑', ids: ['ArrowLeft'], centerRotate: -1.5707963267948966 },
                { center: '↓', ids: ['ArrowDown'] },
                { center: '↑', ids: ['ArrowRight'], centerRotate: 1.5707963267948966 },
                { top: 'Ins', bottom: '0', ids: ['Numpad0'] },
                { top: 'Del', bottom: '.', ids: ['NumpadDecimal'] }
            ]
        ];
    }

    private groupKeyboardRows(keys: Array<{
        outer: { x: number; y: number; width: number; height: number; rx: number };
        inner: { x: number; y: number; width: number; height: number; rx: number };
    }>): Array<typeof keys> {
        const rows: Array<typeof keys> = [];
        const sorted = [...keys].sort((a, b) => {
            if (a.outer.y === b.outer.y) {
                return a.outer.x - b.outer.x;
            }
            return a.outer.y - b.outer.y;
        });
        const rowTolerance = 10;
        let currentRowY: number | null = null;
        let currentRow: typeof keys = [];
        sorted.forEach(key => {
            const y = key.outer.y;
            if (currentRowY === null || Math.abs(y - currentRowY) > rowTolerance) {
                if (currentRow.length > 0) {
                    rows.push(currentRow);
                }
                currentRow = [key];
                currentRowY = y;
            } else {
                currentRow.push(key);
                currentRowY = (currentRowY * (currentRow.length - 1) + y) / currentRow.length;
            }
        });
        if (currentRow.length > 0) {
            rows.push(currentRow);
        }
        rows.forEach(row => row.sort((a, b) => a.outer.x - b.outer.x));
        return rows;
    }

    private getKeyboardIdsFromLabel(label: KeyboardLabel | undefined, rowIndex: number, colIndex: number): string[] {
        if (!label) {
            return [];
        }
        const ids: string[] = [];
        if (label.ids && label.ids.length > 0) {
            ids.push(...label.ids);
            if (label.suppressDefaultIds !== false) {
                return Array.from(new Set(ids));
            }
        }
        if (label.top) {
            ids.push(label.top);
        }
        if (label.bottom) {
            ids.push(label.bottom);
        }
        if (label.center !== undefined) {
            const name = label.center.trim();
            if (!name) {
                ids.push('Space');
            } else if (name.length === 1) {
                ids.push(name.toUpperCase());
            } else {
                const specialMap: Record<string, string> = {
                    'Esc': 'Escape',
                    'Backspace': 'Backspace',
                    'Tab': 'Tab',
                    'Caps Lock': 'CapsLock',
                    'Enter': 'Enter',
                    'Menu': 'ContextMenu',
                    'Insert': 'Insert',
                    'Delete': 'Delete',
                    'Home': 'Home',
                    'End': 'End',
                    'PgUp': 'PageUp',
                    'PgDn': 'PageDown',
                    'PrtSc': 'PrintScreen',
                    'Print Screen': 'PrintScreen',
                    'Scroll Lock': 'ScrollLock',
                    'Pause Break': 'Pause',
                    'Num Lock': 'NumLock',
                    'Up': 'ArrowUp',
                    'Down': 'ArrowDown',
                    'Left': 'ArrowLeft',
                    'Right': 'ArrowRight'
                };
                if (name === 'Shift') {
                    if (rowIndex === 4) {
                        ids.push(colIndex === 0 ? 'ShiftLeft' : 'ShiftRight');
                    } else {
                        ids.push('ShiftLeft');
                    }
                } else if (name === 'Ctrl') {
                    if (rowIndex === 5) {
                        ids.push(colIndex === 0 ? 'ControlLeft' : 'ControlRight');
                    } else {
                        ids.push('ControlLeft');
                    }
                } else if (name === 'Alt') {
                    if (rowIndex === 5) {
                        ids.push(colIndex < 3 ? 'AltLeft' : 'AltRight');
                    } else {
                        ids.push('AltLeft');
                    }
                } else if (name === 'Win') {
                    if (rowIndex === 5) {
                        ids.push(colIndex < 3 ? 'MetaLeft' : 'MetaRight');
                    } else {
                        ids.push('MetaLeft');
                    }
                } else {
                    const mapped = specialMap[name];
                    if (mapped) {
                        ids.push(mapped);
                    }
                }
            }
        }
        return ids;
    }

    private getKeyboardIdsFromEvent(event: KeyboardEvent): string[] {
        const ids: string[] = [];
        const key = event.key;
        const codeMap: Record<string, string> = {
            'ShiftLeft': 'ShiftLeft',
            'ShiftRight': 'ShiftRight',
            'ControlLeft': 'ControlLeft',
            'ControlRight': 'ControlRight',
            'AltLeft': 'AltLeft',
            'AltRight': 'AltRight',
            'MetaLeft': 'MetaLeft',
            'MetaRight': 'MetaRight',
            'Space': 'Space',
            'ArrowUp': 'ArrowUp',
            'ArrowDown': 'ArrowDown',
            'ArrowLeft': 'ArrowLeft',
            'ArrowRight': 'ArrowRight',
            'Insert': 'Insert',
            'Delete': 'Delete',
            'Home': 'Home',
            'End': 'End',
            'PageUp': 'PageUp',
            'PageDown': 'PageDown',
            'PrintScreen': 'PrintScreen',
            'ScrollLock': 'ScrollLock',
            'Pause': 'Pause',
            'NumLock': 'NumLock',
            'Numpad0': 'Numpad0',
            'Numpad1': 'Numpad1',
            'Numpad2': 'Numpad2',
            'Numpad3': 'Numpad3',
            'Numpad4': 'Numpad4',
            'Numpad5': 'Numpad5',
            'Numpad6': 'Numpad6',
            'Numpad7': 'Numpad7',
            'Numpad8': 'Numpad8',
            'Numpad9': 'Numpad9',
            'NumpadDecimal': 'NumpadDecimal',
            'NumpadDivide': 'NumpadDivide',
            'NumpadMultiply': 'NumpadMultiply',
            'NumpadSubtract': 'NumpadSubtract',
            'NumpadAdd': 'NumpadAdd',
            'NumpadEnter': 'NumpadEnter'
        };
        const codeMapped = codeMap[event.code];
        if (codeMapped) {
            ids.push(codeMapped);
        }
        if (key === ' ') {
            ids.push('Space');
        } else if (!event.code.startsWith('Numpad') && key.length === 1) {
            ids.push(key.toUpperCase());
        } else if (key.startsWith('F')) {
            const fn = Number(key.slice(1));
            if (Number.isFinite(fn) && fn >= 1 && fn <= 12) {
                ids.push(key);
            }
        }
        const specialMap: Record<string, string> = {
            'Escape': 'Escape',
            'Backspace': 'Backspace',
            'Tab': 'Tab',
            'CapsLock': 'CapsLock',
            'Enter': 'Enter',
            'ContextMenu': 'ContextMenu',
            'Menu': 'ContextMenu',
            'Insert': 'Insert',
            'Delete': 'Delete',
            'Home': 'Home',
            'End': 'End',
            'PageUp': 'PageUp',
            'PageDown': 'PageDown',
            'PrintScreen': 'PrintScreen',
            'ScrollLock': 'ScrollLock',
            'Pause': 'Pause',
            'ArrowUp': 'ArrowUp',
            'ArrowDown': 'ArrowDown',
            'ArrowLeft': 'ArrowLeft',
            'ArrowRight': 'ArrowRight',
            'NumLock': 'NumLock'
        };
        const mapped = specialMap[key];
        if (mapped) {
            ids.push(mapped);
        }
        return Array.from(new Set(ids));
    }

    private setKeyboardKeyPressed(id: string, pressed: boolean): void {
        const meshes = this.keyboardKeyMeshes.get(id);
        if (!meshes || meshes.length === 0) {
            return;
        }
        for (const mesh of meshes) {
            const pressDepth = mesh.userData.pressDepth ?? 0;
            mesh.position.z = pressed ? -pressDepth : 0;
        }
    }

    handleKeyboardKey(event: KeyboardEvent, pressed: boolean): void {
        if (!(this.parameters.keyboardEnabled ?? true)) {
            return;
        }
        if (this.keyboardKeyMeshes.size === 0) {
            return;
        }
        const ids = this.getKeyboardIdsFromEvent(event);
        if (ids.length === 0) {
            return;
        }
        ids.forEach(id => this.setKeyboardKeyPressed(id, pressed));
    }

    

    private formatSignedValue(value: number, digits = 2): string {
        const snapped = Math.abs(value) < 1e-6 ? 0 : value;
        const sign = snapped >= 0 ? '+' : '-';
        return `${sign}${Math.abs(snapped).toFixed(digits)}`;
    }

    private renderShaderOverlay(): void {
        if (!this.contentCanvas || !this.contentContext) {
            return;
        }
        const ctx = this.contentContext;
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, this.contentCanvas.width, this.contentCanvas.height);
        ctx.restore();

        if ((this.parameters.shaderType ?? 'mandelbrot') !== 'julia') {
            if (this.contentTexture) {
                this.contentTexture.needsUpdate = true;
            }
            return;
        }

        this.ensureTerminalFont();
        const cReal = this.parameters.juliaCReal ?? -0.8;
        const cImag = this.parameters.juliaCImag ?? 0.156;
        const centerX = this.parameters.juliaPanX ?? 0.0;
        const centerY = this.parameters.juliaPanY ?? 0.0;
        const line = `c=${this.formatSignedValue(cReal)} ${this.formatSignedValue(cImag)}i  X=${this.formatSignedValue(centerX)} Y=${this.formatSignedValue(centerY)}`;
        const fontSize = Math.max(10, Math.round(this.logicalHeight * 0.07));
        const padX = Math.round(this.contentCanvas.width * 0.5);
        const padY = Math.max(6, Math.round(this.logicalHeight * 0.02));
        ctx.save();
        ctx.font = `${fontSize}px '${this.shaderOverlayFontFamily}', monospace`;
        ctx.fillStyle = 'rgba(255, 255, 255, 1.0)';
        ctx.textBaseline = 'top';
        ctx.textAlign = 'center';
        ctx.imageSmoothingEnabled = false;
        ctx.fillText(line, padX, padY);
        ctx.restore();

        if (this.contentTexture) {
            this.contentTexture.needsUpdate = true;
        }
    }

    private buildKeyboardLabelTextureFromLayout(layout: {
        keys: Array<{
            outer: { x: number; y: number; width: number; height: number; rx: number };
            inner: { x: number; y: number; width: number; height: number; rx: number };
        }>;
        bounds: { minX: number; minY: number; maxX: number; maxY: number };
    }): THREE.CanvasTexture | null {
        const { keys, bounds } = layout;
        const svgWidth = bounds.maxX - bounds.minX;
        const svgHeight = bounds.maxY - bounds.minY;
        if (svgWidth <= 0 || svgHeight <= 0) {
            return null;
        }

        const canvas = document.createElement('canvas');
        const dpr = typeof window !== 'undefined' ? Math.min(window.devicePixelRatio || 1, 2) : 1;
        canvas.width = Math.max(1, Math.ceil(svgWidth * dpr));
        canvas.height = Math.max(1, Math.ceil(svgHeight * dpr));
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            return null;
        }

        ctx.scale(dpr, dpr);
        ctx.translate(-bounds.minX, -bounds.minY);
        ctx.textBaseline = 'middle';
        ctx.textAlign = 'left';
        ctx.fillStyle = 'rgba(30, 30, 30, 0.9)';
        ctx.imageSmoothingEnabled = false;

        const rows = this.groupKeyboardRows(keys);
        const labelRows = this.getKeyboardLabelRows();

        const fontFamily = this.terminalFontFamily;
        const drawText = (
            text: string,
            x: number,
            y: number,
            rotation: number | undefined,
            align: CanvasTextAlign
        ): void => {
            if (!rotation) {
                ctx.textAlign = align;
                ctx.fillText(text, x, y);
                return;
            }
            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(rotation);
            ctx.textAlign = align;
            ctx.textBaseline = 'middle';
            ctx.fillText(text, 0, 0);
            ctx.restore();
        };
        rows.forEach((row, rowIndex) => {
            const labels = labelRows[rowIndex];
            if (!labels) {
                return;
            }
            const count = Math.min(row.length, labels.length);
            for (let i = 0; i < count; i += 1) {
                const key = row[i];
                const label = labels[i];
                if (!label) {
                    continue;
                }
                const rect = key.inner ?? key.outer;
                const padX = rect.width * 0.12;
                const padY = rect.height * 0.18;

                if (label.top || label.bottom) {
                    const fontSize = Math.min(rect.height * 0.33, 13);
                    ctx.font = `${fontSize}px '${fontFamily}', monospace`;
                    ctx.textBaseline = 'middle';
                    const x = rect.x + padX;
                    const topY = rect.y + rect.height * 0.32;
                    const bottomY = rect.y + rect.height * 0.72;
                    if (label.top) {
                        drawText(label.top, x, topY, label.topRotate, 'left');
                    }
                    if (label.bottom) {
                        drawText(label.bottom, x, bottomY, label.bottomRotate, 'left');
                    }
                } else if (label.center) {
                    const align = label.align ?? 'center';
                    let fontSize = Math.min(rect.height * 0.5, 16);
                    ctx.font = `${fontSize}px '${fontFamily}', monospace`;
                    const text = label.center;
                    let maxWidth = rect.width - padX * 2;
                    let metrics = ctx.measureText(text);
                    if (metrics.width > maxWidth && metrics.width > 0) {
                        fontSize = Math.max(10, fontSize * (maxWidth / metrics.width));
                        ctx.font = `${fontSize}px '${fontFamily}', monospace`;
                        metrics = ctx.measureText(text);
                    }
                    ctx.textBaseline = 'middle';
                    const x = align === 'left' ? rect.x + padX : rect.x + rect.width * 0.5;
                    const y = rect.y + rect.height * 0.58;
                    drawText(text, x, y, label.centerRotate, align === 'left' ? 'left' : 'center');
                }
            }
        });

        const texture = new THREE.CanvasTexture(canvas);
        texture.flipY = false;
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.needsUpdate = true;
        return texture;
    }

    private parseTranslate(value: string | null): { x: number; y: number } {
        if (!value) {
            return { x: 0, y: 0 };
        }
        const match = /translate\\(\\s*([\\-\\d.]+)(?:[\\s,]+([\\-\\d.]+))?\\s*\\)/.exec(value);
        if (!match) {
            return { x: 0, y: 0 };
        }
        const x = parseFloat(match[1] ?? '0') || 0;
        const y = parseFloat(match[2] ?? '0') || 0;
        return { x, y };
    }

    private getAccumulatedTranslate(element: Element | null): { x: number; y: number } {
        let x = 0;
        let y = 0;
        let current: Element | null = element;
        while (current) {
            const transform = current.getAttribute('transform');
            if (transform) {
                const delta = this.parseTranslate(transform);
                x += delta.x;
                y += delta.y;
            }
            current = current.parentElement;
        }
        return { x, y };
    }

    private buildKeyboardLayout(layout: {
        keys: Array<{
            outer: { x: number; y: number; width: number; height: number; rx: number };
            inner: { x: number; y: number; width: number; height: number; rx: number };
        }>;
        bounds: { minX: number; minY: number; maxX: number; maxY: number };
    }): void {
        if (!this.scene) {
            return;
        }

        this.ensureScreenLightDownsample();
        this.clearKeyboard();

        const { keys, bounds } = layout;
        const svgWidth = bounds.maxX - bounds.minX;
        const svgHeight = bounds.maxY - bounds.minY;
        const widthScale = this.parameters.keyboardWidthScale ?? 0.9;
        const targetWidth = CRTScreenScene.DEFAULT_KEYBOARD_WIDTH * widthScale;
        const scale = targetWidth / svgWidth;
        const keyboardZ = 0;
        const keyboardY = 0;

        this.ensureTerminalFont();
        if (!this.keyboardLabelTexture) {
            const labelTexture = this.buildKeyboardLabelTextureFromLayout(layout);
            if (labelTexture) {
                this.keyboardLabelTexture = labelTexture;
            }
        }

        const baseKeyWidth = 52 * scale;
        this.keyboardBaseKeyWidth = baseKeyWidth;
        const depthRatio = this.parameters.keyboardDepthRatio ?? 0.6;
        const topInset = Math.max(0, this.parameters.keyboardTopInset ?? 0.12);
        const bumpScale = Math.max(0.1, this.parameters.keyboardBumpScale ?? 2.5);
        const bumpStrength = Math.max(0, this.parameters.keyboardBumpStrength ?? 0.01);
        const baseColorHex = this.parameters.keyboardBaseColor ?? '#DAD8CE';
        const roughnessMin = this.parameters.keyboardRoughnessMin ?? 0.78;
        const roughnessMax = this.parameters.keyboardRoughnessMax ?? 0.92;
        const metalness = this.parameters.keyboardMetalness ?? 0.0;
        const labelOpacity = this.parameters.keyboardLabelOpacity ?? 1.0;
        const cornerRadiusBottom = this.parameters.keyboardCornerRadiusBottom ?? 0.08;
        const cornerRadiusTop = this.parameters.keyboardCornerRadiusTop ?? 0.04;
        const topEdgeRadius = Math.max(0, this.parameters.keyboardTopEdgeRadius ?? 0.0);

        if (!this.keyboardBaseColorUniform) {
            this.keyboardBaseColorUniform = uniform(new THREE.Color(baseColorHex));
        } else {
            this.keyboardBaseColorUniform.value.set(baseColorHex);
        }
        if (!this.keyboardBumpScaleUniform) {
            this.keyboardBumpScaleUniform = uniform(bumpScale, 'float');
        } else {
            this.keyboardBumpScaleUniform.value = bumpScale;
        }
        if (!this.keyboardBumpStrengthUniform) {
            this.keyboardBumpStrengthUniform = uniform(baseKeyWidth * bumpStrength, 'float');
        } else {
            this.keyboardBumpStrengthUniform.value = baseKeyWidth * bumpStrength;
        }
        if (!this.keyboardRoughnessMinUniform) {
            this.keyboardRoughnessMinUniform = uniform(roughnessMin, 'float');
        } else {
            this.keyboardRoughnessMinUniform.value = roughnessMin;
        }
        if (!this.keyboardRoughnessMaxUniform) {
            this.keyboardRoughnessMaxUniform = uniform(roughnessMax, 'float');
        } else {
            this.keyboardRoughnessMaxUniform.value = roughnessMax;
        }
        if (!this.keyboardMetalnessUniform) {
            this.keyboardMetalnessUniform = uniform(metalness, 'float');
        } else {
            this.keyboardMetalnessUniform.value = metalness;
        }
        if (!this.keyboardLabelOpacityUniform) {
            this.keyboardLabelOpacityUniform = uniform(labelOpacity, 'float');
        } else {
            this.keyboardLabelOpacityUniform.value = labelOpacity;
        }

        const outerDepth = baseKeyWidth * depthRatio;
        const radiusWorldBottom = Math.max(0, cornerRadiusBottom) * baseKeyWidth;
        const radiusWorldTop = Math.max(0, cornerRadiusTop) * baseKeyWidth;
        const edgeRadiusWorld = topEdgeRadius * baseKeyWidth;

        const createRoundedRectGeometry = (
            width: number,
            height: number,
            depth: number,
            radiusBottom: number,
            radiusTop: number,
            inset: number,
            edgeRadius: number
        ): THREE.BufferGeometry => {
            const insetX = Math.min(inset, width * 0.45);
            const insetY = Math.min(inset, height * 0.45);
            const topWidth = Math.max(0.001, width - 2 * insetX);
            const topHeight = Math.max(0.001, height - 2 * insetY);
            const rBottom = Math.min(radiusBottom, width * 0.5, height * 0.5);
            const rTop = Math.min(radiusTop, topWidth * 0.5, topHeight * 0.5);
            const curveSegments = 6;
            const buildRoundedShape = (shapeWidth: number, shapeHeight: number, radius: number): THREE.Shape => {
                const clampedRadius = Math.min(radius, shapeWidth * 0.5, shapeHeight * 0.5);
                const shape = new THREE.Shape();
                const x = -shapeWidth * 0.5;
                const y = -shapeHeight * 0.5;
                shape.moveTo(x + clampedRadius, y);
                shape.lineTo(x + shapeWidth - clampedRadius, y);
                shape.quadraticCurveTo(x + shapeWidth, y, x + shapeWidth, y + clampedRadius);
                shape.lineTo(x + shapeWidth, y + shapeHeight - clampedRadius);
                shape.quadraticCurveTo(x + shapeWidth, y + shapeHeight, x + shapeWidth - clampedRadius, y + shapeHeight);
                shape.lineTo(x + clampedRadius, y + shapeHeight);
                shape.quadraticCurveTo(x, y + shapeHeight, x, y + shapeHeight - clampedRadius);
                shape.lineTo(x, y + clampedRadius);
                shape.quadraticCurveTo(x, y, x + clampedRadius, y);
                return shape;
            };

            const bottomShape = buildRoundedShape(width, height, rBottom);
            const topShape = buildRoundedShape(topWidth, topHeight, rTop);
            const bottomPoints = bottomShape.getPoints(curveSegments);
            const topPoints = topShape.getPoints(curveSegments);

            if (bottomPoints.length > 1 && bottomPoints[0].equals(bottomPoints[bottomPoints.length - 1])) {
                bottomPoints.pop();
            }
            if (topPoints.length > 1 && topPoints[0].equals(topPoints[topPoints.length - 1])) {
                topPoints.pop();
            }

            const bevelDepth = Math.min(edgeRadius, depth * 0.45);
            const maxBevelX = insetX;
            const maxBevelY = insetY;
            const bevelX = Math.min(edgeRadius, maxBevelX);
            const bevelY = Math.min(edgeRadius, maxBevelY);
            const bevelSegments = (bevelDepth > 0 && (bevelX > 0 || bevelY > 0)) ? 3 : 0;
            const rings: Array<{ points: THREE.Vector2[]; z: number }> = [];
            rings.push({ points: bottomPoints, z: -depth * 0.5 });

            if (bevelSegments > 0) {
                const topZ = depth * 0.5;
                const bevelStartZ = topZ - bevelDepth;
                for (let s = 0; s <= bevelSegments; s += 1) {
                    const t = s / bevelSegments;
                    const widthStep = topWidth + (bevelX * 2) * (1 - t);
                    const heightStep = topHeight + (bevelY * 2) * (1 - t);
                    const radiusStep = rTop + bevelX * (1 - t);
                    const ringShape = buildRoundedShape(widthStep, heightStep, radiusStep);
                    const points = ringShape.getPoints(curveSegments);
                    if (points.length > 1 && points[0].equals(points[points.length - 1])) {
                        points.pop();
                    }
                    const z = bevelStartZ + bevelDepth * t;
                    rings.push({ points, z });
                }
            } else {
                rings.push({ points: topPoints, z: depth * 0.5 });
            }

            const ringCount = Math.min(...rings.map(ring => ring.points.length));
            const positions: number[] = [];
            const indices: number[] = [];

            rings.forEach(ring => {
                for (let i = 0; i < ringCount; i += 1) {
                    const p = ring.points[i];
                    positions.push(p.x, p.y, ring.z);
                }
            });

            const ringStride = ringCount;
            for (let ringIndex = 0; ringIndex < rings.length - 1; ringIndex += 1) {
                const base = ringIndex * ringStride;
                const nextBase = base + ringStride;
                for (let i = 0; i < ringCount; i += 1) {
                    const next = (i + 1) % ringCount;
                    const b0 = base + i;
                    const b1 = base + next;
                    const t0 = nextBase + i;
                    const t1 = nextBase + next;
                    indices.push(b0, b1, t1, b0, t1, t0);
                }
            }

            const sideGeom = new THREE.BufferGeometry();
            sideGeom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            sideGeom.setIndex(indices);

            const topGeom = new THREE.ShapeGeometry(topShape, curveSegments);
            topGeom.translate(0, 0, depth * 0.5);
            const bottomGeom = new THREE.ShapeGeometry(bottomShape, curveSegments);
            bottomGeom.translate(0, 0, -depth * 0.5);

            const flipWinding = (geometry: THREE.BufferGeometry): void => {
                const index = geometry.index;
                if (!index) {
                    return;
                }
                const array = index.array as ArrayLike<number>;
                for (let i = 0; i < array.length; i += 3) {
                    const i1 = array[i + 1];
                    const i2 = array[i + 2];
                    (array as any)[i + 1] = i2;
                    (array as any)[i + 2] = i1;
                }
                index.needsUpdate = true;
            };

            flipWinding(bottomGeom);

            delete (topGeom as any).attributes.normal;
            delete (bottomGeom as any).attributes.normal;
            delete (topGeom as any).attributes.uv;
            delete (bottomGeom as any).attributes.uv;

            const merged = mergeGeometries([sideGeom, topGeom, bottomGeom], false);
            if (!merged) {
                topGeom.dispose();
                bottomGeom.dispose();
                return sideGeom;
            }

            sideGeom.dispose();
            topGeom.dispose();
            bottomGeom.dispose();

            merged.computeVertexNormals();
            return merged;
        };
        const hash3 = (p: any): any => {
            return hash(p.x.mul(float(127.1)).add(p.y.mul(float(311.7))).add(p.z.mul(float(74.7))));
        };

        const pebbleNoise3 = (p: any): any => {
            const pShift = p.add(vec3(1000.0, 1000.0, 1000.0));
            const cell = floor(pShift);
            const f = fract(pShift);
            const u = f.mul(f).mul(float(3.0).sub(f.mul(float(2.0))));
            const c000 = hash3(cell);
            const c100 = hash3(cell.add(vec3(1.0, 0.0, 0.0)));
            const c010 = hash3(cell.add(vec3(0.0, 1.0, 0.0)));
            const c110 = hash3(cell.add(vec3(1.0, 1.0, 0.0)));
            const c001 = hash3(cell.add(vec3(0.0, 0.0, 1.0)));
            const c101 = hash3(cell.add(vec3(1.0, 0.0, 1.0)));
            const c011 = hash3(cell.add(vec3(0.0, 1.0, 1.0)));
            const c111 = hash3(cell.add(vec3(1.0, 1.0, 1.0)));
            const x00 = mix(c000, c100, u.x);
            const x10 = mix(c010, c110, u.x);
            const x01 = mix(c001, c101, u.x);
            const x11 = mix(c011, c111, u.x);
            const y0 = mix(x00, x10, u.y);
            const y1 = mix(x01, x11, u.y);
            return mix(y0, y1, u.z);
        };

        const outerMat = new MeshBasicNodeMaterial();
        const baseColorUniform = this.keyboardBaseColorUniform;
        const noisePos = positionLocal.mul(this.keyboardBumpScaleUniform);
        const n1 = pebbleNoise3(noisePos.mul(float(3.0)));
        const n2 = pebbleNoise3(noisePos.mul(float(9.0)));
        const pebbles = mix(n1, n2, float(0.6));
        const tint = mix(float(0.99), float(1.01), pebbles);
        const baseColorNode = baseColorUniform.mul(tint);
        const roughnessNode = mix(this.keyboardRoughnessMinUniform, this.keyboardRoughnessMaxUniform, pebbles);
        const height = pebbles.sub(float(0.5)).mul(this.keyboardBumpStrengthUniform);
        const displacedWorld = positionWorld.add(normalWorld.mul(height));
        const bumpedNormal = normalize(cross(dFdx(displacedWorld.xyz), dFdy(displacedWorld.xyz)));
        const labelTextureNode = this.keyboardLabelTexture ? texture(this.keyboardLabelTexture) : null;
        const baseWithLabels = Fn(() => {
            let base = baseColorNode;
            if (labelTextureNode) {
                const labelSample = labelTextureNode.sample(uv());
                const normalZ = max(attribute('normal', 'vec3').z, float(0.0));
                const labelMask = smoothstep(float(0.4), float(0.8), normalZ);
                const alpha = labelSample.a.mul(labelMask).mul(this.keyboardLabelOpacityUniform);
                base = mix(base, labelSample.rgb, alpha);
            }
            return base;
        })();
        const metalnessNode = this.keyboardMetalnessUniform;
        const lightIntensityNode = this.keyboardLightIntensityUniform;
        const lightSaturationNode = this.keyboardLightSaturationUniform;
        const lightWrapNode = this.keyboardLightWrapUniform;
        const lightOriginNode = this.keyboardLightOriginUniform;
        const screenRightNode = this.keyboardLightScreenRightUniform;
        const screenUpNode = this.keyboardLightScreenUpUniform;
        const screenSizeNode = this.keyboardLightScreenSizeUniform;
        const screenLightTextureNode = this.screenLightTextureNode;
        const fallbackTextureNode = this.contentTextureNode;
        const gridX = CRTScreenScene.KEYBOARD_LIGHT_GRID_X;
        const gridY = CRTScreenScene.KEYBOARD_LIGHT_GRID_Y;
        const invGridX = 1 / gridX;
        const invGridY = 1 / gridY;
        outerMat.colorNode = Fn(() => {
            const worldPos = positionWorld;
            const viewDir = normalize(cameraPosition.sub(worldPos));
            const wrap = max(lightWrapNode, float(0.0));
            const gloss = max(float(0.0), float(1.0).sub(roughnessNode));
            const specExp = mix(float(8.0), float(128.0), gloss);
            const specColor = mix(vec3(0.04, 0.04, 0.04), baseWithLabels, metalnessNode);
            let diffuseAccum = vec3(0.0, 0.0, 0.0);
            let specAccum = vec3(0.0, 0.0, 0.0);

            for (let y = 0; y < gridY; y += 1) {
                for (let x = 0; x < gridX; x += 1) {
                    const u = float((x + 0.5) * invGridX);
                    const v = float((y + 0.5) * invGridY);
                    const offsetX = u.sub(0.5).mul(screenSizeNode.x);
                    const offsetY = v.sub(0.5).mul(screenSizeNode.y);
                    const lightPos = lightOriginNode
                        .add(screenRightNode.mul(offsetX))
                        .add(screenUpNode.mul(offsetY));
                    const toLight = lightPos.sub(worldPos);
                    const dist2 = dot(toLight, toLight);
                    const lightDir = normalize(toLight);
                    const nDotL = max(dot(bumpedNormal, lightDir), float(0.0));
                    const wrapped = clamp(nDotL.add(wrap).div(float(1.0).add(wrap)), 0.0, 1.0);
                    const halfDir = normalize(lightDir.add(viewDir));
                    const nDotH = max(dot(bumpedNormal, halfDir), float(0.0));
                    const spec = pow(nDotH, specExp).mul(gloss);
                    const screenRadius = screenSizeNode.x.add(screenSizeNode.y).mul(0.5);
                    const radius2 = screenRadius.mul(screenRadius).add(float(1.0));
                    const falloff = float(1.0).div(float(1.0).add(dist2.div(radius2)));
                    const sampleColor = screenLightTextureNode
                        ? screenLightTextureNode.sample(vec2(u, v)).rgb
                        : (fallbackTextureNode
                            ? fallbackTextureNode.sample(vec2(u, v)).rgb
                            : vec3(1.0, 1.0, 1.0));
                    const sat = clamp(lightSaturationNode, 0.0, 1.0);
                    const luma = dot(sampleColor, vec3(0.2126, 0.7152, 0.0722));
                    const gray = vec3(luma, luma, luma);
                    const saturated = mix(gray, sampleColor, sat);
                    diffuseAccum = diffuseAccum.add(saturated.mul(wrapped).mul(falloff));
                    specAccum = specAccum.add(saturated.mul(spec).mul(falloff));
                }
            }

            const gridScale = float(1.0 / (gridX * gridY));
            const lit = baseWithLabels.mul(diffuseAccum.mul(gridScale))
                .add(specColor.mul(specAccum.mul(gridScale)));
            return lit.mul(lightIntensityNode);
        })();

        const centerX = bounds.minX + svgWidth * 0.5;
        const centerY = bounds.minY + svgHeight * 0.5;

        const labelRows = this.getKeyboardLabelRows();
        const rows = this.groupKeyboardRows(keys);
        const keyLabelMap = new Map<typeof keys[number], KeyboardLabel>();
        rows.forEach((row, rowIndex) => {
            const labels = labelRows[rowIndex];
            if (!labels) {
                return;
            }
            const count = Math.min(row.length, labels.length);
            for (let i = 0; i < count; i += 1) {
                keyLabelMap.set(row[i], labels[i]);
            }
        });

        this.keyboardKeyMeshes.clear();
        this.keyboardKeyMeshList = [];
        const pressDepth = outerDepth * 0.3;
        const group = new THREE.Group();

        keys.forEach(key => {
            const outer = key.outer;

            const outerWidth = outer.width * scale;
            const outerHeight = outer.height * scale;

            const outerCenterX = (outer.x + outer.width * 0.5 - centerX) * scale;
            const outerCenterY = (outer.y + outer.height * 0.5 - centerY) * -scale;

            const outerGeom = createRoundedRectGeometry(
                outerWidth,
                outerHeight,
                outerDepth,
                radiusWorldBottom,
                radiusWorldTop,
                topInset,
                edgeRadiusWorld
            );
            outerGeom.translate(outerCenterX, outerCenterY + keyboardY, keyboardZ);
            this.applyKeyboardLabelUVs(outerGeom, {
                centerX,
                centerY,
                scale,
                bounds
            });

            const mesh = new THREE.Mesh(outerGeom, outerMat);
            mesh.userData.pressDepth = pressDepth;
            this.keyboardKeyMeshList.push(mesh);
            const label = keyLabelMap.get(key);
            let rowIndex = -1;
            let colIndex = -1;
            rows.some((row, r) => {
                const c = row.indexOf(key);
                if (c >= 0) {
                    rowIndex = r;
                    colIndex = c;
                    return true;
                }
                return false;
            });
            const ids = this.getKeyboardIdsFromLabel(label, rowIndex, colIndex);
            ids.forEach(id => {
                const list = this.keyboardKeyMeshes.get(id) ?? [];
                list.push(mesh);
                this.keyboardKeyMeshes.set(id, list);
            });
            group.add(mesh);
        });

        const outerMesh = this.keyboardKeyMeshList[0] ?? null;
        if (!outerMesh) {
            return;
        }
        group.rotation.x = -Math.PI / 2;

        this.keyboardGroup = group;
        this.keyboardOuterMesh = outerMesh;
        this.updateKeyboardPlacement();
        this.scene.add(group);
        this.addKeyboardLights();
    }

    private async loadKeyboardLayout(): Promise<void> {
        if (!this.scene || this.keyboardLoading || !(this.parameters.keyboardEnabled ?? true)) {
            return;
        }

        if (typeof DOMParser === 'undefined' || typeof fetch === 'undefined') {
            return;
        }

        if (this.keyboardLayout) {
            this.buildKeyboardLayout(this.keyboardLayout);
            return;
        }

        this.keyboardLoading = true;
        try {
            const response = await fetch('keyboard-layout-104.svg');
            if (!response.ok) {
                this.keyboardLoading = false;
                return;
            }

            const svgText = await response.text();
            const parser = new DOMParser();
            const doc = parser.parseFromString(svgText, 'image/svg+xml');
            const keyGroups = Array.from(doc.querySelectorAll('g.keycap'));
            if (keyGroups.length === 0) {
                this.keyboardLoading = false;
                return;
            }

            const keys: Array<{
                outer: { x: number; y: number; width: number; height: number; rx: number };
                inner: { x: number; y: number; width: number; height: number; rx: number };
            }> = [];

            let minX = Number.POSITIVE_INFINITY;
            let minY = Number.POSITIVE_INFINITY;
            let maxX = Number.NEGATIVE_INFINITY;
            let maxY = Number.NEGATIVE_INFINITY;

            for (const group of keyGroups) {
                const rects = Array.from(group.querySelectorAll('rect'));
                if (rects.length === 0) {
                    continue;
                }

                const offset = this.getAccumulatedTranslate(group);
                const rectInfos = rects.map(rect => {
                    const x = (parseFloat(rect.getAttribute('x') ?? '0') || 0) + offset.x;
                    const y = (parseFloat(rect.getAttribute('y') ?? '0') || 0) + offset.y;
                    const width = parseFloat(rect.getAttribute('width') ?? '0') || 0;
                    const height = parseFloat(rect.getAttribute('height') ?? '0') || 0;
                    const rx = parseFloat(rect.getAttribute('rx') ?? '0') || 0;
                    return { x, y, width, height, rx, area: width * height };
                }).filter(rect => rect.width > 0 && rect.height > 0);

                if (rectInfos.length < 2) {
                    continue;
                }

                rectInfos.sort((a, b) => b.area - a.area);
                const outer = rectInfos[0];
                const inner = rectInfos[rectInfos.length - 1];

                keys.push({
                    outer: { x: outer.x, y: outer.y, width: outer.width, height: outer.height, rx: outer.rx },
                    inner: { x: inner.x, y: inner.y, width: inner.width, height: inner.height, rx: inner.rx }
                });

                minX = Math.min(minX, outer.x);
                minY = Math.min(minY, outer.y);
                maxX = Math.max(maxX, outer.x + outer.width);
                maxY = Math.max(maxY, outer.y + outer.height);
            }

            if (keys.length === 0 || !isFinite(minX) || !isFinite(maxX)) {
                this.keyboardLoading = false;
                return;
            }

            this.keyboardLayout = {
                keys,
                bounds: { minX, minY, maxX, maxY }
            };
            const labelTexture = this.buildKeyboardLabelTextureFromLayout(this.keyboardLayout);
            if (this.keyboardLabelTexture) {
                this.keyboardLabelTexture.dispose();
                this.keyboardLabelTexture = null;
            }
            if (labelTexture) {
                this.keyboardLabelTexture = labelTexture;
            }
            this.scheduleKeyboardRebuild();
        } catch (error) {
            console.warn('Failed to load keyboard layout', error);
        } finally {
            this.keyboardLoading = false;
        }
    }

    dispose(): void {
        this.cleanup();
    }

    private initializeGPUComputeShaders(): void {
        const screenWidth = this.logicalWidth;
        const screenHeight = this.logicalHeight;
        const totalSubpixels = screenWidth * screenHeight * 3;
        const totalPixels = screenWidth * screenHeight;

        this.ensureContentCanvas();
        if (this.contentTexture) {
            this.contentTextureNode = texture(this.contentTexture);
        }
        
        // Initialize CPU-side target array (only updated when content changes)
        this.targetColorArray = new Float32Array(totalPixels * 3);
        
        // Create GPU storage buffers
        this.currentColors = instancedArray(totalSubpixels, 'vec3');
        this.targetColors = instancedArray(this.targetColorArray, 'vec3');
        
        // Initialize static pattern uniforms
        this.staticSpeedUniform = uniform(this.parameters.staticSpeed ?? 15.0, 'float');
        this.staticContrastUniform = uniform(this.parameters.staticContrast ?? 0.2, 'float');
        if (!this.minBrightnessUniform) {
            this.minBrightnessUniform = uniform(this.parameters.minBrightness ?? 0.1, 'float');
        } else {
            this.minBrightnessUniform.value = this.parameters.minBrightness ?? 0.1;
        }
        if (!this.colorAttackUniform) {
            this.colorAttackUniform = uniform(this.parameters.colorAttack ?? 8.0, 'float');
        } else {
            this.colorAttackUniform.value = this.parameters.colorAttack ?? 8.0;
        }
        if (!this.colorDecayUniform) {
            this.colorDecayUniform = uniform(this.parameters.colorDecay ?? 4.0, 'float');
        } else {
            this.colorDecayUniform.value = this.parameters.colorDecay ?? 4.0;
        }
        this.useExternalContentUniform = uniform(this.useExternalContent ? 1.0 : 0.0, 'float');
        this.useExternalTextureUniform = uniform(this.useExternalContent ? 1.0 : 0.0, 'float');
        const activeShader = this.getActiveShaderParams();
        this.shaderPanXUniform = uniform(activeShader.panX, 'float');
        this.shaderPanYUniform = uniform(activeShader.panY, 'float');
        this.shaderZoomUniform = uniform(Math.max(0.0001, activeShader.zoom), 'float');
        this.shaderEnabledUniform = uniform((this.parameters.displayMode ?? 'video') === 'shader' ? 1.0 : 0.0, 'float');
        this.shaderTypeUniform = uniform((this.parameters.shaderType ?? 'mandelbrot') === 'julia' ? 1.0 : 0.0, 'float');
        this.shaderJuliaCUniform = uniform(
            new THREE.Vector2(this.parameters.juliaCReal ?? -0.8, this.parameters.juliaCImag ?? 0.156),
            'vec2'
        );

        const shaderComputeFn = Fn(() => {
            const idx = instanceIndex;
            const screenWidthF = float(screenWidth);
            const screenHeightF = float(screenHeight);
            const pixelX = idx.mod(uint(screenWidth));
            const pixelY = idx.div(uint(screenWidth));
            const shaderUV = vec2(
                pixelX.toFloat().add(float(0.5)).div(screenWidthF),
                pixelY.toFloat().add(float(0.5)).div(screenHeightF)
            );

            const zoom = max(this.shaderZoomUniform, float(0.0001));
            const viewWidth = float(3.5).div(zoom);
            const viewHeight = viewWidth.mul(screenHeightF.div(screenWidthF));
            const center = vec2(this.shaderPanXUniform, this.shaderPanYUniform);
            const coord = center.add(shaderUV.sub(float(0.5)).mul(vec2(viewWidth, viewHeight)));
            const juliaMix = clamp(this.shaderTypeUniform, 0.0, 1.0);
            const c = mix(coord, this.shaderJuliaCUniform, juliaMix);
            const z = mix(vec2(0.0), coord, juliaMix).toVar();
            const iter = float(0.0).toVar();
            const maxIter = float(CRTScreenScene.SHADER_MAX_ITER);

            Loop({ start: 0, end: CRTScreenScene.SHADER_MAX_ITER, type: 'int' }, () => {
                const x2 = z.x.mul(z.x);
                const y2 = z.y.mul(z.y);
                If(x2.add(y2).greaterThan(float(4.0)), () => {
                    Break();
                });
                z.assign(vec2(x2.sub(y2).add(c.x), z.x.mul(z.y).mul(float(2.0)).add(c.y)));
                iter.addAssign(1.0);
            });

            const inSet = iter.greaterThanEqual(maxIter);
            const band = floor(iter).mod(float(8.0));
            const tBand = band.div(float(7.0));
            const r = clamp(float(1.5).sub(abs(tBand.mul(float(4.0)).sub(float(3.0)))), 0.0, 1.0);
            const g = clamp(float(1.5).sub(abs(tBand.mul(float(4.0)).sub(float(2.0)))), 0.0, 1.0);
            const b = clamp(float(1.5).sub(abs(tBand.mul(float(4.0)).sub(float(1.0)))), 0.0, 1.0);
            const bandColor = vec3(r, g, b);
            const finalColor = select(inSet, vec3(0.0), bandColor);
            this.targetColors.element(idx).assign(finalColor);
        });

        this.shaderComputeNode = shaderComputeFn().compute(totalPixels);
        
        // Initialize beam scan uniforms
        this.scanFramerateUniform = uniform(this.parameters.scanFramerate ?? 30, 'float');
        this.scanHeadUniform = uniform(0, 'float');
        this.beamPixelDurationUniform = uniform(this.parameters.beamPixelDuration ?? 1.0, 'float');
        
        // Power effect uniforms (used by compute shader)
        this.powerWarmup = this.parameters.powerOn ? 1.0 : 0.0;
        this.powerOnUniform = uniform(this.parameters.powerOn ? 1.0 : 0.0, 'float');
        this.powerTransitionUniform = uniform(1.0, 'float');
        this.powerDirectionUniform = uniform(0.0, 'float');
        this.powerFlashUniform = uniform(this.parameters.powerFlash ?? 0.6, 'float');
        this.powerWarmupUniform = uniform(this.powerWarmup, 'float');
        this.powerCollapseRatioUniform = uniform(1.0, 'float');
        this.updatePowerCollapseRatio();
        
        // Create GPU compute shader for color interpolation and animation
        const computeShaderFn = Fn(() => {
            // Get subpixel index
            const idx = instanceIndex;
            
            // Get current color from storage
            const currentColor = this.currentColors.element(idx);
            
            // Calculate which logical pixel and subpixel this is
            // CRT scans horizontally: left to right, top to bottom
            // Layout: row 0 all columns, then row 1 all columns, etc.
            const totalColumns = uint(screenWidth * 3);  // Total subpixel columns
            const rowIdx = idx.div(totalColumns);        // Which row (0 to screenHeight-1)
            const columnIdx = idx.mod(totalColumns);     // Which column (0 to screenWidth*3-1)
            const pixelX = columnIdx.div(uint(3));       // Which logical pixel horizontally
            const pixelY = rowIdx;                       // Which logical pixel vertically
            const subpixelIdx = columnIdx.mod(uint(3));  // 0=R, 1=G, 2=B
            
            // Normalized pixel position for power collapse ([-1, 1])
            const screenWidthF = float(screenWidth);
            const screenHeightF = float(screenHeight);
            const nx = pixelX.toFloat().add(float(0.5)).div(screenWidthF).mul(float(2.0)).sub(float(1.0));
            const ny = pixelY.toFloat().add(float(0.5)).div(screenHeightF).mul(float(2.0)).sub(float(1.0));
            const nxAbs = abs(nx);
            const nyAbs = abs(ny);
            
            // Power transition controls active scan region
            const powerPhase = this.powerTransitionUniform;
            const powerDir = this.powerDirectionUniform;
            const isPoweringOn = powerDir.greaterThan(float(0.5));
            const isTransition = abs(powerDir).greaterThan(float(0.1));
            const minExtentX = float(1.0).div(screenWidthF);
            const minExtentY = float(1.0).div(screenHeightF);
            
            const collapseRatio = max(this.powerCollapseRatioUniform, float(0.001));
            const collapsePhase = clamp(powerPhase.div(collapseRatio), 0.0, 1.0);
            const widthPhase = select(isPoweringOn, powerPhase, collapsePhase);
            const onWidthX = mix(minExtentX, float(1.0), smoothstep(float(0.0), float(0.6), widthPhase));
            const onWidthY = mix(minExtentY, float(1.0), smoothstep(float(0.2), float(1.0), widthPhase));
            const offWidthY = mix(float(1.0), minExtentY, smoothstep(float(0.0), float(0.6), widthPhase));
            const offWidthX = mix(float(1.0), minExtentX, smoothstep(float(0.3), float(1.0), widthPhase));
            const steadyExtentX = select(this.powerOnUniform.greaterThan(float(0.5)), float(1.0), minExtentX);
            const steadyExtentY = select(this.powerOnUniform.greaterThan(float(0.5)), float(1.0), minExtentY);
            const widthX = select(isTransition, select(isPoweringOn, onWidthX, offWidthX), steadyExtentX);
            const widthY = select(isTransition, select(isPoweringOn, onWidthY, offWidthY), steadyExtentY);
            const featherPixels = float(4.0);
            const featherX = featherPixels.div(screenWidthF);
            const featherY = featherPixels.div(screenHeightF);
            const edgeX = smoothstep(widthX, widthX.add(featherX), nxAbs);
            const edgeY = smoothstep(widthY, widthY.add(featherY), nyAbs);
            const xCover = float(1.0).sub(edgeX);
            const yCover = float(1.0).sub(edgeY);
            const inActiveFactor = xCover.mul(yCover);
            
            // Generate TV static pattern using hash for true per-pixel randomness
            // Use quantized time for authentic static that updates in discrete frames
            const quantizedTime = floor(this.timeUniform.mul(this.staticSpeedUniform));
            
            // Create unique seed for each pixel at each time step
            // Combine pixel coordinates and time into a single seed value
            // Large prime multipliers ensure no correlation between adjacent pixels
            const seed1 = pixelX.mul(uint(73856093))
                .bitXor(pixelY.mul(uint(19349663)))
                .bitXor(quantizedTime.toUint().mul(uint(83492791)));
            
            // Generate three independent random values using hash function
            // Each with a different offset to ensure independence
            const random = hash(seed1.toFloat());
            const random2 = hash(seed1.add(uint(1)).toFloat());
            const random3 = hash(seed1.add(uint(2)).toFloat());
            
            // TV static is typically black and white with sharp transitions
            const threshold = float(0.5);
            const staticValue = select(
                random.lessThan(threshold),
                float(0.0),  // Black
                float(1.0)   // White
            );
            
            // Mix in some gray values based on contrast parameter
            // staticContrast controls the ratio of pure black/white vs gray
            const finalValue = select(
                random2.lessThan(this.staticContrastUniform),
                random3,     // Gray value (using third random for variety)
                staticValue  // Pure black or white
            );
            
            // Add slight overall brightness variation for organic feel
            const brightness = finalValue.mul(float(0.9).add(random3.mul(0.1)));
            
            // Create monochrome color (same value for R, G, B)
            // Real TV static is grayscale
            const fullColor = vec3(brightness, brightness, brightness);
            
            // Extract single channel based on subpixel type
            const staticIntensity = select(
                subpixelIdx.equal(uint(0)), fullColor.r,
                select(subpixelIdx.equal(uint(1)), fullColor.g, fullColor.b)
            );

            // External target color (per logical pixel)
            const pixelIndex = pixelY.mul(uint(screenWidth)).add(pixelX);
            const targetFullColor = this.targetColors.element(pixelIndex);

            const contentUV = vec2(
                pixelX.toFloat().add(float(0.5)).div(screenWidthF),
                float(1.0).sub(pixelY.toFloat().add(float(0.5)).div(screenHeightF))
            );
            const contentSample = this.contentTextureNode.sample(contentUV);

            const overlayMask = clamp(contentSample.r, 0.0, 1.0)
                .mul(this.shaderEnabledUniform)
                .mul(this.shaderTypeUniform);
            const bufferColor = mix(targetFullColor, vec3(1.0).sub(targetFullColor), overlayMask);
            const bufferIntensity = select(
                subpixelIdx.equal(uint(0)), bufferColor.r,
                select(subpixelIdx.equal(uint(1)), bufferColor.g, bufferColor.b)
            );

            const contentIntensity = select(
                subpixelIdx.equal(uint(0)), contentSample.r,
                select(subpixelIdx.equal(uint(1)), contentSample.g, contentSample.b)
            );

            const useTexture = this.useExternalTextureUniform.greaterThan(float(0.5));
            const externalIntensity = select(useTexture, contentIntensity, bufferIntensity);
            const useExternal = this.useExternalContentUniform.greaterThan(float(0.5));
            const intensity = select(useExternal, externalIntensity, staticIntensity);
            
            // Create target color (single channel intensity, stored in all RGB for shader compatibility)
            const targetIntensity = intensity.add(this.minBrightnessUniform.mul(this.powerOnUniform));
            const targetColor = vec3(targetIntensity, targetIntensity, targetIntensity);
            
            // Continuous scan timing based on floating scan head
            const totalSubpixelsF = float(totalSubpixels);
            const scanHead = this.scanHeadUniform;
            const idxF = idx.toFloat();
            const delta = scanHead.sub(idxF);
            const deltaWrapped = select(
                delta.greaterThanEqual(float(0.0)),
                delta,
                delta.add(totalSubpixelsF)
            );
            const pixelsPerSecond = this.scanFramerateUniform.mul(totalSubpixelsF);
            const safePixelsPerSecond = max(pixelsPerSecond, float(0.0001));
            const timeSinceHit = deltaWrapped.div(safePixelsPerSecond);
            const baseDwell = float(1.0).div(safePixelsPerSecond);
            
            // Simulate the beam hitting the pixel and then decay
            // First: Apply attack with beamPixelDuration (fake longer dwell)
            // Then: Apply decay for remaining time based on position
            let finalColor = currentColor;
            
            // If in scan window: First attack to target, then decay
            const attackDiff = targetColor.sub(currentColor);
            
            // Beam current envelope and boost as active area collapses
            const onLevel = smoothstep(float(0.0), float(1.0), powerPhase);
            const offLevel = float(1.0).sub(smoothstep(float(0.0), float(1.0), powerPhase));
            const steadyLevel = select(this.powerOnUniform.greaterThan(float(0.5)), float(1.0), float(0.0));
            const transitionLevel = select(isPoweringOn, onLevel, offLevel);
            const baseLevel = select(isTransition, transitionLevel, steadyLevel);
            const onFlash = smoothstep(float(0.0), float(0.2), powerPhase)
                .mul(float(1.0).sub(smoothstep(float(0.2), float(0.6), powerPhase)));
            const offFlash = smoothstep(float(0.0), float(0.15), powerPhase)
                .mul(float(1.0).sub(smoothstep(float(0.15), float(0.4), powerPhase)));
            const flash = select(isPoweringOn, onFlash, offFlash);
            const flashBoost = float(1.0).add(this.powerFlashUniform.mul(flash));
            const warmupLevel = smoothstep(float(0.0), float(1.0), this.powerWarmupUniform);
            const offBoostMax = float(6.0);
            const offDurationScale = mix(float(1.0), offBoostMax, powerPhase);
            const durationScale = select(
                isTransition,
                select(isPoweringOn, float(1.0), offDurationScale),
                float(1.0)
            );
            const beamCurrent = baseLevel.mul(flashBoost).mul(warmupLevel).mul(durationScale);
            
            // Attack phase: Use beamPixelDuration to fake longer beam dwell
            // This is packed into the infinitesimal actual beam time
            const dwellTime = baseDwell.mul(this.beamPixelDurationUniform).mul(durationScale);
            const dwellHalf = dwellTime.mul(0.5);
            const beamStart = timeSinceHit.sub(dwellHalf);
            const beamEnd = timeSinceHit.add(dwellHalf);
            const shutterTime = max(this.deltaTimeUniform, float(0.000001));
            const overlapStart = max(float(0.0), beamStart);
            const overlapEnd = min(shutterTime, beamEnd);
            const overlapTime = clamp(overlapEnd.sub(overlapStart), 0.0, dwellTime);
            const exposureTime = overlapTime
                .mul(safePixelsPerSecond)
                .mul(this.deltaTimeUniform)
                .mul(inActiveFactor);
            const attackStrength = this.colorAttackUniform
                .mul(exposureTime)
                .mul(beamCurrent);
            const attackAlpha = clamp(
                float(1.0).sub(
                    pow(float(2.718281828), attackStrength.mul(float(-1.0)))
                ),
                0.0,
                1.0
            );
            finalColor = finalColor.add(attackDiff.mul(attackAlpha));
            
            // Decay phase: After beam passes, decay based on position
            // Pixels at start of scan (position 0) decay for full deltaTime
            // Pixels at end of scan (position 1) have minimal decay time
            const decayTarget = vec3(0.0, 0.0, 0.0);  // Always decay toward black
            const decayDiff = decayTarget.sub(finalColor);
            
            const decayAlpha = clamp(
                float(1.0).sub(
                    pow(float(2.718281828), this.colorDecayUniform.mul(this.deltaTimeUniform).mul(float(-1.0)))
                ),
                0.0,
                1.0
            );
            const newColor = finalColor.add(decayDiff.mul(decayAlpha));
            
            // Store result back
            this.currentColors.element(idx).assign(newColor);
        });
        
        // Create compute node for parallel execution
        this.colorComputeNode = computeShaderFn().compute(totalSubpixels);
    }
    
    private initPostProcessing(): void {
        if (!this.renderer || !this.scene || !this.camera) return;
        
        const scenePass = pass(this.scene, this.camera);
        // Set up MRT (Multiple Render Targets) for proper emissive handling
        scenePass.setMRT(mrt({ output, emissive }));
        
        // Get the base scene color
        const scenePassColor = scenePass.getTextureNode('output');
        const bloomCamera = this.bloomCamera ?? this.camera;
        const bloomPass = pass(this.scene, bloomCamera);
        bloomPass.setMRT(mrt({ output, emissive }));
        const bloomPassColor = bloomPass.getTextureNode('output');
        
        // Create bloom from the scene
        this.bloomNode = bloom(
            bloomPassColor,
            this.parameters.bloomStrength ?? 3.0,
            this.parameters.bloomRadius ?? 1.0,
            this.parameters.bloomThreshold ?? 0.01
        );
        
        this.postProcessing = new PostProcessing(this.renderer);
        // Add bloom to the base scene color (not replace it!)
        this.postProcessing.outputNode = scenePassColor.add(this.bloomNode);
    }

    private createCRTScreen(): void {
        if (!this.scene || !this.renderer || !this.currentColors) {
            console.warn('Cannot create CRT screen: missing required components');
            return;
        }
        
        const screenWidth = this.logicalWidth;
        const screenHeight = this.logicalHeight;
        const screenPhysicalWidth = this.getScreenPhysicalWidth();
        const screenPhysicalHeight = this.getScreenPhysicalHeight();
        const totalColumns = screenWidth * 3;
        const totalSubpixels = screenWidth * screenHeight * 3;
        
        console.log('Creating CRT screen:', {
            screenWidth,
            screenHeight,
            totalInstances: totalSubpixels
        });
        
        const geom = new THREE.PlaneGeometry(screenPhysicalWidth, screenPhysicalHeight, 1, 1);
        
        const mat = new MeshStandardNodeMaterial({
            transparent: false
        });
        
        // Avoid disappearing quads from backface culling or depth fighting
        mat.side = THREE.DoubleSide;
        mat.depthWrite = false;
        mat.depthTest = true;  // Keep depth testing on
        
        // CRT uniforms
        this.crtAmountUniform = uniform(this.parameters.crtAmount ?? 0.5, 'float');
        this.crtBarrelUniform = uniform(this.parameters.crtBarrel ?? -0.07, 'float');
        this.crtKeystoneXUniform = uniform(this.parameters.crtKeystoneX ?? 0.0, 'float');
        this.crtKeystoneYUniform = uniform(this.parameters.crtKeystoneY ?? 0.0, 'float');
        this.crtZoomUniform = uniform(this.parameters.crtZoom ?? 0.97, 'float');
        this.screenCurvatureUniform = uniform(this.parameters.screenCurvature ?? 0.0, 'float');
        this.brightnessUniform = uniform(this.parameters.brightness ?? 1.0, 'float');

        
        // Phosphor uniforms - these control the visual gap/black matrix
        this.slotDutyXUniform = uniform(this.parameters.slotDutyX ?? 0.65, 'float');  // horizontal fill
        this.slotDutyYUniform = uniform(this.parameters.slotDutyY ?? 0.85, 'float');  // vertical fill
        this.subpixelFeatherUniform = uniform(this.parameters.subpixelFeather ?? 0.08, 'float');  // anti-aliasing
        this.phosphorTintUniform = uniform(this.parameters.phosphorTint ?? 0.15, 'float');  // secondary brightness
        this.moireStrengthUniform = uniform(this.parameters.moireStrength ?? 1.0, 'float');
        this.moireChromaUniform = uniform(this.parameters.moireChroma ?? 1.0, 'float');
        this.moireFeatherUniform = uniform(this.parameters.moireFeather ?? 0.5, 'float');
        this.moireThresholdUniform = uniform(this.parameters.moireThreshold ?? 2.0, 'float');
        this.screenLightModeUniform = uniform(0.0, 'float');
        
        // Beam physics uniforms
        this.beamGammaUniform = uniform(this.parameters.beamGamma ?? 1.6, 'float');
        this.beamSpreadUniform = uniform(this.parameters.beamSpread ?? 1.3, 'float');
        this.vignetteStrengthUniform = uniform(this.parameters.vignetteStrength ?? 0.1, 'float');
        this.phaseShearAmountUniform = uniform(this.parameters.phaseShearAmount ?? 0.0, 'float');
        
        mat.colorNode = vec4(0, 0, 0, 1);

        const totalColumnsU = uint(totalColumns);
        const totalColumnsF = float(totalColumns);
        const screenHeightF = float(screenHeight);

        mat.emissiveNode = Fn(() => {
            const uvNode = uv();
            const nx = uvNode.x.mul(float(2.0)).sub(float(1.0));
            const ny = uvNode.y.mul(float(2.0)).sub(float(1.0));

            const screenK = this.screenCurvatureUniform;
            const r2Screen = nx.mul(nx).add(ny.mul(ny));
            const r4Screen = r2Screen.mul(r2Screen);
            const screenFacRaw = float(1.0).add(screenK.mul(r2Screen)).add(screenK.mul(float(0.25)).mul(r4Screen));
            const screenFac = clamp(screenFacRaw, 0.2, 5.0);
            const nxScreen = nx.mul(screenFac);
            const nyScreen = ny.mul(screenFac);
            const uvScreen = vec2(nxScreen.mul(0.5).add(0.5), nyScreen.mul(0.5).add(0.5));
            const edgeSoft = float(0.002);
            const insideX = smoothstep(float(0.0), edgeSoft, uvScreen.x)
                .mul(smoothstep(float(0.0), edgeSoft, float(1.0).sub(uvScreen.x)));
            const insideY = smoothstep(float(0.0), edgeSoft, uvScreen.y)
                .mul(smoothstep(float(0.0), edgeSoft, float(1.0).sub(uvScreen.y)));
            const inBounds = insideX.mul(insideY);

            const projR2 = r2Screen;
            const projR4 = projR2.mul(projR2);
            const k = this.crtBarrelUniform;
            const facRaw = float(1.0).add(k.mul(projR2)).add(k.mul(float(0.25)).mul(projR4));
            const fac = clamp(facRaw, 0.2, 5.0);
            let dx = nxScreen.mul(fac);
            let dy = nyScreen.mul(fac);
            dx = dx.add(this.crtKeystoneXUniform.mul(nyScreen));
            dy = dy.add(this.crtKeystoneYUniform.mul(nxScreen));
            const invAmt = float(1.0).sub(this.crtAmountUniform);
            const nxFinalRaw = dx.mul(this.crtAmountUniform).add(nxScreen.mul(invAmt)).mul(this.crtZoomUniform);
            const nyFinalRaw = dy.mul(this.crtAmountUniform).add(nyScreen.mul(invAmt)).mul(this.crtZoomUniform);

            const cornerR2 = float(2.0);
            const cornerR4 = cornerR2.mul(cornerR2);
            const cornerFacRaw = float(1.0).add(k.mul(cornerR2)).add(k.mul(float(0.25)).mul(cornerR4));
            const cornerFac = clamp(cornerFacRaw, 0.2, 5.0);

            const cornerAX = float(1.0);
            const cornerAY = float(1.0);
            const cornerBX = float(-1.0);
            const cornerBY = float(1.0);
            const cornerCX = float(1.0);
            const cornerCY = float(-1.0);
            const cornerDX = float(-1.0);
            const cornerDY = float(-1.0);

            const cornerDxA = cornerAX.mul(cornerFac).add(this.crtKeystoneXUniform.mul(cornerAY));
            const cornerDyA = cornerAY.mul(cornerFac).add(this.crtKeystoneYUniform.mul(cornerAX));
            const cornerNxA = cornerDxA.mul(this.crtAmountUniform).add(cornerAX.mul(invAmt)).mul(this.crtZoomUniform);
            const cornerNyA = cornerDyA.mul(this.crtAmountUniform).add(cornerAY.mul(invAmt)).mul(this.crtZoomUniform);
            const cornerMaxA = max(abs(cornerNxA), abs(cornerNyA));

            const cornerDxB = cornerBX.mul(cornerFac).add(this.crtKeystoneXUniform.mul(cornerBY));
            const cornerDyB = cornerBY.mul(cornerFac).add(this.crtKeystoneYUniform.mul(cornerBX));
            const cornerNxB = cornerDxB.mul(this.crtAmountUniform).add(cornerBX.mul(invAmt)).mul(this.crtZoomUniform);
            const cornerNyB = cornerDyB.mul(this.crtAmountUniform).add(cornerBY.mul(invAmt)).mul(this.crtZoomUniform);
            const cornerMaxB = max(abs(cornerNxB), abs(cornerNyB));

            const cornerDxC = cornerCX.mul(cornerFac).add(this.crtKeystoneXUniform.mul(cornerCY));
            const cornerDyC = cornerCY.mul(cornerFac).add(this.crtKeystoneYUniform.mul(cornerCX));
            const cornerNxC = cornerDxC.mul(this.crtAmountUniform).add(cornerCX.mul(invAmt)).mul(this.crtZoomUniform);
            const cornerNyC = cornerDyC.mul(this.crtAmountUniform).add(cornerCY.mul(invAmt)).mul(this.crtZoomUniform);
            const cornerMaxC = max(abs(cornerNxC), abs(cornerNyC));

            const cornerDxD = cornerDX.mul(cornerFac).add(this.crtKeystoneXUniform.mul(cornerDY));
            const cornerDyD = cornerDY.mul(cornerFac).add(this.crtKeystoneYUniform.mul(cornerDX));
            const cornerNxD = cornerDxD.mul(this.crtAmountUniform).add(cornerDX.mul(invAmt)).mul(this.crtZoomUniform);
            const cornerNyD = cornerDyD.mul(this.crtAmountUniform).add(cornerDY.mul(invAmt)).mul(this.crtZoomUniform);
            const cornerMaxD = max(abs(cornerNxD), abs(cornerNyD));

            const cornerMax = max(max(cornerMaxA, cornerMaxB), max(cornerMaxC, cornerMaxD));
            const normScale = float(1.0).div(max(float(1.0), cornerMax));
            const nxFinal = nxFinalRaw.mul(normScale);
            const nyFinal = nyFinalRaw.mul(normScale);
            const uvProjection = vec2(nxFinal.mul(0.5).add(0.5), nyFinal.mul(0.5).add(0.5));
            const projInsideX = smoothstep(float(0.0), edgeSoft, uvProjection.x)
                .mul(smoothstep(float(0.0), edgeSoft, float(1.0).sub(uvProjection.x)));
            const projInsideY = smoothstep(float(0.0), edgeSoft, uvProjection.y)
                .mul(smoothstep(float(0.0), edgeSoft, float(1.0).sub(uvProjection.y)));
            const inBoundsProjection = projInsideX.mul(projInsideY);

            const uvSample = vec2(uvProjection.x, float(1.0).sub(uvProjection.y));
            const pixelCoord = uvSample.mul(vec2(float(screenWidth), screenHeightF));
            const clampedCoordX = clamp(pixelCoord.x, float(0.0), float(screenWidth - 1));
            const clampedCoordY = clamp(pixelCoord.y, float(0.0), screenHeightF.sub(float(1.0)));
            const pixelX0f = floor(clampedCoordX);
            const pixelY0f = floor(clampedCoordY);
            const pixelX1f = min(pixelX0f.add(float(1.0)), float(screenWidth - 1));
            const pixelY1f = min(pixelY0f.add(float(1.0)), screenHeightF.sub(float(1.0)));
            const fracX = clampedCoordX.sub(pixelX0f);
            const fracY = clampedCoordY.sub(pixelY0f);
            const pixelX0 = pixelX0f.toUint();
            const pixelY0 = pixelY0f.toUint();
            const pixelX1 = pixelX1f.toUint();
            const pixelY1 = pixelY1f.toUint();
            const idx00 = pixelY0.mul(totalColumnsU).add(pixelX0.mul(uint(3)));
            const idx10 = pixelY0.mul(totalColumnsU).add(pixelX1.mul(uint(3)));
            const idx01 = pixelY1.mul(totalColumnsU).add(pixelX0.mul(uint(3)));
            const idx11 = pixelY1.mul(totalColumnsU).add(pixelX1.mul(uint(3)));
            const redIntensity = mix(
                mix(this.currentColors.element(idx00).r, this.currentColors.element(idx10).r, fracX),
                mix(this.currentColors.element(idx01).r, this.currentColors.element(idx11).r, fracX),
                fracY
            );
            const greenIntensity = mix(
                mix(this.currentColors.element(idx00.add(uint(1))).r, this.currentColors.element(idx10.add(uint(1))).r, fracX),
                mix(this.currentColors.element(idx01.add(uint(1))).r, this.currentColors.element(idx11.add(uint(1))).r, fracX),
                fracY
            );
            const blueIntensity = mix(
                mix(this.currentColors.element(idx00.add(uint(2))).r, this.currentColors.element(idx10.add(uint(2))).r, fracX),
                mix(this.currentColors.element(idx01.add(uint(2))).r, this.currentColors.element(idx11.add(uint(2))).r, fracX),
                fracY
            );

            const subpixelCoord = uvScreen.mul(vec2(totalColumnsF, screenHeightF));
            const safeR2 = max(r2Screen, float(0.000001));
            const invLen = inversesqrt(safeR2);
            const dirPre = vec2(nxScreen, nyScreen).mul(invLen);
            const phaseShear = this.phaseShearAmountUniform.mul(sqrt(r2Screen)).mul(dirPre.x);
            const subpixelCoordSheared = vec2(subpixelCoord.x.add(phaseShear), subpixelCoord.y);

            const subpixelIndexX = floor(subpixelCoordSheared.x);
            const subpixelIndexY = floor(subpixelCoordSheared.y);
            const clampedX = clamp(subpixelIndexX, float(0.0), totalColumnsF.sub(float(1.0)));
            const clampedY = clamp(subpixelIndexY, float(0.0), screenHeightF.sub(float(1.0)));
            const columnIdx = clampedX.toUint();
            const subpixelIdx = columnIdx.mod(uint(3));
            const isRed = subpixelIdx.equal(uint(0));
            const isGreen = subpixelIdx.equal(uint(1));
            const isBlue = subpixelIdx.equal(uint(2));
            const beamCoord = uvProjection.mul(vec2(totalColumnsF, screenHeightF));
            const beamLocal = fract(beamCoord).sub(vec2(0.5, 0.5));
            const beamDist = sqrt(dot(beamLocal, beamLocal));
            const beamRadius = float(0.35).add(this.subpixelFeatherUniform.mul(float(2.5)));
            const beamWeight = smoothstep(beamRadius, float(0.0), beamDist);

            const channelIntensity = select(
                isRed,
                redIntensity,
                select(isGreen, greenIntensity, blueIntensity)
            ).mul(inBoundsProjection);
            const intensity = channelIntensity.mul(beamWeight);
            const tint = vec3(
                select(isRed, float(1.0), this.phosphorTintUniform),
                select(isGreen, float(1.0), this.phosphorTintUniform),
                select(isBlue, float(1.0), this.phosphorTintUniform)
            );

            const uvLocal = fract(subpixelCoordSheared).sub(vec2(0.5, 0.5));

            const dCoordX = max(abs(dFdx(subpixelCoordSheared.x)), abs(dFdy(subpixelCoordSheared.x)));
            const dCoordY = max(abs(dFdx(subpixelCoordSheared.y)), abs(dFdy(subpixelCoordSheared.y)));
            const gridFootprint = max(dCoordX, dCoordY);
            const moireFeather = clamp(this.moireFeatherUniform, 0.0, 2.0);
            const moireInput = gridFootprint.mul(this.moireThresholdUniform).add(moireFeather.mul(0.25));
            const moireBlendRaw = smoothstep(float(0.15), float(0.9), moireInput);
            const moireBlend = clamp(moireBlendRaw.mul(this.moireStrengthUniform), 0.0, 1.0);
            const chromaBlend = clamp(moireBlend.mul(this.moireChromaUniform), 0.0, 1.0);
            const subpixelColor = tint.mul(intensity);
            const triadColor = vec3(redIntensity, greenIntensity, blueIntensity).mul(inBoundsProjection).mul(beamWeight);
            const tintedColor = mix(subpixelColor, triadColor, chromaBlend);

            const slotDutyX = this.slotDutyXUniform;
            const slotDutyY = this.slotDutyYUniform;
            const feather = this.subpixelFeatherUniform;
            const adaptiveFeatherX = feather;
            const adaptiveFeatherY = feather;
            const maxPadX = float(0.5).sub(slotDutyX.mul(0.5));
            const maxPadY = float(0.5).sub(slotDutyY.mul(0.5));
            const coverageBoost = moireFeather.mul(moireBlend).mul(float(0.15));
            const coveragePadX = min(dCoordX.mul(moireBlend).mul(float(0.35)).add(coverageBoost), maxPadX);
            const coveragePadY = min(dCoordY.mul(moireBlend).mul(float(0.35)).add(coverageBoost), maxPadY);
            const halfSizeX = slotDutyX.mul(0.5).add(coveragePadX);
            const halfSizeY = slotDutyY.mul(0.5).add(coveragePadY);
            const distX = halfSizeX.sub(abs(uvLocal.x));
            const distY = halfSizeY.sub(abs(uvLocal.y));
            const maskX = smoothstep(float(0.0), adaptiveFeatherX, distX);
            const maskY = smoothstep(float(0.0), adaptiveFeatherY, distY);
            const coverBase = maskX.mul(maskY);
            const coverExpanded = mix(coverBase, float(1.0), moireBlend);
            const baseDuty = slotDutyX.mul(slotDutyY);
            const dutyMix = mix(baseDuty, float(1.0), moireBlend);
            const cover = coverExpanded.mul(baseDuty.div(max(dutyMix, float(0.0001))));

            const p = vec2(nxFinal, nyFinal);
            const rho2 = p.x.mul(p.x).add(p.y.mul(p.y));
            const cosTheta = inversesqrt(float(1.0).add(this.beamSpreadUniform.mul(rho2)));
            const gain = pow(cosTheta, this.beamGammaUniform);

            const vignetteAmount = float(1.0).sub(this.vignetteStrengthUniform);
            const vignette = mix(float(1.0), vignetteAmount, rho2);

            const lightPass = this.screenLightModeUniform;
            const baseColor = tintedColor.mul(cover);
            const finalColor = mix(baseColor, triadColor, lightPass);
            return finalColor.mul(gain).mul(vignette).mul(this.brightnessUniform).mul(inBounds);
        })();

        mat.opacityNode = float(1.0);

        const mesh = new THREE.Mesh(geom, mat);
        mesh.frustumCulled = false;
        
        // Cleanup old visualization
        if (this.screenMesh) {
            this.scene.remove(this.screenMesh);
            if (this.screenMesh.geometry) this.screenMesh.geometry.dispose();
            if (this.screenMesh.material) {
                (this.screenMesh.material as THREE.Material).dispose();
            }
        }
        
        this.screenMesh = mesh;
        this.screenMesh.layers.set(CRTScreenScene.BLOOM_LAYER);
        this.screenMesh.position.set(0, 0, 0);
        this.scene.add(this.screenMesh);
        
        console.log('CRT screen created with', totalSubpixels, 'subpixel samples');
    }
    
    // Stub for updating GPU target colors when content changes
    private updateGPUTargetColors(): void {
        if (this.targetColors && this.targetColorArray) {
            const storageAttribute = this.targetColors.value;
            if (storageAttribute && storageAttribute.array) {
                storageAttribute.array.set(this.targetColorArray);
                storageAttribute.needsUpdate = true;
            }
        }
    }
}
