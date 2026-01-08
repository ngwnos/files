type WorkerGrid = {
  countX: number;
  countY: number;
  countZ: number;
  startX: number;
  startY: number;
  startZ: number;
  voxelSizeXZ: number;
  voxelHeight: number;
  basePlatformHeight: number;
  xEdgesOpen: boolean;
  zEdgesOpen: boolean;
  twoSided: boolean;
  gridX: number;
  gridZ: number;
  gridStartX: number;
  gridStartZ: number;
  step: number;
  instanceCount: number;
  paramA: Float32Array | ArrayBuffer;
  paramB: Float32Array | ArrayBuffer;
  paramC: Float32Array | ArrayBuffer;
  minHeightUnits: number;
  maxHeightUnits: number;
  minPedestalUnits: number;
  maxPedestalUnits: number;
  minPaddingUnits: number;
  maxPaddingUnits: number;
  minPedestalPaddingUnits: number;
  maxPedestalPaddingUnits: number;
  minPedestalTaperX: number;
  maxPedestalTaperX: number;
  minPedestalTaperZ: number;
  maxPedestalTaperZ: number;
  minTaperX: number;
  maxTaperX: number;
  minTaperZ: number;
  maxTaperZ: number;
  buildingExponent: number;
  pedestalExponent: number;
  paddingLimitCap: number;
  rollStep: number;
  twistStep: number;
};

type BuildMessage = {
  type: 'build';
  id: number;
  grid: WorkerGrid;
  maxPaths: number;
  maxTries: number;
  maxPathsPerVertex: number;
};

type CancelMessage = {
  type: 'cancel';
  id: number;
};

type InboundMessage = BuildMessage | CancelMessage;

type PathMessage = {
  type: 'path';
  id: number;
  path: Int32Array;
};

type DoneMessage = {
  type: 'done';
  id: number;
  placed: number;
  validCount: number;
  boundaryFlags?: Uint8Array;
};

type ErrorMessage = {
  type: 'error';
  id: number;
  message: string;
};

type DebugMessage = {
  type: 'debug';
  id: number;
  stage: 'start' | 'valid' | 'path' | 'done';
  detail: Record<string, number | string | boolean>;
};

type OutboundMessage = PathMessage | DoneMessage | ErrorMessage | DebugMessage;

type PathResult =
  | { status: 'found'; path: Int32Array }
  | { status: 'failed' }
  | { status: 'cancelled' };

const ctx: DedicatedWorkerGlobalScope = self as DedicatedWorkerGlobalScope;
let activeToken = 0;

// Post with optional transfer list to avoid extra copies.
const postMessageSafe = (message: OutboundMessage, transfer?: Transferable[]) => {
  if (transfer) {
    ctx.postMessage(message, transfer);
  } else {
    ctx.postMessage(message);
  }
};

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));
const mixValue = (a: number, b: number, t: number) => a + (b - a) * t;

// Select a start/end pair while respecting per-node usage limits.
const selectPair = (
  startPool: number[],
  endPool: number[],
  useSides: boolean,
  nodeUsage: Uint16Array,
  maxPathsPerVertex: number
): [number, number] | null => {
  if (startPool.length === 0 || endPool.length === 0) return null;
  const isBlocked = (value: number) => nodeUsage[value] >= maxPathsPerVertex;
  const pickUnblocked = (pool: number[], avoid: number | null) => {
    const maxPickAttempts = Math.max(10, pool.length * 2);
    for (let attempt = 0; attempt < maxPickAttempts; attempt += 1) {
      const value = pool[Math.floor(Math.random() * pool.length)];
      if (avoid !== null && value === avoid) continue;
      if (isBlocked(value)) continue;
      return value;
    }
    for (let i = 0; i < pool.length; i += 1) {
      const value = pool[i];
      if (avoid !== null && value === avoid) continue;
      if (isBlocked(value)) continue;
      return value;
    }
    return null;
  };

  const startIndex = pickUnblocked(startPool, null);
  if (startIndex === null) return null;
  if (useSides) {
    const endIndex = pickUnblocked(endPool, null);
    if (endIndex === null) return null;
    return [startIndex, endIndex];
  }
  const endIndex = pickUnblocked(endPool, startIndex);
  if (endIndex === null) return null;
  return [startIndex, endIndex];
};

const buildPaths = (message: BuildMessage, token: number) => {
  const { id, grid, maxPaths, maxTries } = message;
  const maxPathsPerVertex = Math.min(0xffff, Math.max(1, Math.round(message.maxPathsPerVertex)));
  const paramA =
    grid.paramA instanceof Float32Array ? grid.paramA : new Float32Array(grid.paramA);
  const paramB =
    grid.paramB instanceof Float32Array ? grid.paramB : new Float32Array(grid.paramB);
  const paramC =
    grid.paramC instanceof Float32Array ? grid.paramC : new Float32Array(grid.paramC);

  // Grid + procedural building settings.
  const {
    countX,
    countY,
    countZ,
    startX,
    startY,
    startZ,
    voxelSizeXZ,
    voxelHeight,
    basePlatformHeight,
    xEdgesOpen,
    zEdgesOpen,
    twoSided,
    gridX,
    gridZ,
    gridStartX,
    gridStartZ,
    step,
    instanceCount,
    minHeightUnits,
    maxHeightUnits,
    minPedestalUnits,
    maxPedestalUnits,
    minPaddingUnits,
    maxPaddingUnits,
    minPedestalPaddingUnits,
    maxPedestalPaddingUnits,
    minPedestalTaperX,
    maxPedestalTaperX,
    minPedestalTaperZ,
    maxPedestalTaperZ,
    minTaperX,
    maxTaperX,
    minTaperZ,
    maxTaperZ,
    buildingExponent,
    pedestalExponent,
    paddingLimitCap,
    rollStep,
    twistStep
  } = grid;

  const vertexCountX = countX + 1;
  const vertexCountY = countY + 1;
  const vertexCountZ = countZ + 1;
  const totalVertices = vertexCountX * vertexCountY * vertexCountZ;
  const boundaryFlags = new Uint8Array(totalVertices);
  const maxVertexX = vertexCountX - 1;
  const maxVertexZ = vertexCountZ - 1;
  const edgesOpenX = xEdgesOpen === true;
  const edgesOpenZ = zEdgesOpen === true;
  const baseVertexCount = totalVertices;
  const useHoleCenters = Math.abs(voxelSizeXZ - step) < 1e-6;
  const cellCount = gridX * gridZ;
  const holeCells: number[] = [];
  const holeIndexByCell = new Int32Array(cellCount);
  holeIndexByCell.fill(-1);
  const holeCenterBaseIndex = baseVertexCount;
  let holeCenterCount = 0;
  let totalNodeCount = baseVertexCount;
  const isBoundaryEdge = (ix: number, nx: number, iz: number, nz: number) =>
    (!edgesOpenX &&
      (ix === 0 || nx === 0 || ix === maxVertexX || nx === maxVertexX)) ||
    (!edgesOpenZ &&
      (iz === 0 || nz === 0 || iz === maxVertexZ || nz === maxVertexZ));
  const isBoundaryVertex = (ix: number, iz: number) =>
    (!edgesOpenX && (ix === 0 || ix === maxVertexX)) ||
    (!edgesOpenZ && (iz === 0 || iz === maxVertexZ));

  const perCell = twoSided ? 2 : 1;
  const instanceDirection = new Int8Array(instanceCount);
  const instanceBaseHalf = new Float32Array(instanceCount);
  const instanceBasePlatformHeight = new Float32Array(instanceCount);
  const instancePedestalHeight = new Float32Array(instanceCount);
  const instanceTowerHeight = new Float32Array(instanceCount);
  const instancePedestalBaseHalfX = new Float32Array(instanceCount);
  const instancePedestalBaseHalfZ = new Float32Array(instanceCount);
  const instancePedestalTopHalfX = new Float32Array(instanceCount);
  const instancePedestalTopHalfZ = new Float32Array(instanceCount);
  const instanceTowerBaseHalfX = new Float32Array(instanceCount);
  const instanceTowerBaseHalfZ = new Float32Array(instanceCount);
  const instanceTowerTopHalfX = new Float32Array(instanceCount);
  const instanceTowerTopHalfZ = new Float32Array(instanceCount);

  // Derive per-instance geometry from packed parameter arrays.
  for (let i = 0; i < instanceCount; i += 1) {
    const baseIndex = i * 4;
    const baseSize = Math.max(0.01, paramA[baseIndex]);
    const baseHalf = baseSize * 0.5;
    const paddingT = clamp01(paramA[baseIndex + 1]);
    const pedestalHeightT = clamp01(paramA[baseIndex + 2]);
    const towerHeightT = clamp01(paramA[baseIndex + 3]);
    const pedestalTaperXT = clamp01(paramB[baseIndex]);
    const pedestalTaperZT = clamp01(paramB[baseIndex + 1]);
    const taperXT = clamp01(paramB[baseIndex + 2]);
    const taperZT = clamp01(paramB[baseIndex + 3]);
    const basePlatformHeightValue = Math.max(0, paramC[baseIndex]);
    const pedestalPaddingT = clamp01(paramC[baseIndex + 1]);
    const directionValue = paramC[baseIndex + 3];
    if (Math.abs(directionValue) < 0.5) {
      instanceDirection[i] = 0;
      continue;
    }
    const side = directionValue >= 0 ? 1 : -1;
    instanceDirection[i] = side;
    instanceBaseHalf[i] = baseHalf;
    instanceBasePlatformHeight[i] = basePlatformHeightValue;

    const pedestalHeight = mixValue(
      minPedestalUnits,
      maxPedestalUnits,
      Math.pow(pedestalHeightT, pedestalExponent)
    );
    const towerHeight = mixValue(
      minHeightUnits,
      maxHeightUnits,
      Math.pow(towerHeightT, buildingExponent)
    );
    const pedestalPadding = mixValue(
      minPedestalPaddingUnits,
      maxPedestalPaddingUnits,
      pedestalPaddingT
    );
    const pedestalTaperX = mixValue(
      clamp01(minPedestalTaperX),
      clamp01(maxPedestalTaperX),
      pedestalTaperXT
    );
    const pedestalTaperZ = mixValue(
      clamp01(minPedestalTaperZ),
      clamp01(maxPedestalTaperZ),
      pedestalTaperZT
    );
    const taperX = mixValue(clamp01(minTaperX), clamp01(maxTaperX), taperXT);
    const taperZ = mixValue(clamp01(minTaperZ), clamp01(maxTaperZ), taperZT);

    const pedestalBaseX = Math.max(0.01, baseSize - pedestalPadding * 2);
    const pedestalBaseZ = Math.max(0.01, baseSize - pedestalPadding * 2);
    const pedestalTopX = pedestalBaseX * Math.max(0, 1 - pedestalTaperX);
    const pedestalTopZ = pedestalBaseZ * Math.max(0, 1 - pedestalTaperZ);
    const paddingLimit = Math.min(pedestalTopX, pedestalTopZ) * 0.49;
    const paddingCap = Math.max(0, Math.min(paddingLimit, paddingLimitCap));
    const paddingLower = Math.min(minPaddingUnits, paddingCap);
    const paddingUpper = Math.min(maxPaddingUnits, paddingCap);
    const padding = mixValue(paddingLower, paddingUpper, paddingT);
    const towerBaseX = Math.max(0.01, pedestalTopX - padding * 2);
    const towerBaseZ = Math.max(0.01, pedestalTopZ - padding * 2);
    const towerTopX = towerBaseX * Math.max(0, 1 - taperX);
    const towerTopZ = towerBaseZ * Math.max(0, 1 - taperZ);

    instancePedestalHeight[i] = pedestalHeight;
    instanceTowerHeight[i] = towerHeight;
    instancePedestalBaseHalfX[i] = pedestalBaseX * 0.5;
    instancePedestalBaseHalfZ[i] = pedestalBaseZ * 0.5;
    instancePedestalTopHalfX[i] = pedestalTopX * 0.5;
    instancePedestalTopHalfZ[i] = pedestalTopZ * 0.5;
    instanceTowerBaseHalfX[i] = towerBaseX * 0.5;
    instanceTowerBaseHalfZ[i] = towerBaseZ * 0.5;
    instanceTowerTopHalfX[i] = towerTopX * 0.5;
    instanceTowerTopHalfZ[i] = towerTopZ * 0.5;
  }

  if (useHoleCenters && cellCount > 0) {
    for (let cell = 0; cell < cellCount; cell += 1) {
      const baseInstance = cell * perCell;
      if (baseInstance >= instanceCount) continue;
      if (instanceDirection[baseInstance] !== 0) continue;
      holeIndexByCell[cell] = holeCells.length;
      holeCells.push(cell);
    }
    holeCenterCount = holeCells.length * vertexCountY;
    totalNodeCount = baseVertexCount + holeCenterCount;
  }
  const nodeUsage = new Uint16Array(totalNodeCount);

  postMessageSafe({
    type: 'debug',
    id,
    stage: 'start',
    detail: {
      countX,
      countY,
      countZ,
      vertexCountX,
      vertexCountY,
      vertexCountZ,
      totalVertices,
      holeCenterCount,
      maxPaths,
      maxTries,
      twoSided
    }
  });

  const neighborOffsets: Array<{ dx: number; dy: number; dz: number; delta: number; cost: number }> = [];
  const strideY = vertexCountX;
  const strideZ = vertexCountX * vertexCountY;
  // 26-neighborhood with anisotropic voxel cost.
  for (let dz = -1; dz <= 1; dz += 1) {
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        if (dx === 0 && dy === 0 && dz === 0) continue;
        const cost = Math.sqrt(
          (dx * voxelSizeXZ) * (dx * voxelSizeXZ) +
          (dy * voxelHeight) * (dy * voxelHeight) +
          (dz * voxelSizeXZ) * (dz * voxelSizeXZ)
        );
        neighborOffsets.push({ dx, dy, dz, delta: dx + dy * strideY + dz * strideZ, cost });
      }
    }
  }
  const cornerOffsets = [
    { dx: 0, dz: 0 },
    { dx: 1, dz: 0 },
    { dx: 0, dz: 1 },
    { dx: 1, dz: 1 }
  ];

  const halfStep = step * 0.5;
  const axisRelY = -basePlatformHeight;
  const axisRelZ = halfStep;
  const buildingRowOffsetsY = new Float32Array(gridZ);
  const buildingRowOffsetsZ = new Float32Array(gridZ);
  const buildingRowCos = new Float32Array(gridZ);
  const buildingRowSin = new Float32Array(gridZ);
  const buildingRowTwistCos = new Float32Array(gridZ);
  const buildingRowTwistSin = new Float32Array(gridZ);
  if (gridZ > 0) {
    let rowOffsetY = 0;
    let rowOffsetZ = gridStartZ;
    for (let row = 0; row < gridZ; row += 1) {
      if (row > 0) {
        const prevAngle = rollStep * (row - 1);
        const prevPivotY = rowOffsetY + basePlatformHeight;
        const prevPivotZ = rowOffsetZ - halfStep;
        const hingeY = prevPivotY - step * Math.sin(prevAngle);
        const hingeZ = prevPivotZ + step * Math.cos(prevAngle);
        rowOffsetY = hingeY - basePlatformHeight;
        rowOffsetZ = hingeZ + halfStep;
      }
      const rollAngle = rollStep * row;
      const twistAngle = twistStep;
      buildingRowOffsetsY[row] = rowOffsetY;
      buildingRowOffsetsZ[row] = rowOffsetZ;
      buildingRowCos[row] = Math.cos(rollAngle);
      buildingRowSin[row] = Math.sin(rollAngle);
      buildingRowTwistCos[row] = Math.cos(twistAngle);
      buildingRowTwistSin[row] = Math.sin(twistAngle);
    }
  }

  const vertexStartX = startX - voxelSizeXZ * 0.5;
  const vertexStartY = startY - voxelHeight * 0.5;
  const vertexStartZ = startZ - voxelSizeXZ * 0.5;
  // Precompute world-space positions for grid vertices (with roll/twist).
  const holeCenterZ = new Float32Array(holeCells.length);
  for (let holeIndex = 0; holeIndex < holeCells.length; holeIndex += 1) {
    const cellIndex = holeCells[holeIndex];
    const cellX = cellIndex % gridX;
    const cellZ = Math.floor(cellIndex / gridX);
    holeCenterZ[holeIndex] = gridStartZ + cellZ * step;
  }
  const worldX = new Float32Array(totalNodeCount);
  const worldY = new Float32Array(totalNodeCount);
  const worldZ = new Float32Array(totalNodeCount);
  const rowStepValid = step > 0 && gridZ > 0;
  const vertexCellStartZ = gridStartZ - halfStep;
  for (let idx = 0; idx < totalVertices; idx += 1) {
    const ix = idx % vertexCountX;
    const iy = Math.floor(idx / strideY) % vertexCountY;
    const iz = Math.floor(idx / strideZ);
    const cityX = vertexStartX + ix * voxelSizeXZ;
    const cityY = vertexStartY + iy * voxelHeight;
    const cityZ = vertexStartZ + iz * voxelSizeXZ;
    if (!rowStepValid) {
      worldX[idx] = cityX;
      worldY[idx] = cityY;
      worldZ[idx] = cityZ;
      continue;
    }
    let rowIndex = Math.floor((cityZ - vertexCellStartZ) / step);
    if (rowIndex < 0) rowIndex = 0;
    if (rowIndex >= gridZ) rowIndex = gridZ - 1;
    const offsetY = buildingRowOffsetsY[rowIndex];
    const offsetZ = buildingRowOffsetsZ[rowIndex];
    const cosValue = buildingRowCos[rowIndex];
    const sinValue = buildingRowSin[rowIndex];
    const twistCosValue = buildingRowTwistCos[rowIndex];
    const twistSinValue = buildingRowTwistSin[rowIndex];
    const rowCenterZ = gridStartZ + rowIndex * step;
    const localZ = cityZ - rowCenterZ;
    const preY = cityY + offsetY;
    const preZ = localZ + offsetZ;
    const pivotY = offsetY + basePlatformHeight;
    const pivotZ = offsetZ - halfStep;
    const relY = preY - pivotY;
    const relZ = preZ - pivotZ;
    const rolledY = relY * cosValue - relZ * sinValue + pivotY;
    const rolledZ = relY * sinValue + relZ * cosValue + pivotZ;
    const axisY = axisRelY * cosValue - axisRelZ * sinValue + pivotY;
    const axisZ = axisRelY * sinValue + axisRelZ * cosValue + pivotZ;
    const axisDirY = -sinValue;
    const axisDirZ = cosValue;
    const vX = cityX;
    const vY = rolledY - axisY;
    const vZ = rolledZ - axisZ;
    const crossX = axisDirY * vZ - axisDirZ * vY;
    const crossY = axisDirZ * vX;
    const crossZ = -axisDirY * vX;
    const dotV = axisDirY * vY + axisDirZ * vZ;
    const vRotX = vX * twistCosValue + crossX * twistSinValue;
    const vRotY =
      vY * twistCosValue + crossY * twistSinValue + axisDirY * dotV * (1 - twistCosValue);
    const vRotZ =
      vZ * twistCosValue + crossZ * twistSinValue + axisDirZ * dotV * (1 - twistCosValue);
    worldX[idx] = vRotX;
    worldY[idx] = axisY + vRotY;
    worldZ[idx] = axisZ + vRotZ;
  }
  if (holeCenterCount > 0) {
    // Add hole-center nodes at cell centers for extra routing options.
    for (let holeIndex = 0; holeIndex < holeCells.length; holeIndex += 1) {
      const centerZ = holeCenterZ[holeIndex];
      const baseIndex = holeCenterBaseIndex + holeIndex * vertexCountY;
      for (let iy = 0; iy < vertexCountY; iy += 1) {
        const idx = baseIndex + iy;
        const cityY = vertexStartY + iy * voxelHeight;
        if (!rowStepValid) {
          worldX[idx] = gridStartX + (holeCells[holeIndex] % gridX) * step;
          worldY[idx] = cityY;
          worldZ[idx] = centerZ;
          continue;
        }
        let rowIndex = Math.floor((centerZ - vertexCellStartZ) / step);
        if (rowIndex < 0) rowIndex = 0;
        if (rowIndex >= gridZ) rowIndex = gridZ - 1;
        const offsetY = buildingRowOffsetsY[rowIndex];
        const offsetZ = buildingRowOffsetsZ[rowIndex];
        const cosValue = buildingRowCos[rowIndex];
        const sinValue = buildingRowSin[rowIndex];
        const twistCosValue = buildingRowTwistCos[rowIndex];
        const twistSinValue = buildingRowTwistSin[rowIndex];
        const rowCenterZ = gridStartZ + rowIndex * step;
        const localZ = centerZ - rowCenterZ;
        const preY = cityY + offsetY;
        const preZ = localZ + offsetZ;
        const pivotY = offsetY + basePlatformHeight;
        const pivotZ = offsetZ - halfStep;
        const relY = preY - pivotY;
        const relZ = preZ - pivotZ;
        const rolledY = relY * cosValue - relZ * sinValue + pivotY;
        const rolledZ = relY * sinValue + relZ * cosValue + pivotZ;
        const axisY = axisRelY * cosValue - axisRelZ * sinValue + pivotY;
        const axisZ = axisRelY * sinValue + axisRelZ * cosValue + pivotZ;
        const axisDirY = -sinValue;
        const axisDirZ = cosValue;
        const vX = gridStartX + (holeCells[holeIndex] % gridX) * step;
        const vY = rolledY - axisY;
        const vZ = rolledZ - axisZ;
        const crossX = axisDirY * vZ - axisDirZ * vY;
        const crossY = axisDirZ * vX;
        const crossZ = -axisDirY * vX;
        const dotV = axisDirY * vY + axisDirZ * vZ;
        const vRotX = vX * twistCosValue + crossX * twistSinValue;
        const vRotY =
          vY * twistCosValue + crossY * twistSinValue + axisDirY * dotV * (1 - twistCosValue);
        const vRotZ =
          vZ * twistCosValue + crossZ * twistSinValue + axisDirZ * dotV * (1 - twistCosValue);
        worldX[idx] = vRotX;
        worldY[idx] = axisY + vRotY;
        worldZ[idx] = axisZ + vRotZ;
      }
    }
  }

  const rowWorldMinZ = new Float32Array(gridZ);
  const rowWorldMaxZ = new Float32Array(gridZ);
  let rowWorldMonotonic = true;
  // Row bounds in world Z to prune collision checks.
  if (gridZ > 0) {
    const minCityX = vertexStartX;
    const maxCityX = vertexStartX + countX * voxelSizeXZ;
    const minCityY = vertexStartY;
    const maxCityY = vertexStartY + (countY - 0.5) * voxelHeight;
    for (let row = 0; row < gridZ; row += 1) {
      const offsetY = buildingRowOffsetsY[row];
      const offsetZ = buildingRowOffsetsZ[row];
      const sinValue = buildingRowSin[row];
      const cosValue = buildingRowCos[row];
      const twistCosValue = buildingRowTwistCos[row];
      const twistSinValue = buildingRowTwistSin[row];
      const pivotY = offsetY + basePlatformHeight;
      const pivotZ = offsetZ - halfStep;
      const axisY = axisRelY * cosValue - axisRelZ * sinValue + pivotY;
      const axisZ = axisRelY * sinValue + axisRelZ * cosValue + pivotZ;
      const axisDirY = -sinValue;
      const axisDirZ = cosValue;
      const z0 = offsetZ - halfStep;
      const z1 = offsetZ + halfStep;
      let minZ = Infinity;
      let maxZ = -Infinity;
      const xs = [minCityX, maxCityX];
      const ys = [minCityY, maxCityY];
      const zs = [z0, z1];
      for (const xValue of xs) {
        for (const yValue of ys) {
          for (const zValue of zs) {
            const relY = yValue - pivotY;
            const relZ = zValue - pivotZ;
            const rolledY = relY * cosValue - relZ * sinValue + pivotY;
            const rolledZ = relY * sinValue + relZ * cosValue + pivotZ;
            const vX = xValue;
            const vY = rolledY - axisY;
            const vZ = rolledZ - axisZ;
            const crossZ = -axisDirY * vX;
            const dotV = axisDirY * vY + axisDirZ * vZ;
            const vRotZ =
              vZ * twistCosValue +
              crossZ * twistSinValue +
              axisDirZ * dotV * (1 - twistCosValue);
            const twistedZ = axisZ + vRotZ;
            minZ = Math.min(minZ, twistedZ);
            maxZ = Math.max(maxZ, twistedZ);
          }
        }
      }
      rowWorldMinZ[row] = minZ;
      rowWorldMaxZ[row] = maxZ;
      if (
        row > 0 &&
        (rowWorldMinZ[row] < rowWorldMinZ[row - 1] - 1e-6 ||
          rowWorldMaxZ[row] < rowWorldMaxZ[row - 1] - 1e-6)
      ) {
        rowWorldMonotonic = false;
      }
    }
  }

  const EPS = 1e-6;
  const segmentIntersectsFrustum = (
    x0: number,
    y0: number,
    z0: number,
    x1: number,
    y1: number,
    z1: number,
    yMin: number,
    yMax: number,
    baseHalfX: number,
    baseHalfZ: number,
    topHalfX: number,
    topHalfZ: number
  ) => {
    // Conservative segment vs. tapered box (frustum) test in local space.
    if (yMax <= yMin) return false;
    if (baseHalfX <= 0 || baseHalfZ <= 0 || topHalfX <= 0 || topHalfZ <= 0) {
      return false;
    }
    const height = yMax - yMin;
    const slopeX = (topHalfX - baseHalfX) / height;
    const slopeZ = (topHalfZ - baseHalfZ) / height;
    const dx = x1 - x0;
    const dy = y1 - y0;
    const dz = z1 - z0;

    let t0 = 0;
    let t1 = 1;

    const clip = (nx: number, ny: number, nz: number, d: number) => {
      const denom = nx * dx + ny * dy + nz * dz;
      const num = d - (nx * x0 + ny * y0 + nz * z0);
      if (Math.abs(denom) < 1e-10) {
        if (num < 0) return false;
        return true;
      }
      const t = num / denom;
      if (denom > 0) {
        if (t < t1) t1 = t;
      } else {
        if (t > t0) t0 = t;
      }
      return t0 <= t1;
    };

    const rhs = (value: number) => value + EPS;
    if (!clip(0, 1, 0, rhs(yMax))) return false;
    if (!clip(0, -1, 0, rhs(-yMin))) return false;

    const xPlane = baseHalfX - slopeX * yMin;
    if (!clip(1, -slopeX, 0, rhs(xPlane))) return false;
    if (!clip(-1, -slopeX, 0, rhs(xPlane))) return false;

    const zPlane = baseHalfZ - slopeZ * yMin;
    if (!clip(0, -slopeZ, 1, rhs(zPlane))) return false;
    if (!clip(0, -slopeZ, -1, rhs(zPlane))) return false;

    return t1 >= 0 && t0 <= 1;
  };

  const segmentHitsInstance = (
    wx0: number,
    wy0: number,
    wz0: number,
    wx1: number,
    wy1: number,
    wz1: number,
    offsetX: number,
    offsetY: number,
    offsetZ: number,
    rollCos: number,
    rollSin: number,
    twistCos: number,
    twistSin: number,
    instanceIndex: number
  ) => {
    // Transform segment into instance space and test against platform/pedestal/tower frustums.
    const side = instanceDirection[instanceIndex];
    if (side === 0) return false;
    const baseHalf = instanceBaseHalf[instanceIndex];
    const basePlatformHeightValue = instanceBasePlatformHeight[instanceIndex];
    const pivotY = offsetY + basePlatformHeightValue;
    const pivotZ = offsetZ - baseHalf;
    const axisRelY = -basePlatformHeightValue;
    const axisRelZ = baseHalf;
    const axisY = axisRelY * rollCos - axisRelZ * rollSin + pivotY;
    const axisZ = axisRelY * rollSin + axisRelZ * rollCos + pivotZ;
    const axisDirY = -rollSin;
    const axisDirZ = rollCos;

    const vX0 = wx0;
    const vY0 = wy0 - axisY;
    const vZ0 = wz0 - axisZ;
    const crossX0 = axisDirY * vZ0 - axisDirZ * vY0;
    const crossY0 = axisDirZ * vX0;
    const crossZ0 = -axisDirY * vX0;
    const dot0 = axisDirY * vY0 + axisDirZ * vZ0;
    const untwistX0 = vX0 * twistCos - crossX0 * twistSin;
    const untwistY0 =
      vY0 * twistCos - crossY0 * twistSin + axisDirY * dot0 * (1 - twistCos);
    const untwistZ0 =
      vZ0 * twistCos - crossZ0 * twistSin + axisDirZ * dot0 * (1 - twistCos);
    const untwistedY0 = axisY + untwistY0;
    const untwistedZ0 = axisZ + untwistZ0;
    const relY0 = untwistedY0 - pivotY;
    const relZ0 = untwistedZ0 - pivotZ;
    const unrolledY0 = relY0 * rollCos + relZ0 * rollSin + pivotY;
    const unrolledZ0 = relZ0 * rollCos - relY0 * rollSin + pivotZ;
    const lx0 = untwistX0 - offsetX;
    const ly0 = unrolledY0 - offsetY;
    const lz0 = unrolledZ0 - offsetZ;

    const vX1 = wx1;
    const vY1 = wy1 - axisY;
    const vZ1 = wz1 - axisZ;
    const crossX1 = axisDirY * vZ1 - axisDirZ * vY1;
    const crossY1 = axisDirZ * vX1;
    const crossZ1 = -axisDirY * vX1;
    const dot1 = axisDirY * vY1 + axisDirZ * vZ1;
    const untwistX1 = vX1 * twistCos - crossX1 * twistSin;
    const untwistY1 =
      vY1 * twistCos - crossY1 * twistSin + axisDirY * dot1 * (1 - twistCos);
    const untwistZ1 =
      vZ1 * twistCos - crossZ1 * twistSin + axisDirZ * dot1 * (1 - twistCos);
    const untwistedY1 = axisY + untwistY1;
    const untwistedZ1 = axisZ + untwistZ1;
    const relY1 = untwistedY1 - pivotY;
    const relZ1 = untwistedZ1 - pivotZ;
    const unrolledY1 = relY1 * rollCos + relZ1 * rollSin + pivotY;
    const unrolledZ1 = relZ1 * rollCos - relY1 * rollSin + pivotZ;
    const lx1 = untwistX1 - offsetX;
    const ly1 = unrolledY1 - offsetY;
    const lz1 = unrolledZ1 - offsetZ;

    if (side > 0 && basePlatformHeightValue > 0) {
      const taperDeltaZ = -basePlatformHeightValue * Math.tan(rollStep);
      const bottomHalfZ = Math.max(0.001, baseHalf + taperDeltaZ);
      const topHalfZ = baseHalf;
      if (
        segmentIntersectsFrustum(
          lx0,
          ly0,
          lz0,
          lx1,
          ly1,
          lz1,
          0,
          basePlatformHeightValue,
          baseHalf,
          bottomHalfZ,
          baseHalf,
          topHalfZ
        )
      ) {
        return true;
      }
    }

    const pedestalHeight = instancePedestalHeight[instanceIndex];
    if (pedestalHeight > 0) {
      const baseY = side > 0 ? basePlatformHeightValue : 0;
      const topY = side > 0 ? basePlatformHeightValue + pedestalHeight : -pedestalHeight;
      const yMin = Math.min(baseY, topY);
      const yMax = Math.max(baseY, topY);
      const baseHalfX = instancePedestalBaseHalfX[instanceIndex];
      const baseHalfZ = instancePedestalBaseHalfZ[instanceIndex];
      const topHalfX = instancePedestalTopHalfX[instanceIndex];
      const topHalfZ = instancePedestalTopHalfZ[instanceIndex];
      const halfMinX = side > 0 ? baseHalfX : topHalfX;
      const halfMinZ = side > 0 ? baseHalfZ : topHalfZ;
      const halfMaxX = side > 0 ? topHalfX : baseHalfX;
      const halfMaxZ = side > 0 ? topHalfZ : baseHalfZ;
      if (
        segmentIntersectsFrustum(
          lx0,
          ly0,
          lz0,
          lx1,
          ly1,
          lz1,
          yMin,
          yMax,
          halfMinX,
          halfMinZ,
          halfMaxX,
          halfMaxZ
        )
      ) {
        return true;
      }
    }

    const towerHeight = instanceTowerHeight[instanceIndex];
    if (towerHeight > 0) {
      const baseY = side > 0 ? basePlatformHeightValue + pedestalHeight : -pedestalHeight;
      const topY =
        side > 0
          ? basePlatformHeightValue + pedestalHeight + towerHeight
          : -(pedestalHeight + towerHeight);
      const yMin = Math.min(baseY, topY);
      const yMax = Math.max(baseY, topY);
      const baseHalfX = instanceTowerBaseHalfX[instanceIndex];
      const baseHalfZ = instanceTowerBaseHalfZ[instanceIndex];
      const topHalfX = instanceTowerTopHalfX[instanceIndex];
      const topHalfZ = instanceTowerTopHalfZ[instanceIndex];
      const halfMinX = side > 0 ? baseHalfX : topHalfX;
      const halfMinZ = side > 0 ? baseHalfZ : topHalfZ;
      const halfMaxX = side > 0 ? topHalfX : baseHalfX;
      const halfMaxZ = side > 0 ? topHalfZ : baseHalfZ;
      if (
        segmentIntersectsFrustum(
          lx0,
          ly0,
          lz0,
          lx1,
          ly1,
          lz1,
          yMin,
          yMax,
          halfMinX,
          halfMinZ,
          halfMaxX,
          halfMaxZ
        )
      ) {
        return true;
      }
    }

    return false;
  };

  const cellStartX = gridStartX - halfStep;
  const findMinRow = (segmentMinZ: number) => {
    if (!rowWorldMonotonic) {
      return 0;
    }
    let low = 0;
    let high = gridZ - 1;
    let result = gridZ;
    while (low <= high) {
      const mid = (low + high) >> 1;
      if (rowWorldMaxZ[mid] >= segmentMinZ) {
        result = mid;
        high = mid - 1;
      } else {
        low = mid + 1;
      }
    }
    return result === gridZ ? gridZ : result;
  };

  const findMaxRow = (segmentMaxZ: number) => {
    if (!rowWorldMonotonic) {
      return gridZ - 1;
    }
    let low = 0;
    let high = gridZ - 1;
    let result = -1;
    while (low <= high) {
      const mid = (low + high) >> 1;
      if (rowWorldMinZ[mid] <= segmentMaxZ) {
        result = mid;
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }
    return result;
  };

  const segmentHitsBuildings = (
    x0: number,
    y0: number,
    z0: number,
    x1: number,
    y1: number,
    z1: number,
    wx0: number,
    wy0: number,
    wz0: number,
    wx1: number,
    wy1: number,
    wz1: number
  ) => {
    // Broad phase: find rows/cells the segment overlaps, then test instances.
    const minX = Math.min(x0, x1);
    const maxX = Math.max(x0, x1);

    let minCellX = Math.floor((minX - cellStartX) / step);
    let maxCellX = Math.floor((maxX - cellStartX) / step);
    if (maxCellX < 0 || minCellX >= gridX) {
      return false;
    }

    minCellX = Math.max(0, minCellX);
    maxCellX = Math.min(gridX - 1, maxCellX);

    const segmentMinZ = Math.min(wz0, wz1);
    const segmentMaxZ = Math.max(wz0, wz1);
    let minRow = findMinRow(segmentMinZ);
    let maxRow = findMaxRow(segmentMaxZ);

    if (!rowWorldMonotonic) {
      minRow = gridZ;
      maxRow = -1;
      for (let row = 0; row < gridZ; row += 1) {
        if (segmentMaxZ < rowWorldMinZ[row] || segmentMinZ > rowWorldMaxZ[row]) {
          continue;
        }
        if (row < minRow) minRow = row;
        if (row > maxRow) maxRow = row;
      }
    }

    if (minRow > maxRow || minRow >= gridZ || maxRow < 0) {
      return false;
    }

    minRow = Math.max(0, minRow);
    maxRow = Math.min(gridZ - 1, maxRow);

    for (let z = minRow; z <= maxRow; z += 1) {
      const rowIndex = z * gridX;
      const offsetY = buildingRowOffsetsY[z];
      const offsetZ = buildingRowOffsetsZ[z];
      const rollCos = buildingRowCos[z];
      const rollSin = buildingRowSin[z];
      const twistCos = buildingRowTwistCos[z];
      const twistSin = buildingRowTwistSin[z];
      for (let x = minCellX; x <= maxCellX; x += 1) {
        const cellIndex = rowIndex + x;
        const offsetX = gridStartX + x * step;
        const baseInstance = cellIndex * perCell;
        for (let i = 0; i < perCell; i += 1) {
          const instanceIndex = baseInstance + i;
          if (instanceIndex >= instanceCount) continue;
          if (
            segmentHitsInstance(
              wx0,
              wy0,
              wz0,
              wx1,
              wy1,
              wz1,
              offsetX,
              offsetY,
              offsetZ,
              rollCos,
              rollSin,
              twistCos,
              twistSin,
              instanceIndex
            )
          ) {
            return true;
          }
        }
      }
    }

    return false;
  };

  const cancelMask = 0x3fff;
  let cancelCounter = 0;
  const shouldCancel = () => (cancelCounter++ & cancelMask) === 0 && token !== activeToken;

  // Classify vertices by visibility and boundary status.
  const anyClearIndices: number[] = [];
  const boundaryIndices: number[] = [];
  const anyClearPositive: number[] = [];
  const anyClearNegative: number[] = [];
  const boundaryPositive: number[] = [];
  const boundaryNegative: number[] = [];

  for (let idx = 0; idx < totalVertices; idx += 1) {
    if (shouldCancel()) return;
    const ix = idx % vertexCountX;
    const iy = Math.floor(idx / strideY) % vertexCountY;
    const iz = Math.floor(idx / strideZ);
    const x0 = vertexStartX + ix * voxelSizeXZ;
    const y0 = vertexStartY + iy * voxelHeight;
    const z0 = vertexStartZ + iz * voxelSizeXZ;
    const wx0 = worldX[idx];
    const wy0 = worldY[idx];
    const wz0 = worldZ[idx];

    let hasClear = false;
    let hasBlocked = false;
    for (const neighbor of neighborOffsets) {
      const nx = ix + neighbor.dx;
      const ny = iy + neighbor.dy;
      const nz = iz + neighbor.dz;
      if (nx < 0 || ny < 0 || nz < 0 || nx >= vertexCountX || ny >= vertexCountY || nz >= vertexCountZ) {
        continue;
      }
      if (isBoundaryEdge(ix, nx, iz, nz)) {
        hasBlocked = true;
        continue;
      }
      const nIdx = idx + neighbor.delta;
      const x1 = x0 + neighbor.dx * voxelSizeXZ;
      const y1 = y0 + neighbor.dy * voxelHeight;
      const z1 = z0 + neighbor.dz * voxelSizeXZ;
      const blocked = segmentHitsBuildings(
        x0,
        y0,
        z0,
        x1,
        y1,
        z1,
        wx0,
        wy0,
        wz0,
        worldX[nIdx],
        worldY[nIdx],
        worldZ[nIdx]
      );
      if (blocked) {
        hasBlocked = true;
      } else {
        hasClear = true;
      }
      if (hasClear && hasBlocked) {
        break;
      }
    }
    if (holeCenterCount > 0) {
      for (let cellDx = 0; cellDx <= 1; cellDx += 1) {
        const cellX = ix - cellDx;
        if (cellX < 0 || cellX >= gridX) continue;
        for (let cellDz = 0; cellDz <= 1; cellDz += 1) {
          const cellZ = iz - cellDz;
          if (cellZ < 0 || cellZ >= gridZ) continue;
          const cellIndex = cellZ * gridX + cellX;
          const holeIndex = holeIndexByCell[cellIndex];
          if (holeIndex < 0) continue;
          hasClear = true;
          if (hasClear && hasBlocked) {
            break;
          }
        }
        if (hasClear && hasBlocked) {
          break;
        }
      }
    }

    if (!hasClear) continue;
    if (hasBlocked) {
      boundaryFlags[idx] = 1;
    }
    anyClearIndices.push(idx);
    if (hasBlocked) {
      boundaryIndices.push(idx);
    }
    if (twoSided) {
      if (y0 >= basePlatformHeight) {
        anyClearPositive.push(idx);
        if (hasBlocked) {
          boundaryPositive.push(idx);
        }
      } else {
        anyClearNegative.push(idx);
        if (hasBlocked) {
          boundaryNegative.push(idx);
        }
      }
    }
  }

  const useSides = twoSided;
  const startPool = useSides
    ? (boundaryPositive.length > 0 ? boundaryPositive : anyClearPositive)
    : (boundaryIndices.length > 0 ? boundaryIndices : anyClearIndices);
  const endPool = useSides
    ? (boundaryNegative.length > 0 ? boundaryNegative : anyClearNegative)
    : (boundaryIndices.length > 0 ? boundaryIndices : anyClearIndices);

  const validCount = useSides ? startPool.length + endPool.length : startPool.length;
  const boundaryCount = boundaryIndices.length;

  postMessageSafe({
    type: 'debug',
    id,
    stage: 'valid',
    detail: { validCount, boundaryCount }
  });
  const parents = new Int32Array(totalNodeCount);
  const closed = new Int32Array(totalNodeCount);
  const gScore = new Float32Array(totalNodeCount);
  const gStamp = new Int32Array(totalNodeCount);
  const heapIndices = new Int32Array(totalNodeCount);
  const heapScores = new Float32Array(totalNodeCount);
  let stamp = 1;

  const heapPush = (idx: number, score: number, size: number) => {
    let i = size;
    heapIndices[i] = idx;
    heapScores[i] = score;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (heapScores[parent] <= score) break;
      heapIndices[i] = heapIndices[parent];
      heapScores[i] = heapScores[parent];
      heapIndices[parent] = idx;
      heapScores[parent] = score;
      i = parent;
    }
    return size + 1;
  };

  const heapPop = (size: number) => {
    if (size === 0) return null;
    const idx = heapIndices[0];
    const score = heapScores[0];
    const lastIndex = size - 1;
    if (lastIndex > 0) {
      const lastIdx = heapIndices[lastIndex];
      const lastScore = heapScores[lastIndex];
      let i = 0;
      while (true) {
        const left = i * 2 + 1;
        const right = left + 1;
        if (left >= lastIndex) break;
        let child = left;
        if (right < lastIndex && heapScores[right] < heapScores[left]) {
          child = right;
        }
        if (heapScores[child] >= lastScore) break;
        heapIndices[i] = heapIndices[child];
        heapScores[i] = heapScores[child];
        i = child;
      }
      heapIndices[i] = lastIdx;
      heapScores[i] = lastScore;
    }
    return { idx, score, size: lastIndex };
  };

  const getNodeCoords = (idx: number) => {
    if (idx < holeCenterBaseIndex) {
      const ix = idx % vertexCountX;
      const iy = Math.floor(idx / strideY) % vertexCountY;
      const iz = Math.floor(idx / strideZ);
      return {
        x: vertexStartX + ix * voxelSizeXZ,
        y: vertexStartY + iy * voxelHeight,
        z: vertexStartZ + iz * voxelSizeXZ
      };
    }
    const holeVertexIndex = idx - holeCenterBaseIndex;
    const holeIndex = Math.floor(holeVertexIndex / vertexCountY);
    const iy = holeVertexIndex - holeIndex * vertexCountY;
    const cellIndex = holeCells[holeIndex];
    const cellX = cellIndex % gridX;
    const cellZ = Math.floor(cellIndex / gridX);
    return {
      x: gridStartX + cellX * step,
      y: vertexStartY + iy * voxelHeight,
      z: gridStartZ + cellZ * step
    };
  };

  const aStar = (startIndex: number, endIndex: number): PathResult => {
    // A* with reuseable arrays and per-node usage caps.
    stamp += 1;
    if (stamp >= 0x7ffffffe) {
      closed.fill(0);
      gStamp.fill(0);
      stamp = 1;
    }
    if (nodeUsage[startIndex] >= maxPathsPerVertex || nodeUsage[endIndex] >= maxPathsPerVertex) {
      return { status: 'failed' };
    }
    const endCoords = getNodeCoords(endIndex);
    const heuristic = (x: number, y: number, z: number) => {
      const dx = x - endCoords.x;
      const dy = y - endCoords.y;
      const dz = z - endCoords.z;
      return Math.sqrt(dx * dx + dy * dy + dz * dz);
    };
    const startCoords = getNodeCoords(startIndex);
    gScore[startIndex] = 0;
    gStamp[startIndex] = stamp;
    parents[startIndex] = startIndex;
    let heapSize = 0;
    heapSize = heapPush(startIndex, heuristic(startCoords.x, startCoords.y, startCoords.z), heapSize);

    let iterations = 0;
    while (heapSize > 0) {
      if ((iterations++ & cancelMask) === 0 && token !== activeToken) {
        return { status: 'cancelled' };
      }
      const popped = heapPop(heapSize);
      if (!popped) break;
      heapSize = popped.size;
      const idx = popped.idx;
      if (closed[idx] === stamp) {
        continue;
      }
      closed[idx] = stamp;
      if (idx === endIndex) {
        break;
      }

      if (idx < holeCenterBaseIndex) {
        const ix = idx % vertexCountX;
        const iy = Math.floor(idx / strideY) % vertexCountY;
        const iz = Math.floor(idx / strideZ);
        const x0 = vertexStartX + ix * voxelSizeXZ;
        const y0 = vertexStartY + iy * voxelHeight;
        const z0 = vertexStartZ + iz * voxelSizeXZ;
        const wx0 = worldX[idx];
        const wy0 = worldY[idx];
        const wz0 = worldZ[idx];
        const baseCost = gStamp[idx] === stamp ? gScore[idx] : Number.POSITIVE_INFINITY;

        for (const neighbor of neighborOffsets) {
          const nx = ix + neighbor.dx;
          const ny = iy + neighbor.dy;
          const nz = iz + neighbor.dz;
          if (nx < 0 || ny < 0 || nz < 0 || nx >= vertexCountX || ny >= vertexCountY || nz >= vertexCountZ) {
            continue;
          }
          if (isBoundaryEdge(ix, nx, iz, nz)) {
            continue;
          }
          const nIdx = idx + neighbor.delta;
          if (nodeUsage[nIdx] >= maxPathsPerVertex) continue;
          if (closed[nIdx] === stamp) continue;

          const x1 = x0 + neighbor.dx * voxelSizeXZ;
          const y1 = y0 + neighbor.dy * voxelHeight;
          const z1 = z0 + neighbor.dz * voxelSizeXZ;
          if (
            segmentHitsBuildings(
              x0,
              y0,
              z0,
              x1,
              y1,
              z1,
              wx0,
              wy0,
              wz0,
              worldX[nIdx],
              worldY[nIdx],
              worldZ[nIdx]
            )
          ) {
            continue;
          }

          const tentative = baseCost + neighbor.cost;
          if (gStamp[nIdx] !== stamp || tentative < gScore[nIdx] - 1e-6) {
            gScore[nIdx] = tentative;
            gStamp[nIdx] = stamp;
            parents[nIdx] = idx;
            heapSize = heapPush(nIdx, tentative + heuristic(x1, y1, z1), heapSize);
          }
        }

        if (holeCenterCount > 0) {
          for (let cellDx = 0; cellDx <= 1; cellDx += 1) {
            const cellX = ix - cellDx;
            if (cellX < 0 || cellX >= gridX) continue;
            for (let cellDz = 0; cellDz <= 1; cellDz += 1) {
              const cellZ = iz - cellDz;
              if (cellZ < 0 || cellZ >= gridZ) continue;
              const cellIndex = cellZ * gridX + cellX;
              const holeIndex = holeIndexByCell[cellIndex];
              if (holeIndex < 0) continue;
              const nIdx = holeCenterBaseIndex + holeIndex * vertexCountY + iy;
              if (nodeUsage[nIdx] >= maxPathsPerVertex) continue;
              if (closed[nIdx] === stamp) continue;
              const centerX = gridStartX + cellX * step;
              const centerZ = gridStartZ + cellZ * step;
              const dx = centerX - x0;
              const dz = centerZ - z0;
              const cost = Math.sqrt(dx * dx + dz * dz);
              const tentative = baseCost + cost;
              if (gStamp[nIdx] !== stamp || tentative < gScore[nIdx] - 1e-6) {
                gScore[nIdx] = tentative;
                gStamp[nIdx] = stamp;
                parents[nIdx] = idx;
                heapSize = heapPush(nIdx, tentative + heuristic(centerX, y0, centerZ), heapSize);
              }
            }
          }
        }
      } else {
        const holeVertexIndex = idx - holeCenterBaseIndex;
        const holeIndex = Math.floor(holeVertexIndex / vertexCountY);
        if (holeIndex < 0 || holeIndex >= holeCells.length) {
          continue;
        }
        const iy = holeVertexIndex - holeIndex * vertexCountY;
        const cellIndex = holeCells[holeIndex];
        const cellX = cellIndex % gridX;
        const cellZ = Math.floor(cellIndex / gridX);
        const x0 = gridStartX + cellX * step;
        const y0 = vertexStartY + iy * voxelHeight;
        const z0 = gridStartZ + cellZ * step;
        const baseCost = gStamp[idx] === stamp ? gScore[idx] : Number.POSITIVE_INFINITY;

        for (const corner of cornerOffsets) {
          const ix = cellX + corner.dx;
          const iz = cellZ + corner.dz;
          const nIdx = ix + iy * strideY + iz * strideZ;
          if (nodeUsage[nIdx] >= maxPathsPerVertex) continue;
          if (closed[nIdx] === stamp) continue;
          const x1 = vertexStartX + ix * voxelSizeXZ;
          const z1 = vertexStartZ + iz * voxelSizeXZ;
          const dx = x1 - x0;
          const dz = z1 - z0;
          const cost = Math.sqrt(dx * dx + dz * dz);
          const tentative = baseCost + cost;
          if (gStamp[nIdx] !== stamp || tentative < gScore[nIdx] - 1e-6) {
            gScore[nIdx] = tentative;
            gStamp[nIdx] = stamp;
            parents[nIdx] = idx;
            heapSize = heapPush(nIdx, tentative + heuristic(x1, y0, z1), heapSize);
          }
        }

        if (iy > 0) {
          const nIdx = idx - 1;
          if (nodeUsage[nIdx] >= maxPathsPerVertex) {
            continue;
          }
          if (closed[nIdx] !== stamp) {
            const cost = voxelHeight;
            const tentative = baseCost + cost;
            if (gStamp[nIdx] !== stamp || tentative < gScore[nIdx] - 1e-6) {
              gScore[nIdx] = tentative;
              gStamp[nIdx] = stamp;
              parents[nIdx] = idx;
              heapSize = heapPush(nIdx, tentative + heuristic(x0, y0 - voxelHeight, z0), heapSize);
            }
          }
        }

        if (iy + 1 < vertexCountY) {
          const nIdx = idx + 1;
          if (nodeUsage[nIdx] >= maxPathsPerVertex) {
            continue;
          }
          if (closed[nIdx] !== stamp) {
            const cost = voxelHeight;
            const tentative = baseCost + cost;
            if (gStamp[nIdx] !== stamp || tentative < gScore[nIdx] - 1e-6) {
              gScore[nIdx] = tentative;
              gStamp[nIdx] = stamp;
              parents[nIdx] = idx;
              heapSize = heapPush(nIdx, tentative + heuristic(x0, y0 + voxelHeight, z0), heapSize);
            }
          }
        }
      }
    }

    if (closed[endIndex] !== stamp) {
      return { status: 'failed' };
    }

    const path: number[] = [];
    let current = endIndex;
    let safety = 0;
    while (current !== startIndex && safety < totalNodeCount) {
      if (gStamp[current] !== stamp) {
        return { status: 'failed' };
      }
      path.push(current);
      current = parents[current];
      safety += 1;
    }
    if (current !== startIndex) {
      return { status: 'failed' };
    }
    path.push(startIndex);
    path.reverse();
    return { status: 'found', path: Int32Array.from(path) };
  };

  const pickDistinct = (
    pool: number[],
    count: number,
    nodeUsage: Uint16Array,
    maxPathsPerVertex: number
  ) => {
    // Randomly pick distinct nodes, with a fallback linear scan.
    if (pool.length < count) return null;
    const picks: number[] = [];
    const used = new Set<number>();
    const maxPickAttempts = Math.max(10, pool.length * 2);
    let attempts = 0;
    const isBlocked = (value: number) => nodeUsage[value] >= maxPathsPerVertex;
    while (picks.length < count && attempts < maxPickAttempts) {
      const value = pool[Math.floor(Math.random() * pool.length)];
      if (!used.has(value) && !isBlocked(value)) {
        used.add(value);
        picks.push(value);
      }
      attempts += 1;
    }
    if (picks.length < count) {
      for (let i = 0; i < pool.length && picks.length < count; i += 1) {
        const value = pool[i];
        if (!used.has(value) && !isBlocked(value)) {
          used.add(value);
          picks.push(value);
        }
      }
    }
    if (picks.length < count) return null;
    return picks;
  };

  const mergeSegments = (segments: Int32Array[]) => {
    const merged: number[] = [];
    for (let i = 0; i < segments.length; i += 1) {
      const segment = segments[i];
      const start = i === 0 ? 0 : 1;
      for (let j = start; j < segment.length; j += 1) {
        merged.push(segment[j]);
      }
    }
    return Int32Array.from(merged);
  };

  const blockPath = (path: Int32Array) => {
    // Increase usage to limit how many paths share a vertex.
    for (let i = 0; i < path.length; i += 1) {
      const idx = path[i];
      const next = nodeUsage[idx] + 1;
      nodeUsage[idx] = next >= maxPathsPerVertex ? maxPathsPerVertex : next;
    }
  };

  const buildLoop = (): PathResult => {
    // Build a closed loop across both sides (twoSided mode).
    const sideA = pickDistinct(startPool, 2, nodeUsage, maxPathsPerVertex);
    const sideB = pickDistinct(endPool, 2, nodeUsage, maxPathsPerVertex);
    if (!sideA || !sideB) {
      return { status: 'failed' };
    }
    const [a1, a2] = sideA;
    const [b1, b2] = sideB;
    const seg1 = aStar(a1, a2);
    if (seg1.status !== 'found') return seg1;
    const seg2 = aStar(a2, b1);
    if (seg2.status !== 'found') return seg2;
    const seg3 = aStar(b1, b2);
    if (seg3.status !== 'found') return seg3;
    const seg4 = aStar(b2, a1);
    if (seg4.status !== 'found') return seg4;
    const path = mergeSegments([seg1.path, seg2.path, seg3.path, seg4.path]);
    return { status: 'found', path };
  };

  if (startPool.length === 0 || endPool.length === 0) {
    postMessageSafe({
      type: 'debug',
      id,
      stage: 'done',
      detail: { placed: 0, validCount, attempts: 0, maxAttempts: 0 }
    });
    postMessageSafe({ type: 'done', id, placed: 0, validCount, boundaryFlags }, [
      boundaryFlags.buffer
    ]);
    return;
  }

  const useLoopPaths = useSides;
  if (useLoopPaths && (startPool.length < 2 || endPool.length < 2)) {
    postMessageSafe({
      type: 'debug',
      id,
      stage: 'done',
      detail: { placed: 0, validCount, attempts: 0, maxAttempts: 0 }
    });
    postMessageSafe({ type: 'done', id, placed: 0, validCount, boundaryFlags }, [
      boundaryFlags.buffer
    ]);
    return;
  }

  let placed = 0;
  let pairTries = 0;
  let pairStart = -1;
  let pairEnd = -1;
  let attempts = 0;
  const maxAttempts = Math.max(1, maxPaths * maxTries);

  while (placed < maxPaths) {
    if (token !== activeToken) return;
    if (attempts >= maxAttempts) break;
    let result: PathResult;
    if (useLoopPaths) {
      result = buildLoop();
    } else {
      if (pairStart === -1 || pairEnd === -1 || pairTries >= maxTries) {
        const pair = selectPair(startPool, endPool, useSides, nodeUsage, maxPathsPerVertex);
        if (!pair) break;
        pairStart = pair[0];
        pairEnd = pair[1];
        pairTries = 0;
      }
      result = aStar(pairStart, pairEnd);
    }
    if (result.status === 'cancelled') return;
    if (result.status === 'found') {
      const path = result.path;
      if (placed === 0) {
        postMessageSafe({
          type: 'debug',
          id,
          stage: 'path',
          detail: { pathLength: path.length }
        });
      }
      blockPath(path);
      postMessageSafe({ type: 'path', id, path }, [path.buffer]);
      placed += 1;
      if (!useLoopPaths) {
        pairStart = -1;
        pairEnd = -1;
        pairTries = 0;
      }
    } else {
      if (!useLoopPaths) {
        pairTries += 1;
      }
      attempts += 1;
    }
  }

  if (token !== activeToken) return;
  postMessageSafe({
    type: 'debug',
    id,
    stage: 'done',
    detail: { placed, validCount, attempts, maxAttempts }
  });
  postMessageSafe({ type: 'done', id, placed, validCount, boundaryFlags }, [
    boundaryFlags.buffer
  ]);
};

ctx.onmessage = (event: MessageEvent<InboundMessage>) => {
  const message = event.data;
  if (message.type === 'cancel') {
    activeToken += 1;
    return;
  }
  // Update cancellation token and start a new build.
  activeToken += 1;
  const token = activeToken;
  try {
    buildPaths(message, token);
  } catch (error) {
    const text = error instanceof Error ? error.message : String(error);
    postMessageSafe({ type: 'error', id: message.id, message: text });
  }
};
