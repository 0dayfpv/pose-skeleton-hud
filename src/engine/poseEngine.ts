import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type FaceLandmarkerOptions,
  type FaceLandmarkerResult,
  type HandLandmarkerOptions,
  type HandLandmarkerResult,
  type NormalizedLandmark,
  type PoseLandmarkerOptions,
  type PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";
import type {
  DelegateType,
  HandednessType,
  ModelVariant,
  Point3D,
  TrackedHand,
} from "./geometry";

export type VisionWasmFileset = Awaited<
  ReturnType<typeof FilesetResolver.forVisionTasks>
>;

export const DEFAULT_WASM_BASE_PATH = "/wasm";

const MAX_SAFE_TIMESTAMP_MS = Number.MAX_SAFE_INTEGER - 1_000_000_000;

let mediaPipeConsoleFilterInstalled = false;

/**
 * MediaPipe's Emscripten WASM runtime routes C++ stderr ABSL_LOG(INFO) messages
 * (such as "INFO: Created TensorFlow Lite XNNPACK delegate for CPU.") to
 * `console.error`. Redirect those benign C++ INFO/I0000 status lines to
 * `console.info` while preserving all genuine `console.error` calls.
 */
function ensureMediaPipeConsoleFilter(): void {
  if (mediaPipeConsoleFilterInstalled || typeof console === "undefined") {
    return;
  }
  mediaPipeConsoleFilterInstalled = true;
  const originalConsoleError = console.error.bind(console);
  const originalConsoleInfo =
    typeof console.info === "function"
      ? console.info.bind(console)
      : originalConsoleError;

  console.error = (...args: unknown[]): void => {
    if (args.length > 0 && typeof args[0] === "string") {
      const msg = args[0].trim();
      if (
        msg.startsWith("INFO: Created TensorFlow Lite") ||
        /^I\d{4}\s/.test(msg) ||
        /^W\d{4}\s.*feedback manager/i.test(msg)
      ) {
        originalConsoleInfo(...args);
        return;
      }
    }
    originalConsoleError(...args);
  };
}

export const MODEL_URLS: Readonly<Record<ModelVariant, string>> = {
  lite: "https://huggingface.co/litert-community/MediaPipe-PoseLandmarker-Lite/resolve/main/pose_landmarker_lite.task",
  full: "https://huggingface.co/litert-community/MediaPipe-PoseLandmarker-Full/resolve/main/pose_landmarker_full.task",
  heavy:
    "https://huggingface.co/litert-community/MediaPipe-PoseLandmarker-Heavy/resolve/main/pose_landmarker_heavy.task",
};

export const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

export interface PoseEngineOptions {
  model: ModelVariant;
  minPoseDetectionConfidence: number;
  minTrackingConfidence: number;
  minPosePresenceConfidence?: number;
  wasmBasePath?: string; // defaults to "/wasm"
  enableHands?: boolean;
  enableFace?: boolean;
}

export interface InferenceFrameResult {
  landmarks: Point3D[] | null;
  hands?: TrackedHand[];
  faceLandmarks?: Point3D[] | null;
  inferenceMs: number;
  delegate: DelegateType;
}

export interface PoseLandmarkerLike {
  setOptions(options: PoseLandmarkerOptions): Promise<void>;
  detectForVideo(
    videoFrame: HTMLVideoElement,
    timestamp: number,
  ): PoseLandmarkerResult;
  close(): void;
}

export interface HandLandmarkerLike {
  setOptions?(options: HandLandmarkerOptions): Promise<void>;
  detectForVideo(
    videoFrame: HTMLVideoElement,
    timestamp: number,
  ): HandLandmarkerResult;
  close(): void;
}

export interface FaceLandmarkerLike {
  setOptions?(options: FaceLandmarkerOptions): Promise<void>;
  detectForVideo(
    videoFrame: HTMLVideoElement,
    timestamp: number,
  ): FaceLandmarkerResult;
  close(): void;
}

export interface PoseEngineDeps {
  forVisionTasks?: (basePath?: string) => Promise<VisionWasmFileset>;
  createFromOptions?: (
    wasmFileset: VisionWasmFileset,
    options: PoseLandmarkerOptions,
  ) => Promise<PoseLandmarkerLike>;
  createHandFromOptions?: (
    wasmFileset: VisionWasmFileset,
    options: HandLandmarkerOptions,
  ) => Promise<HandLandmarkerLike>;
  createFaceFromOptions?: (
    wasmFileset: VisionWasmFileset,
    options: FaceLandmarkerOptions,
  ) => Promise<FaceLandmarkerLike>;
  now?: () => number;
}

export class PoseEngineCancelledError extends Error {
  constructor(message = "PoseEngine operation was superseded or disposed") {
    super(message);
    this.name = "PoseEngineCancelledError";
  }
}

export function isPoseEngineCancelledError(error: unknown): boolean {
  return (
    error instanceof PoseEngineCancelledError ||
    (error instanceof Error && error.name === "PoseEngineCancelledError")
  );
}

interface NormalizedEngineOptions {
  model: ModelVariant;
  minPoseDetectionConfidence: number;
  minTrackingConfidence: number;
  minPosePresenceConfidence: number;
  wasmBasePath: string;
}

interface ThresholdSnapshot {
  minPoseDetectionConfidence: number;
  minPosePresenceConfidence: number;
  minTrackingConfidence: number;
}

function clampConfidence(value: number, fallback: number): number {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  return Math.max(0, Math.min(1, value));
}

function normalizeWasmBasePath(wasmBasePath: string | undefined): string {
  const trimmed = wasmBasePath?.trim();
  if (!trimmed || trimmed.length === 0) {
    return DEFAULT_WASM_BASE_PATH;
  }
  return trimmed.replace(/\/+$/, "") || "/";
}

function safeClose(
  instance:
    | PoseLandmarkerLike
    | HandLandmarkerLike
    | FaceLandmarkerLike
    | null
    | undefined,
): void {
  if (!instance) {
    return;
  }
  try {
    instance.close();
  } catch {
    /* ignore cleanup errors on partial or already closed instances */
  }
}

function toPoint3DArray(raw: readonly NormalizedLandmark[]): Point3D[] {
  const len = raw.length;
  const out = new Array<Point3D>(len);
  for (let i = 0; i < len; i++) {
    const lm = raw[i];
    out[i] =
      lm.presence !== undefined
        ? {
            x: lm.x,
            y: lm.y,
            z: lm.z,
            visibility: lm.visibility,
            presence: lm.presence,
          }
        : {
            x: lm.x,
            y: lm.y,
            z: lm.z,
            visibility: lm.visibility,
          };
  }
  return out;
}

function extractTrackedHands(result: HandLandmarkerResult | undefined): TrackedHand[] {
  if (!result || !Array.isArray(result.landmarks) || result.landmarks.length === 0) {
    return [];
  }
  const hands: TrackedHand[] = [];
  for (let i = 0; i < result.landmarks.length; i++) {
    const rawLms = result.landmarks[i];
    if (!rawLms || rawLms.length < 21) {
      continue;
    }
    const category = result.handedness?.[i]?.[0];
    const rawName = category?.categoryName ?? category?.displayName ?? (i === 0 ? "Left" : "Right");
    const handedness: HandednessType =
      /right/i.test(rawName) ? "Right" : "Left";
    const score =
      category && Number.isFinite(category.score) ? category.score : 1;

    hands.push({
      handedness,
      score,
      landmarks: toPoint3DArray(rawLms),
    });
  }
  return hands;
}

export class PoseEngine {
  private readonly forVisionTasks: (
    basePath?: string,
  ) => Promise<VisionWasmFileset>;
  private readonly createFromOptions: (
    wasmFileset: VisionWasmFileset,
    options: PoseLandmarkerOptions,
  ) => Promise<PoseLandmarkerLike>;
  private readonly createHandFromOptions:
    | ((
        wasmFileset: VisionWasmFileset,
        options: HandLandmarkerOptions,
      ) => Promise<HandLandmarkerLike>)
    | null;
  private readonly createFaceFromOptions:
    | ((
        wasmFileset: VisionWasmFileset,
        options: FaceLandmarkerOptions,
      ) => Promise<FaceLandmarkerLike>)
    | null;
  private readonly now: () => number;

  private landmarker: PoseLandmarkerLike | null = null;
  private handLandmarker: HandLandmarkerLike | null = null;
  private faceLandmarker: FaceLandmarkerLike | null = null;
  private activeDelegate: DelegateType | null = null;
  private currentModel: ModelVariant = "lite";
  private options: NormalizedEngineOptions | null = null;
  private appliedThresholds: ThresholdSnapshot | null = null;
  private wasmFileset: VisionWasmFileset | null = null;
  private wasmFilesetBasePath: string | null = null;
  private loadToken = 0;
  private disposed = false;
  private lastTimestampMs = -1;
  private lastHandTimestampMs = -1;
  private lastFaceTimestampMs = -1;
  private gpuRuntimeFallbackPending = false;
  private lastError: Error | null = null;

  private enableHands = true;
  private enableFace = true;

  constructor(deps: PoseEngineDeps = {}) {
    ensureMediaPipeConsoleFilter();
    this.forVisionTasks =
      deps.forVisionTasks ??
      ((basePath) => FilesetResolver.forVisionTasks(basePath));
    this.createFromOptions =
      deps.createFromOptions ??
      ((wasmFileset, options) =>
        PoseLandmarker.createFromOptions(wasmFileset, options));

    let defaultHandFactory:
      | ((
          wasmFileset: VisionWasmFileset,
          options: HandLandmarkerOptions,
        ) => Promise<HandLandmarkerLike>)
      | null = null;
    if (deps.createFromOptions === undefined) {
      try {
        if (typeof HandLandmarker?.createFromOptions === "function") {
          defaultHandFactory = (wasmFileset, options) =>
            HandLandmarker.createFromOptions(wasmFileset, options);
        }
      } catch {
        defaultHandFactory = null;
      }
    }

    let defaultFaceFactory:
      | ((
          wasmFileset: VisionWasmFileset,
          options: FaceLandmarkerOptions,
        ) => Promise<FaceLandmarkerLike>)
      | null = null;
    if (deps.createFromOptions === undefined) {
      try {
        if (typeof FaceLandmarker?.createFromOptions === "function") {
          defaultFaceFactory = (wasmFileset, options) =>
            FaceLandmarker.createFromOptions(wasmFileset, options);
        }
      } catch {
        defaultFaceFactory = null;
      }
    }

    this.createHandFromOptions = deps.createHandFromOptions ?? defaultHandFactory;
    this.createFaceFromOptions = deps.createFaceFromOptions ?? defaultFaceFactory;

    this.now = deps.now ?? (() => performance.now());
  }

  setDetailModules(config: { enableHands?: boolean; enableFace?: boolean }): void {
    if (config.enableHands !== undefined) {
      this.enableHands = Boolean(config.enableHands);
    }
    if (config.enableFace !== undefined) {
      this.enableFace = Boolean(config.enableFace);
    }
  }

  getDetailModules(): { enableHands: boolean; enableFace: boolean } {
    return {
      enableHands: this.enableHands,
      enableFace: this.enableFace,
    };
  }

  isHandEngineReady(): boolean {
    return !this.disposed && this.handLandmarker !== null;
  }

  isFaceEngineReady(): boolean {
    return !this.disposed && this.faceLandmarker !== null;
  }

  async init(options: PoseEngineOptions): Promise<DelegateType> {
    this.disposed = false;
    this.gpuRuntimeFallbackPending = false;
    this.lastError = null;

    if (options.enableHands !== undefined) {
      this.enableHands = Boolean(options.enableHands);
    }
    if (options.enableFace !== undefined) {
      this.enableFace = Boolean(options.enableFace);
    }

    const token = ++this.loadToken;
    const normalized = this.normalizeOptions(options);
    this.options = normalized;
    this.currentModel = normalized.model;

    this.disposeActiveLandmarker();

    try {
      const fileset = await this.resolveWasmFileset(
        normalized.wasmBasePath,
        token,
      );
      const { instance, delegate, createdOptions } = await this.loadWithFallback(
        normalized,
        fileset,
        token,
      );

      if (!this.attachLandmarker(instance, delegate, createdOptions, token)) {
        throw new PoseEngineCancelledError();
      }

      // Initialize HandLandmarker and FaceLandmarker before resolving init()
      // so all .task assets are ready and cached when app enters "ready" state.
      await this.initAuxiliaryLandmarkers(fileset, delegate, token);

      return delegate;
    } catch (error) {
      if (!isPoseEngineCancelledError(error)) {
        this.lastError =
          error instanceof Error ? error : new Error(String(error));
      }
      throw error;
    }
  }

  async switchModel(model: ModelVariant): Promise<DelegateType> {
    if (this.disposed || !this.options) {
      throw new Error(
        "PoseEngine must be initialized and not disposed before switching models",
      );
    }

    this.gpuRuntimeFallbackPending = false;
    this.lastError = null;

    const token = ++this.loadToken;
    const nextOptions: NormalizedEngineOptions = {
      ...this.options,
      model,
    };
    this.options = nextOptions;
    this.currentModel = model;

    this.disposeActiveLandmarker();

    try {
      const fileset = await this.resolveWasmFileset(
        nextOptions.wasmBasePath,
        token,
      );
      const { instance, delegate, createdOptions } = await this.loadWithFallback(
        nextOptions,
        fileset,
        token,
      );

      if (!this.attachLandmarker(instance, delegate, createdOptions, token)) {
        throw new PoseEngineCancelledError();
      }
      return delegate;
    } catch (error) {
      if (!isPoseEngineCancelledError(error)) {
        this.lastError =
          error instanceof Error ? error : new Error(String(error));
      }
      throw error;
    }
  }

  updateThresholds(
    minPoseDetectionConfidence: number,
    minTrackingConfidence: number,
  ): void {
    if (this.disposed || !this.options) {
      return;
    }

    const nextDetection = clampConfidence(
      minPoseDetectionConfidence,
      this.options.minPoseDetectionConfidence,
    );
    const nextTracking = clampConfidence(
      minTrackingConfidence,
      this.options.minTrackingConfidence,
    );

    const nextDesired: ThresholdSnapshot = {
      minPoseDetectionConfidence: nextDetection,
      minPosePresenceConfidence: nextDetection,
      minTrackingConfidence: nextTracking,
    };

    const matchesDesired = PoseEngine.thresholdsEqual(nextDesired, this.options);
    const matchesApplied =
      this.appliedThresholds !== null &&
      PoseEngine.thresholdsEqual(nextDesired, this.appliedThresholds);

    if (matchesDesired && (!this.landmarker || matchesApplied)) {
      return;
    }

    if (!matchesDesired) {
      this.options = {
        ...this.options,
        ...nextDesired,
      };
    }

    if (!this.landmarker) {
      return;
    }

    this.applyThresholdsToLandmarker(this.landmarker, nextDesired);
  }

  detectFrame(
    video: HTMLVideoElement,
    timestampMs: number,
  ): InferenceFrameResult {
    if (this.disposed || !this.landmarker || !this.activeDelegate) {
      return {
        landmarks: null,
        inferenceMs: 0,
        delegate: this.activeDelegate ?? "GPU",
      };
    }

    if (
      !video ||
      video.readyState < 2 ||
      video.videoWidth <= 0 ||
      video.videoHeight <= 0
    ) {
      return {
        landmarks: null,
        inferenceMs: 0,
        delegate: this.activeDelegate,
      };
    }

    let candidateTs: number;
    if (
      Number.isFinite(timestampMs) &&
      timestampMs >= 0 &&
      timestampMs <= MAX_SAFE_TIMESTAMP_MS
    ) {
      candidateTs = timestampMs;
    } else {
      const fallbackNow = this.now();
      candidateTs =
        Number.isFinite(fallbackNow) &&
        fallbackNow >= 0 &&
        fallbackNow <= MAX_SAFE_TIMESTAMP_MS
          ? fallbackNow
          : 0;
    }

    const safeTimestampMs =
      candidateTs > this.lastTimestampMs
        ? candidateTs
        : this.lastTimestampMs + 1;
    this.lastTimestampMs = safeTimestampMs;

    const delegate = this.activeDelegate;
    const frameToken = this.loadToken;
    const t0 = this.now();
    let result: PoseLandmarkerResult | undefined;

    try {
      result = this.landmarker.detectForVideo(video, safeTimestampMs);

      let hands: TrackedHand[] = [];
      if (this.enableHands && this.handLandmarker) {
        try {
          const safeHandTs =
            safeTimestampMs > this.lastHandTimestampMs
              ? safeTimestampMs
              : this.lastHandTimestampMs + 1;
          this.lastHandTimestampMs = safeHandTs;
          const handRes = this.handLandmarker.detectForVideo(video, safeHandTs);
          hands = extractTrackedHands(handRes);
        } catch {
          hands = [];
        }
      }

      let faceLandmarks: Point3D[] | null = null;
      if (this.enableFace && this.faceLandmarker) {
        try {
          const safeFaceTs =
            safeTimestampMs > this.lastFaceTimestampMs
              ? safeTimestampMs
              : this.lastFaceTimestampMs + 1;
          this.lastFaceTimestampMs = safeFaceTs;
          const faceRes = this.faceLandmarker.detectForVideo(video, safeFaceTs);
          const rawFace = faceRes?.faceLandmarks?.[0];
          if (rawFace && rawFace.length > 0) {
            faceLandmarks = toPoint3DArray(rawFace);
          }
        } catch {
          faceLandmarks = null;
        }
      }

      const t1 = this.now();
      const rawDuration = t1 - t0;
      const inferenceMs = Number.isFinite(rawDuration)
        ? Math.max(0, rawDuration)
        : 0;

      const rawPose = result?.landmarks?.[0];
      const landmarks =
        rawPose && rawPose.length > 0 ? toPoint3DArray(rawPose) : null;

      return {
        landmarks,
        ...(hands.length > 0 ? { hands } : {}),
        ...(faceLandmarks !== null ? { faceLandmarks } : {}),
        inferenceMs,
        delegate,
      };
    } catch (error) {
      if (frameToken === this.loadToken && !this.disposed) {
        this.lastError =
          error instanceof Error ? error : new Error(String(error));
      }
      if (
        delegate === "GPU" &&
        frameToken === this.loadToken &&
        !this.gpuRuntimeFallbackPending &&
        this.options &&
        !this.disposed
      ) {
        this.triggerRuntimeCpuFallback();
      }
      return {
        landmarks: null,
        inferenceMs: 0,
        delegate,
      };
    } finally {
      try {
        result?.close?.();
      } catch {
        /* ignore result cleanup errors */
      }
    }
  }

  dispose(): void {
    this.disposed = true;
    this.loadToken++;
    this.gpuRuntimeFallbackPending = false;
    this.disposeActiveLandmarker();
    this.disposeAuxiliaryLandmarkers();
    this.wasmFileset = null;
    this.wasmFilesetBasePath = null;
  }

  isReady(): boolean {
    return (
      !this.disposed &&
      this.landmarker !== null &&
      this.activeDelegate !== null
    );
  }

  getActiveDelegate(): DelegateType | null {
    return this.disposed ? null : this.activeDelegate;
  }

  getCurrentModel(): ModelVariant {
    return this.currentModel;
  }

  getLastError(): Error | null {
    return this.lastError;
  }

  private normalizeOptions(
    options: PoseEngineOptions,
  ): NormalizedEngineOptions {
    const detection = clampConfidence(options.minPoseDetectionConfidence, 0.5);
    return {
      model: options.model,
      minPoseDetectionConfidence: detection,
      minTrackingConfidence: clampConfidence(
        options.minTrackingConfidence,
        0.5,
      ),
      minPosePresenceConfidence: clampConfidence(
        options.minPosePresenceConfidence ?? detection,
        detection,
      ),
      wasmBasePath: normalizeWasmBasePath(options.wasmBasePath),
    };
  }

  private isStale(token: number): boolean {
    return this.disposed || token !== this.loadToken;
  }

  private disposeActiveLandmarker(): void {
    const prev = this.landmarker;
    this.landmarker = null;
    this.activeDelegate = null;
    this.appliedThresholds = null;
    this.lastTimestampMs = -1;
    safeClose(prev);
  }

  private disposeAuxiliaryLandmarkers(): void {
    const prevHand = this.handLandmarker;
    const prevFace = this.faceLandmarker;
    this.handLandmarker = null;
    this.faceLandmarker = null;
    this.lastHandTimestampMs = -1;
    this.lastFaceTimestampMs = -1;
    safeClose(prevHand);
    safeClose(prevFace);
  }

  private async initAuxiliaryLandmarkers(
    fileset: VisionWasmFileset,
    preferredDelegate: DelegateType,
    token: number,
  ): Promise<void> {
    if (this.isStale(token)) {
      return;
    }

    if (this.createHandFromOptions && !this.handLandmarker) {
      await this.loadHandWithFallback(fileset, preferredDelegate, token).catch(
        () => {
          /* graceful fallback if auxiliary hand model is unavailable */
        },
      );
    }

    if (this.isStale(token)) {
      return;
    }

    if (this.createFaceFromOptions && !this.faceLandmarker) {
      await this.loadFaceWithFallback(fileset, preferredDelegate, token).catch(
        () => {
          /* graceful fallback if auxiliary face model is unavailable */
        },
      );
    }
  }

  private async loadHandWithFallback(
    fileset: VisionWasmFileset,
    preferredDelegate: DelegateType,
    token: number,
  ): Promise<void> {
    if (!this.createHandFromOptions || this.isStale(token)) {
      return;
    }
    const delegates: DelegateType[] =
      preferredDelegate === "GPU" ? ["GPU", "CPU"] : ["CPU"];
    for (const delegate of delegates) {
      let instance: HandLandmarkerLike | null = null;
      try {
        instance = await this.createHandFromOptions(fileset, {
          baseOptions: {
            modelAssetPath: HAND_MODEL_URL,
            delegate,
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.45,
          minHandPresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
        });
        if (this.isStale(token)) {
          safeClose(instance);
          return;
        }
        this.handLandmarker = instance;
        this.lastHandTimestampMs = -1;
        return;
      } catch {
        safeClose(instance);
      }
    }
  }

  private async loadFaceWithFallback(
    fileset: VisionWasmFileset,
    preferredDelegate: DelegateType,
    token: number,
  ): Promise<void> {
    if (!this.createFaceFromOptions || this.isStale(token)) {
      return;
    }
    const delegates: DelegateType[] =
      preferredDelegate === "GPU" ? ["GPU", "CPU"] : ["CPU"];
    for (const delegate of delegates) {
      let instance: FaceLandmarkerLike | null = null;
      try {
        instance = await this.createFaceFromOptions(fileset, {
          baseOptions: {
            modelAssetPath: FACE_MODEL_URL,
            delegate,
          },
          runningMode: "VIDEO",
          numFaces: 1,
          minFaceDetectionConfidence: 0.45,
          minFacePresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
        });
        if (this.isStale(token)) {
          safeClose(instance);
          return;
        }
        this.faceLandmarker = instance;
        this.lastFaceTimestampMs = -1;
        return;
      } catch {
        safeClose(instance);
      }
    }
  }

  private async resolveWasmFileset(
    wasmBasePath: string,
    token: number,
  ): Promise<VisionWasmFileset> {
    if (this.wasmFileset && this.wasmFilesetBasePath === wasmBasePath) {
      if (this.isStale(token)) {
        throw new PoseEngineCancelledError();
      }
      return this.wasmFileset;
    }

    const fileset = await this.forVisionTasks(wasmBasePath);
    if (this.isStale(token)) {
      throw new PoseEngineCancelledError();
    }
    this.wasmFileset = fileset;
    this.wasmFilesetBasePath = wasmBasePath;
    return fileset;
  }

  private buildLandmarkerOptions(
    options: NormalizedEngineOptions,
    delegate: DelegateType,
  ): PoseLandmarkerOptions {
    return {
      baseOptions: {
        modelAssetPath: MODEL_URLS[options.model],
        delegate,
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: options.minPoseDetectionConfidence,
      minPosePresenceConfidence: options.minPosePresenceConfidence,
      minTrackingConfidence: options.minTrackingConfidence,
      outputSegmentationMasks: false,
    };
  }

  private async tryCreateDelegate(
    fileset: VisionWasmFileset,
    options: NormalizedEngineOptions,
    delegate: DelegateType,
    token: number,
  ): Promise<PoseLandmarkerLike> {
    let instance: PoseLandmarkerLike | null = null;
    try {
      instance = await this.createFromOptions(
        fileset,
        this.buildLandmarkerOptions(options, delegate),
      );
      if (this.isStale(token)) {
        throw new PoseEngineCancelledError();
      }
      return instance;
    } catch (error) {
      safeClose(instance);
      if (this.isStale(token) || isPoseEngineCancelledError(error)) {
        throw new PoseEngineCancelledError();
      }
      throw error;
    }
  }

  private async loadWithFallback(
    options: NormalizedEngineOptions,
    fileset: VisionWasmFileset,
    token: number,
  ): Promise<{
    instance: PoseLandmarkerLike;
    delegate: DelegateType;
    createdOptions: ThresholdSnapshot;
  }> {
    const gpuAttemptOptions: NormalizedEngineOptions = this.options
      ? { ...this.options }
      : { ...options };
    try {
      const gpuInstance = await this.tryCreateDelegate(
        fileset,
        gpuAttemptOptions,
        "GPU",
        token,
      );
      return {
        instance: gpuInstance,
        delegate: "GPU",
        createdOptions: PoseEngine.toThresholdSnapshot(gpuAttemptOptions),
      };
    } catch (gpuError) {
      if (isPoseEngineCancelledError(gpuError) || this.isStale(token)) {
        throw new PoseEngineCancelledError();
      }
      const cpuAttemptOptions: NormalizedEngineOptions = this.options
        ? { ...this.options }
        : { ...options };
      const cpuInstance = await this.tryCreateDelegate(
        fileset,
        cpuAttemptOptions,
        "CPU",
        token,
      );
      return {
        instance: cpuInstance,
        delegate: "CPU",
        createdOptions: PoseEngine.toThresholdSnapshot(cpuAttemptOptions),
      };
    }
  }

  private triggerRuntimeCpuFallback(): void {
    if (!this.options || this.disposed) {
      return;
    }
    const token = ++this.loadToken;
    const wasmBasePath = this.options.wasmBasePath;
    this.gpuRuntimeFallbackPending = true;
    this.disposeActiveLandmarker();

    void this.runRuntimeCpuFallback(wasmBasePath, token)
      .catch((err) => {
        if (!isPoseEngineCancelledError(err) && !this.isStale(token)) {
          this.lastError = err instanceof Error ? err : new Error(String(err));
        }
      })
      .finally(() => {
        if (token === this.loadToken) {
          this.gpuRuntimeFallbackPending = false;
        }
      });
  }

  private async runRuntimeCpuFallback(
    wasmBasePath: string,
    token: number,
  ): Promise<void> {
    const fileset = await this.resolveWasmFileset(wasmBasePath, token);
    if (this.isStale(token) || !this.options) {
      throw new PoseEngineCancelledError();
    }
    const cpuOptions: NormalizedEngineOptions = { ...this.options };
    const cpuInstance = await this.tryCreateDelegate(
      fileset,
      cpuOptions,
      "CPU",
      token,
    );
    this.attachLandmarker(
      cpuInstance,
      "CPU",
      PoseEngine.toThresholdSnapshot(cpuOptions),
      token,
    );
  }

  private attachLandmarker(
    instance: PoseLandmarkerLike,
    delegate: DelegateType,
    createdOptions: ThresholdSnapshot,
    token: number,
  ): boolean {
    if (this.isStale(token)) {
      safeClose(instance);
      return false;
    }

    this.landmarker = instance;
    this.activeDelegate = delegate;
    this.appliedThresholds = { ...createdOptions };
    this.syncPendingThresholds();
    return true;
  }

  private syncPendingThresholds(): void {
    if (this.disposed || !this.landmarker || !this.options) {
      return;
    }
    const desired = PoseEngine.toThresholdSnapshot(this.options);
    if (
      this.appliedThresholds !== null &&
      PoseEngine.thresholdsEqual(desired, this.appliedThresholds)
    ) {
      return;
    }
    this.applyThresholdsToLandmarker(this.landmarker, desired);
  }

  private applyThresholdsToLandmarker(
    instance: PoseLandmarkerLike,
    target: ThresholdSnapshot,
  ): void {
    const prevApplied = this.appliedThresholds;
    const nextApplied: ThresholdSnapshot = { ...target };
    this.appliedThresholds = nextApplied;

    const rollback = (): void => {
      if (
        this.landmarker === instance &&
        this.appliedThresholds === nextApplied
      ) {
        this.appliedThresholds = prevApplied;
      }
    };

    try {
      const maybePromise = instance.setOptions({
        minPoseDetectionConfidence: nextApplied.minPoseDetectionConfidence,
        minPosePresenceConfidence: nextApplied.minPosePresenceConfidence,
        minTrackingConfidence: nextApplied.minTrackingConfidence,
      });
      void Promise.resolve(maybePromise).catch(() => {
        rollback();
      });
    } catch {
      rollback();
    }
  }

  private static thresholdsEqual(
    a: ThresholdSnapshot,
    b: ThresholdSnapshot,
  ): boolean {
    return (
      a.minPoseDetectionConfidence === b.minPoseDetectionConfidence &&
      a.minPosePresenceConfidence === b.minPosePresenceConfidence &&
      a.minTrackingConfidence === b.minTrackingConfidence
    );
  }

  private static toThresholdSnapshot(
    options: NormalizedEngineOptions,
  ): ThresholdSnapshot {
    return {
      minPoseDetectionConfidence: options.minPoseDetectionConfidence,
      minPosePresenceConfidence: options.minPosePresenceConfidence,
      minTrackingConfidence: options.minTrackingConfidence,
    };
  }
}
