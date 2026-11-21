import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  NormalizedLandmark,
  PoseLandmarkerOptions,
  PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";
import {
  MODEL_URLS,
  PoseEngine,
  PoseEngineCancelledError,
  isPoseEngineCancelledError,
  type PoseLandmarkerLike,
  type VisionWasmFileset,
} from "./poseEngine";
import type { DelegateType, Point3D } from "./geometry";

interface MockLandmarker extends PoseLandmarkerLike {
  detectForVideo: ReturnType<
    typeof vi.fn<
      (videoFrame: HTMLVideoElement, timestamp: number) => PoseLandmarkerResult
    >
  >;
  setOptions: ReturnType<
    typeof vi.fn<(options: PoseLandmarkerOptions) => Promise<void>>
  >;
  close: ReturnType<typeof vi.fn<() => void>>;
}

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
}

const mediaPipeMocks = vi.hoisted(() => ({
  forVisionTasks: vi.fn<
    (basePath?: string, useModule?: boolean) => Promise<VisionWasmFileset>
  >(),
  createFromOptions: vi.fn<
    (
      wasmFileset: VisionWasmFileset,
      options: PoseLandmarkerOptions,
    ) => Promise<PoseLandmarkerLike>
  >(),
}));

vi.mock("@mediapipe/tasks-vision", () => ({
  FilesetResolver: {
    forVisionTasks: mediaPipeMocks.forVisionTasks,
  },
  PoseLandmarker: {
    createFromOptions: mediaPipeMocks.createFromOptions,
  },
}));

const DEFAULT_FILESET: VisionWasmFileset = {
  wasmLoaderPath: "/wasm/vision_wasm_internal.js",
  wasmBinaryPath: "/wasm/vision_wasm_internal.wasm",
};

function createMockLandmarker(): MockLandmarker {
  return {
    detectForVideo: vi.fn<
      (videoFrame: HTMLVideoElement, timestamp: number) => PoseLandmarkerResult
    >(() => ({
      landmarks: [],
      worldLandmarks: [],
      close: vi.fn(),
    })),
    setOptions: vi
      .fn<(options: PoseLandmarkerOptions) => Promise<void>>()
      .mockResolvedValue(undefined),
    close: vi.fn<() => void>(),
  };
}

function createMockVideo(overrides?: {
  readyState?: number;
  videoWidth?: number;
  videoHeight?: number;
}): HTMLVideoElement {
  const video = document.createElement("video");
  Object.defineProperty(video, "readyState", {
    configurable: true,
    value: overrides?.readyState ?? 4,
  });
  Object.defineProperty(video, "videoWidth", {
    configurable: true,
    value: overrides?.videoWidth ?? 1280,
  });
  Object.defineProperty(video, "videoHeight", {
    configurable: true,
    value: overrides?.videoHeight ?? 720,
  });
  return video;
}

function createMockLandmarks(count = 33): NormalizedLandmark[] {
  return Array.from({ length: count }, (_, i) => ({
    x: 0.1 + i * 0.01,
    y: 0.2 + i * 0.01,
    z: -0.05 + i * 0.002,
    visibility: 0.9,
    presence: 0.95,
  }));
}

function createDeferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

async function flushMicrotasks(): Promise<void> {
  for (let i = 0; i < 8; i += 1) {
    await Promise.resolve();
  }
}

describe("PoseEngine", () => {
  const enginesToDispose: PoseEngine[] = [];

  function trackEngine(engine = new PoseEngine()): PoseEngine {
    enginesToDispose.push(engine);
    return engine;
  }

  beforeEach(() => {
    vi.clearAllMocks();
    mediaPipeMocks.forVisionTasks.mockResolvedValue(DEFAULT_FILESET);
  });

  afterEach(() => {
    while (enginesToDispose.length > 0) {
      enginesToDispose.pop()?.dispose();
    }
    vi.restoreAllMocks();
  });

  describe("init() & delegate fallback", () => {
    it("attempts GPU delegate first, loads default /wasm and lite model URL without canvas, and caches WasmFileset", async () => {
      const gpuInst1 = createMockLandmarker();
      const gpuInst2 = createMockLandmarker();
      mediaPipeMocks.createFromOptions
        .mockResolvedValueOnce(gpuInst1)
        .mockResolvedValueOnce(gpuInst2);

      const engine = trackEngine();
      const delegate = await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.55,
        minTrackingConfidence: 0.65,
      });

      expect(delegate).toBe("GPU");
      expect(engine.isReady()).toBe(true);
      expect(engine.getActiveDelegate()).toBe("GPU");
      expect(engine.getCurrentModel()).toBe("lite");

      expect(mediaPipeMocks.forVisionTasks).toHaveBeenCalledTimes(1);
      expect(mediaPipeMocks.forVisionTasks).toHaveBeenCalledWith("/wasm");

      const firstOpts = mediaPipeMocks.createFromOptions.mock.calls[0][1];
      expect(firstOpts.baseOptions?.delegate).toBe("GPU");
      expect(firstOpts.baseOptions?.modelAssetPath).toBe(MODEL_URLS.lite);
      expect(firstOpts.runningMode).toBe("VIDEO");
      expect(firstOpts.minPoseDetectionConfidence).toBe(0.55);
      expect(firstOpts.minTrackingConfidence).toBe(0.65);
      expect(firstOpts.canvas).toBeUndefined();
      expect("canvas" in firstOpts).toBe(false);

      await engine.switchModel("full");
      expect(mediaPipeMocks.forVisionTasks).toHaveBeenCalledTimes(1);
    });

    it("supports custom wasmBasePath and strips trailing slashes", async () => {
      const gpuInst = createMockLandmarker();
      mediaPipeMocks.createFromOptions.mockResolvedValueOnce(gpuInst);

      const engine = trackEngine();
      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
        wasmBasePath: "/custom-wasm/",
      });

      expect(mediaPipeMocks.forVisionTasks).toHaveBeenCalledWith(
        "/custom-wasm",
      );
    });

    it("falls back to CPU delegate when GPU createFromOptions rejects", async () => {
      const cpuInst = createMockLandmarker();
      mediaPipeMocks.createFromOptions
        .mockRejectedValueOnce(new Error("WebGL init failed"))
        .mockResolvedValueOnce(cpuInst);

      const engine = trackEngine();
      const delegate = await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      expect(delegate).toBe("CPU");
      expect(engine.isReady()).toBe(true);
      expect(engine.getActiveDelegate()).toBe("CPU");
      expect(mediaPipeMocks.forVisionTasks).toHaveBeenCalledTimes(1);
      expect(mediaPipeMocks.createFromOptions).toHaveBeenCalledTimes(2);

      const gpuCallOpts = mediaPipeMocks.createFromOptions.mock.calls[0][1];
      const cpuCallOpts = mediaPipeMocks.createFromOptions.mock.calls[1][1];
      expect(gpuCallOpts.baseOptions?.delegate).toBe("GPU");
      expect(cpuCallOpts.baseOptions?.delegate).toBe("CPU");
      expect(cpuCallOpts.baseOptions?.modelAssetPath).toBe(MODEL_URLS.lite);
      expect(mediaPipeMocks.createFromOptions.mock.calls[1][0]).toBe(
        DEFAULT_FILESET,
      );
    });

    it("throws and leaves isReady() === false when both GPU and CPU reject", async () => {
      mediaPipeMocks.createFromOptions
        .mockRejectedValueOnce(new Error("GPU failed"))
        .mockRejectedValueOnce(new Error("CPU failed"));

      const engine = trackEngine();
      await expect(
        engine.init({
          model: "lite",
          minPoseDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        }),
      ).rejects.toThrow(/CPU failed/);

      expect(engine.isReady()).toBe(false);
      expect(engine.getActiveDelegate()).toBeNull();
      expect(engine.getLastError()?.message).toContain("CPU failed");
    });

    it("throws and leaves isReady() === false if FilesetResolver.forVisionTasks rejects", async () => {
      mediaPipeMocks.forVisionTasks.mockRejectedValueOnce(
        new Error("WASM load failed"),
      );

      const engine = trackEngine();
      await expect(
        engine.init({
          model: "lite",
          minPoseDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        }),
      ).rejects.toThrow(/WASM load failed/i);

      expect(mediaPipeMocks.createFromOptions).not.toHaveBeenCalled();
      expect(engine.isReady()).toBe(false);
      expect(engine.getActiveDelegate()).toBeNull();
    });
  });

  describe("switchModel()", () => {
    it("disposes prior PoseLandmarker before creating new model across lite -> full -> heavy", async () => {
      const inst1 = createMockLandmarker();
      const inst2 = createMockLandmarker();
      const inst3 = createMockLandmarker();
      mediaPipeMocks.createFromOptions
        .mockResolvedValueOnce(inst1)
        .mockResolvedValueOnce(inst2)
        .mockResolvedValueOnce(inst3);

      const engine = trackEngine();
      const video = createMockVideo();

      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      expect(engine.getCurrentModel()).toBe("lite");

      await engine.switchModel("full");
      expect(inst1.close).toHaveBeenCalledTimes(1);
      expect(inst1.close.mock.invocationCallOrder[0]).toBeLessThan(
        mediaPipeMocks.createFromOptions.mock.invocationCallOrder[1],
      );
      expect(engine.getCurrentModel()).toBe("full");

      await engine.switchModel("heavy");
      expect(inst2.close).toHaveBeenCalledTimes(1);
      expect(inst2.close.mock.invocationCallOrder[0]).toBeLessThan(
        mediaPipeMocks.createFromOptions.mock.invocationCallOrder[2],
      );
      expect(engine.getCurrentModel()).toBe("heavy");

      expect(mediaPipeMocks.forVisionTasks).toHaveBeenCalledTimes(1);
      expect(
        mediaPipeMocks.createFromOptions.mock.calls[0][1].baseOptions
          ?.modelAssetPath,
      ).toBe(MODEL_URLS.lite);
      expect(
        mediaPipeMocks.createFromOptions.mock.calls[1][1].baseOptions
          ?.modelAssetPath,
      ).toBe(MODEL_URLS.full);
      expect(
        mediaPipeMocks.createFromOptions.mock.calls[2][1].baseOptions
          ?.modelAssetPath,
      ).toBe(MODEL_URLS.heavy);

      engine.detectFrame(video, 100);
      expect(inst3.detectForVideo).toHaveBeenCalledTimes(1);
      expect(inst1.detectForVideo).not.toHaveBeenCalled();
      expect(inst2.detectForVideo).not.toHaveBeenCalled();
    });

    it("preserves latest thresholds when switching models", async () => {
      const inst1 = createMockLandmarker();
      const inst2 = createMockLandmarker();
      mediaPipeMocks.createFromOptions
        .mockResolvedValueOnce(inst1)
        .mockResolvedValueOnce(inst2);

      const engine = trackEngine();
      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      engine.updateThresholds(0.8, 0.7);
      await engine.switchModel("full");

      const fullOpts = mediaPipeMocks.createFromOptions.mock.calls[1][1];
      expect(fullOpts.minPoseDetectionConfidence).toBe(0.8);
      expect(fullOpts.minTrackingConfidence).toBe(0.7);
      expect(fullOpts.minPosePresenceConfidence).toBe(0.8);
    });

    it("resets ready state safely if switchModel fails on both GPU and CPU and rejects when uninitialized/disposed", async () => {
      const uninitEngine = trackEngine();
      await expect(uninitEngine.switchModel("full")).rejects.toThrow();

      const inst1 = createMockLandmarker();
      mediaPipeMocks.createFromOptions
        .mockResolvedValueOnce(inst1)
        .mockRejectedValueOnce(new Error("Heavy GPU fail"))
        .mockRejectedValueOnce(new Error("Heavy CPU fail"));

      const engine = trackEngine();
      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      await expect(engine.switchModel("heavy")).rejects.toThrow(
        /Heavy CPU fail/,
      );
      expect(inst1.close).toHaveBeenCalledTimes(1);
      expect(engine.isReady()).toBe(false);
      expect(engine.getActiveDelegate()).toBeNull();

      engine.dispose();
      await expect(engine.switchModel("lite")).rejects.toThrow();
    });
  });

  describe("Concurrency & race safety", () => {
    it("closes orphaned full instance and keeps heavy active when switchModel calls resolve out of order", async () => {
      const liteInst = createMockLandmarker();
      const fullInst = createMockLandmarker();
      const heavyInst = createMockLandmarker();

      const fullDeferred = createDeferred<PoseLandmarkerLike>();
      const heavyDeferred = createDeferred<PoseLandmarkerLike>();

      mediaPipeMocks.createFromOptions
        .mockResolvedValueOnce(liteInst)
        .mockReturnValueOnce(fullDeferred.promise)
        .mockReturnValueOnce(heavyDeferred.promise);

      const engine = trackEngine();
      const video = createMockVideo();
      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      const pFull = engine.switchModel("full");
      const pHeavy = engine.switchModel("heavy");
      await flushMicrotasks();

      heavyDeferred.resolve(heavyInst);
      await pHeavy;

      fullDeferred.resolve(fullInst);
      const [fullOutcome] = await Promise.allSettled([pFull]);
      expect(fullOutcome.status).toBe("rejected");
      if (fullOutcome.status === "rejected") {
        expect(isPoseEngineCancelledError(fullOutcome.reason)).toBe(true);
        expect(fullOutcome.reason).toBeInstanceOf(PoseEngineCancelledError);
      }

      expect(fullInst.close).toHaveBeenCalledTimes(1);
      expect(heavyInst.close).not.toHaveBeenCalled();
      expect(engine.getCurrentModel()).toBe("heavy");
      expect(engine.isReady()).toBe(true);

      engine.detectFrame(video, 200);
      expect(heavyInst.detectForVideo).toHaveBeenCalledTimes(1);
      expect(fullInst.detectForVideo).not.toHaveBeenCalled();
    });

    it("immediately closes superseded full instance when it resolves while heavy is still pending", async () => {
      const liteInst = createMockLandmarker();
      const fullInst = createMockLandmarker();
      const heavyInst = createMockLandmarker();

      const fullDeferred = createDeferred<PoseLandmarkerLike>();
      const heavyDeferred = createDeferred<PoseLandmarkerLike>();

      mediaPipeMocks.createFromOptions
        .mockResolvedValueOnce(liteInst)
        .mockReturnValueOnce(fullDeferred.promise)
        .mockReturnValueOnce(heavyDeferred.promise);

      const engine = trackEngine();
      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      const pFull = engine.switchModel("full");
      const pHeavy = engine.switchModel("heavy");
      await flushMicrotasks();

      fullDeferred.resolve(fullInst);
      const [fullOutcome] = await Promise.allSettled([pFull]);
      expect(fullOutcome.status).toBe("rejected");
      expect(fullInst.close).toHaveBeenCalledTimes(1);
      expect(engine.isReady()).toBe(false);

      heavyDeferred.resolve(heavyInst);
      await pHeavy;

      expect(heavyInst.close).not.toHaveBeenCalled();
      expect(engine.getCurrentModel()).toBe("heavy");
      expect(engine.isReady()).toBe(true);
    });

    it("dispose() during pending init() closes late-resolving instance and prevents CPU fallback", async () => {
      const deferredGpu = createDeferred<PoseLandmarkerLike>();
      const lateInst = createMockLandmarker();
      mediaPipeMocks.createFromOptions.mockReturnValueOnce(deferredGpu.promise);

      const engine1 = trackEngine();
      const pInit1 = engine1.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      await flushMicrotasks();

      engine1.dispose();
      deferredGpu.resolve(lateInst);
      const [res1] = await Promise.allSettled([pInit1]);
      expect(res1.status).toBe("rejected");
      if (res1.status === "rejected") {
        expect(isPoseEngineCancelledError(res1.reason)).toBe(true);
      }
      expect(lateInst.close).toHaveBeenCalledTimes(1);
      expect(engine1.isReady()).toBe(false);
      expect(engine1.getActiveDelegate()).toBeNull();

      // Sub-case: GPU rejects after dispose() -> must NOT try CPU fallback
      const deferredReject = createDeferred<PoseLandmarkerLike>();
      mediaPipeMocks.createFromOptions.mockClear();
      mediaPipeMocks.createFromOptions.mockReturnValueOnce(
        deferredReject.promise,
      );

      const engine2 = trackEngine();
      const pInit2 = engine2.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      await flushMicrotasks();

      engine2.dispose();
      deferredReject.reject(new Error("GPU aborted"));
      const [res2] = await Promise.allSettled([pInit2]);
      expect(res2.status).toBe("rejected");
      if (res2.status === "rejected") {
        expect(isPoseEngineCancelledError(res2.reason)).toBe(true);
      }
      expect(mediaPipeMocks.createFromOptions).toHaveBeenCalledTimes(1);
    });

    it("dispose() during pending switchModel() closes both prior active instance and late-resolving new instance", async () => {
      const liteInst = createMockLandmarker();
      const heavyInst = createMockLandmarker();
      const heavyDeferred = createDeferred<PoseLandmarkerLike>();

      mediaPipeMocks.createFromOptions
        .mockResolvedValueOnce(liteInst)
        .mockReturnValueOnce(heavyDeferred.promise);

      const engine = trackEngine();
      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      const pSwitch = engine.switchModel("heavy");
      await flushMicrotasks();
      expect(liteInst.close).toHaveBeenCalledTimes(1);

      engine.dispose();
      heavyDeferred.resolve(heavyInst);
      const [switchOutcome] = await Promise.allSettled([pSwitch]);
      expect(switchOutcome.status).toBe("rejected");
      expect(heavyInst.close).toHaveBeenCalledTimes(1);
      expect(engine.isReady()).toBe(false);
      expect(engine.getActiveDelegate()).toBeNull();
    });

    it("does not spawn a colliding runtime CPU fallback or leak instances on re-entrant switchModel() during throwing detectForVideo()", async () => {
      const liteInst = createMockLandmarker();
      const heavyInst = createMockLandmarker();
      const unexpectedFallbackInst = createMockLandmarker();
      const heavyDeferred = createDeferred<PoseLandmarkerLike>();

      let createCall = 0;
      mediaPipeMocks.createFromOptions.mockImplementation(() => {
        createCall += 1;
        if (createCall === 1) {
          return Promise.resolve(liteInst);
        }
        if (createCall === 2) {
          return heavyDeferred.promise;
        }
        return Promise.resolve(unexpectedFallbackInst);
      });

      try {
        const engine = trackEngine();
        const video = createMockVideo();

        await engine.init({
          model: "lite",
          minPoseDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        const switchRef: { promise: Promise<DelegateType> | null } = {
          promise: null,
        };
        liteInst.detectForVideo.mockImplementationOnce(() => {
          switchRef.promise = engine.switchModel("heavy");
          throw new Error("GPU context lost during re-entrant switchModel");
        });

        const failedFrame = engine.detectFrame(video, 100);
        expect(failedFrame).toEqual({
          landmarks: null,
          inferenceMs: 0,
          delegate: "GPU",
        });
        await flushMicrotasks();

        expect(liteInst.close).toHaveBeenCalledTimes(1);
        expect(mediaPipeMocks.createFromOptions).toHaveBeenCalledTimes(2);
        expect(switchRef.promise).not.toBeNull();

        heavyDeferred.resolve(heavyInst);
        await expect(switchRef.promise).resolves.toBe("GPU");
        await flushMicrotasks();

        expect(mediaPipeMocks.createFromOptions).toHaveBeenCalledTimes(2);
        expect(heavyInst.close).not.toHaveBeenCalled();
        expect(unexpectedFallbackInst.close).not.toHaveBeenCalled();
        expect(engine.isReady()).toBe(true);
        expect(engine.getCurrentModel()).toBe("heavy");
        expect(engine.getActiveDelegate()).toBe("GPU");

        engine.detectFrame(video, 200);
        expect(heavyInst.detectForVideo).toHaveBeenCalledTimes(1);
        expect(liteInst.detectForVideo).toHaveBeenCalledTimes(1);

        engine.dispose();
        expect(heavyInst.close).toHaveBeenCalledTimes(1);
      } finally {
        mediaPipeMocks.createFromOptions.mockReset();
      }
    });
  });

  describe("updateThresholds()", () => {
    it("clamps out-of-range and non-finite confidence values to [0, 1], omits baseOptions/runningMode, and deduplicates", async () => {
      const inst = createMockLandmarker();
      mediaPipeMocks.createFromOptions.mockResolvedValueOnce(inst);

      const engine = trackEngine();
      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      engine.updateThresholds(0.75, 0.85);
      expect(inst.setOptions).toHaveBeenCalledTimes(1);
      expect(inst.setOptions).toHaveBeenLastCalledWith({
        minPoseDetectionConfidence: 0.75,
        minPosePresenceConfidence: 0.75,
        minTrackingConfidence: 0.85,
      });
      const call1Arg = inst.setOptions.mock.calls[0][0];
      expect(call1Arg.baseOptions).toBeUndefined();
      expect(call1Arg.runningMode).toBeUndefined();

      // Deduplicates identical values
      engine.updateThresholds(0.75, 0.85);
      expect(inst.setOptions).toHaveBeenCalledTimes(1);

      // Clamps out-of-range
      engine.updateThresholds(-0.5, 1.8);
      expect(inst.setOptions).toHaveBeenCalledTimes(2);
      expect(inst.setOptions).toHaveBeenLastCalledWith({
        minPoseDetectionConfidence: 0,
        minPosePresenceConfidence: 0,
        minTrackingConfidence: 1,
      });

      // Non-finite inputs fall back to last valid clamped finite numbers
      engine.updateThresholds(Number.NaN, Number.POSITIVE_INFINITY);
      expect(inst.setOptions).toHaveBeenCalledTimes(2);

      // Mixed finite + non-finite updates finite field safely in [0, 1]
      engine.updateThresholds(0.6, Number.NaN);
      expect(inst.setOptions).toHaveBeenCalledTimes(3);
      const call3Arg = inst.setOptions.mock.calls[2][0];
      expect(call3Arg.minPoseDetectionConfidence).toBe(0.6);
      expect(call3Arg.minPosePresenceConfidence).toBe(0.6);
      expect(call3Arg.minTrackingConfidence).toBe(1);
    });

    it("no-ops safely when unready/disposed and swallows async or sync setOptions errors", async () => {
      const inst = createMockLandmarker();
      mediaPipeMocks.createFromOptions.mockResolvedValueOnce(inst);

      const engine = trackEngine();
      // Before init
      expect(() => engine.updateThresholds(0.7, 0.7)).not.toThrow();

      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      // Async rejection
      inst.setOptions.mockRejectedValueOnce(new Error("busy"));
      expect(() => engine.updateThresholds(0.6, 0.6)).not.toThrow();
      await flushMicrotasks();

      // Sync throw
      inst.setOptions.mockImplementationOnce(() => {
        throw new Error("sync graph error");
      });
      expect(() => engine.updateThresholds(0.4, 0.4)).not.toThrow();

      engine.dispose();
      expect(() => engine.updateThresholds(0.3, 0.3)).not.toThrow();
      expect(inst.setOptions).toHaveBeenCalledTimes(2);
    });

    it("reconciles setOptions upon attachment when updateThresholds() is called during pending init()", async () => {
      const initDeferred = createDeferred<PoseLandmarkerLike>();
      const initInst = createMockLandmarker();
      mediaPipeMocks.createFromOptions.mockReturnValueOnce(initDeferred.promise);

      const engine = trackEngine();
      const pInit = engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      await flushMicrotasks();

      expect(mediaPipeMocks.createFromOptions).toHaveBeenCalledTimes(1);
      expect(
        mediaPipeMocks.createFromOptions.mock.calls[0][1]
          .minPoseDetectionConfidence,
      ).toBe(0.5);

      engine.updateThresholds(0.7, 0.65);
      engine.updateThresholds(0.85, 0.9);
      expect(initInst.setOptions).not.toHaveBeenCalled();

      initDeferred.resolve(initInst);
      await expect(pInit).resolves.toBe("GPU");

      expect(initInst.setOptions).toHaveBeenCalledTimes(1);
      expect(initInst.setOptions).toHaveBeenCalledWith({
        minPoseDetectionConfidence: 0.85,
        minPosePresenceConfidence: 0.85,
        minTrackingConfidence: 0.9,
      });
      const reconciledArg = initInst.setOptions.mock.calls[0][0];
      expect(reconciledArg.baseOptions).toBeUndefined();
      expect(reconciledArg.runningMode).toBeUndefined();

      // Re-invoking with the same thresholds deduplicates instead of masking a stale instance
      engine.updateThresholds(0.85, 0.9);
      expect(initInst.setOptions).toHaveBeenCalledTimes(1);
    });

    it("reconciles setOptions upon attachment when updateThresholds() is called during pending switchModel()", async () => {
      const liteInst = createMockLandmarker();
      const heavyInst = createMockLandmarker();
      const switchDeferred = createDeferred<PoseLandmarkerLike>();

      mediaPipeMocks.createFromOptions
        .mockResolvedValueOnce(liteInst)
        .mockReturnValueOnce(switchDeferred.promise);

      const engine = trackEngine();
      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      const pSwitch = engine.switchModel("heavy");
      await flushMicrotasks();

      expect(mediaPipeMocks.createFromOptions).toHaveBeenCalledTimes(2);
      expect(
        mediaPipeMocks.createFromOptions.mock.calls[1][1]
          .minPoseDetectionConfidence,
      ).toBe(0.5);

      engine.updateThresholds(0.25, 0.35);
      expect(liteInst.setOptions).not.toHaveBeenCalled();
      expect(heavyInst.setOptions).not.toHaveBeenCalled();

      switchDeferred.resolve(heavyInst);
      await expect(pSwitch).resolves.toBe("GPU");

      expect(heavyInst.setOptions).toHaveBeenCalledTimes(1);
      expect(heavyInst.setOptions).toHaveBeenCalledWith({
        minPoseDetectionConfidence: 0.25,
        minPosePresenceConfidence: 0.25,
        minTrackingConfidence: 0.35,
      });

      engine.updateThresholds(0.25, 0.35);
      expect(heavyInst.setOptions).toHaveBeenCalledTimes(1);
    });

    it("reconciles setOptions upon attachment when updateThresholds() is called during pending triggerRuntimeCpuFallback()", async () => {
      const gpuInst = createMockLandmarker();
      const cpuInst = createMockLandmarker();
      const cpuFallbackDeferred = createDeferred<PoseLandmarkerLike>();

      gpuInst.detectForVideo.mockImplementationOnce(() => {
        throw new Error("WebGL context lost");
      });

      mediaPipeMocks.createFromOptions
        .mockResolvedValueOnce(gpuInst)
        .mockReturnValueOnce(cpuFallbackDeferred.promise);

      const engine = trackEngine();
      const video = createMockVideo();

      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      const failedFrame = engine.detectFrame(video, 100);
      expect(failedFrame).toEqual({
        landmarks: null,
        inferenceMs: 0,
        delegate: "GPU",
      });
      await flushMicrotasks();

      expect(mediaPipeMocks.createFromOptions).toHaveBeenCalledTimes(2);
      expect(
        mediaPipeMocks.createFromOptions.mock.calls[1][1].baseOptions?.delegate,
      ).toBe("CPU");

      engine.updateThresholds(0.9, 0.85);
      expect(cpuInst.setOptions).not.toHaveBeenCalled();

      cpuFallbackDeferred.resolve(cpuInst);
      await flushMicrotasks();

      expect(engine.isReady()).toBe(true);
      expect(engine.getActiveDelegate()).toBe("CPU");
      expect(cpuInst.setOptions).toHaveBeenCalledTimes(1);
      expect(cpuInst.setOptions).toHaveBeenCalledWith({
        minPoseDetectionConfidence: 0.9,
        minPosePresenceConfidence: 0.9,
        minTrackingConfidence: 0.85,
      });

      engine.updateThresholds(0.9, 0.85);
      expect(cpuInst.setOptions).toHaveBeenCalledTimes(1);
    });

    it("rolls back appliedThresholds on async rejection or sync throw so retrying identical thresholds invokes setOptions again", async () => {
      const inst = createMockLandmarker();
      mediaPipeMocks.createFromOptions.mockResolvedValueOnce(inst);

      const engine = trackEngine();
      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      // Async setOptions rejection should roll back applied state so retry is not deduplicated
      inst.setOptions.mockRejectedValueOnce(new Error("transient busy"));
      engine.updateThresholds(0.8, 0.75);
      expect(inst.setOptions).toHaveBeenCalledTimes(1);
      await flushMicrotasks();

      // Retry same values: must invoke setOptions again and succeed
      engine.updateThresholds(0.8, 0.75);
      expect(inst.setOptions).toHaveBeenCalledTimes(2);
      expect(inst.setOptions).toHaveBeenLastCalledWith({
        minPoseDetectionConfidence: 0.8,
        minPosePresenceConfidence: 0.8,
        minTrackingConfidence: 0.75,
      });
      await flushMicrotasks();

      // Now deduplicates once applied cleanly
      engine.updateThresholds(0.8, 0.75);
      expect(inst.setOptions).toHaveBeenCalledTimes(2);

      // Sync setOptions throw should also roll back applied state so retry works
      inst.setOptions.mockImplementationOnce(() => {
        throw new Error("sync failure");
      });
      engine.updateThresholds(0.3, 0.4);
      expect(inst.setOptions).toHaveBeenCalledTimes(3);

      engine.updateThresholds(0.3, 0.4);
      expect(inst.setOptions).toHaveBeenCalledTimes(4);
      expect(inst.setOptions).toHaveBeenLastCalledWith({
        minPoseDetectionConfidence: 0.3,
        minPosePresenceConfidence: 0.3,
        minTrackingConfidence: 0.4,
      });
    });
  });

  describe("detectFrame()", () => {
    it("skips inference when unready, readyState < 2, videoWidth === 0, or videoHeight === 0", async () => {
      const inst = createMockLandmarker();
      mediaPipeMocks.createFromOptions.mockResolvedValueOnce(inst);

      const engine = trackEngine();
      const readyVideo = createMockVideo({
        readyState: 2,
        videoWidth: 640,
        videoHeight: 480,
      });

      // Uninitialized
      expect(engine.detectFrame(readyVideo, 100)).toEqual({
        landmarks: null,
        inferenceMs: 0,
        delegate: "GPU",
      });

      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      const notLoaded0 = createMockVideo({ readyState: 0 });
      const notLoaded1 = createMockVideo({ readyState: 1 });
      const zeroWidth = createMockVideo({ readyState: 4, videoWidth: 0 });
      const zeroHeight = createMockVideo({ readyState: 4, videoHeight: 0 });

      for (const badVideo of [notLoaded0, notLoaded1, zeroWidth, zeroHeight]) {
        const res = engine.detectFrame(badVideo, 100);
        expect(res).toEqual({
          landmarks: null,
          inferenceMs: 0,
          delegate: "GPU",
        });
      }
      expect(inst.detectForVideo).not.toHaveBeenCalled();

      engine.detectFrame(readyVideo, 100);
      expect(inst.detectForVideo).toHaveBeenCalledTimes(1);
    });

    it("enforces strictly increasing timestamps (100, 101, 102, 120) and sanitizes non-finite/negative timestamps", async () => {
      const inst = createMockLandmarker();
      mediaPipeMocks.createFromOptions.mockResolvedValueOnce(inst);

      const engine = trackEngine();
      const video = createMockVideo();

      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      engine.detectFrame(video, 100);
      engine.detectFrame(video, 100);
      engine.detectFrame(video, 50);
      engine.detectFrame(video, 120);

      expect(inst.detectForVideo.mock.calls.map((c) => c[1])).toEqual([
        100, 101, 102, 120,
      ]);

      // Fresh engine with negative & NaN timestamps
      const inst2 = createMockLandmarker();
      mediaPipeMocks.createFromOptions.mockResolvedValueOnce(inst2);
      const engine2 = trackEngine();
      await engine2.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      vi.spyOn(performance, "now").mockReturnValue(250);
      engine2.detectFrame(video, -5);
      engine2.detectFrame(video, Number.NaN);

      const tsCalls = inst2.detectForVideo.mock.calls.map((c) => c[1]);
      expect(tsCalls[0]).toBeGreaterThanOrEqual(0);
      expect(Number.isFinite(tsCalls[0])).toBe(true);
      expect(tsCalls[1]).toBeGreaterThan(tsCalls[0]);
      expect(Number.isFinite(tsCalls[1])).toBe(true);
    });

    it("advances extreme Number.MAX_SAFE_INTEGER timestamps strictly monotonically without saturation", async () => {
      const inst = createMockLandmarker();
      mediaPipeMocks.createFromOptions.mockResolvedValueOnce(inst);

      const engine = trackEngine();
      const video = createMockVideo();

      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      engine.detectFrame(video, Number.MAX_SAFE_INTEGER);
      engine.detectFrame(video, Number.MAX_SAFE_INTEGER);
      engine.detectFrame(video, Number.MAX_SAFE_INTEGER);
      engine.detectFrame(video, Number.MAX_SAFE_INTEGER);

      const tsCalls = inst.detectForVideo.mock.calls.map((c) => c[1]);
      expect(tsCalls).toHaveLength(4);
      for (let i = 0; i < tsCalls.length; i += 1) {
        expect(Number.isFinite(tsCalls[i])).toBe(true);
        expect(tsCalls[i]).toBeGreaterThanOrEqual(0);
        if (i > 0) {
          expect(tsCalls[i]).toBeGreaterThan(tsCalls[i - 1]);
        }
      }

      // Transition from normal timestamp to MAX_SAFE_INTEGER with static now() clock, then recover
      const inst2 = createMockLandmarker();
      const engine2 = trackEngine(
        new PoseEngine({
          forVisionTasks: async () => DEFAULT_FILESET,
          createFromOptions: async () => inst2,
          now: () => 50,
        }),
      );
      await engine2.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      engine2.detectFrame(video, 100);
      engine2.detectFrame(video, Number.MAX_SAFE_INTEGER);
      engine2.detectFrame(video, Number.MAX_SAFE_INTEGER);
      engine2.detectFrame(video, Number.MAX_SAFE_INTEGER);
      engine2.detectFrame(video, 200);

      const tsCalls2 = inst2.detectForVideo.mock.calls.map((c) => c[1]);
      expect(tsCalls2).toEqual([100, 101, 102, 103, 200]);
    });

    it("maps 33 NormalizedLandmarks to Point3D[], returns null when no pose detected, closes result, and measures inferenceMs", async () => {
      const inst = createMockLandmarker();
      mediaPipeMocks.createFromOptions.mockResolvedValueOnce(inst);

      const mock33 = createMockLandmarks(33);
      const closeSpy1 = vi.fn();
      const closeSpy2 = vi.fn();
      inst.detectForVideo
        .mockReturnValueOnce({
          landmarks: [mock33],
          worldLandmarks: [],
          close: closeSpy1,
        })
        .mockReturnValueOnce({
          landmarks: [],
          worldLandmarks: [],
          close: closeSpy2,
        });

      const engine = trackEngine();
      const video = createMockVideo();

      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      const nowSpy = vi
        .spyOn(performance, "now")
        .mockReturnValueOnce(1000)
        .mockReturnValueOnce(1014.25)
        .mockReturnValueOnce(2000)
        .mockReturnValueOnce(2003.5);

      const frame1 = engine.detectFrame(video, 100);
      expect(frame1.delegate).toBe("GPU");
      expect(frame1.inferenceMs).toBeCloseTo(14.25, 5);
      expect(frame1.landmarks).not.toBeNull();
      expect(frame1.landmarks).toHaveLength(33);
      expect(frame1.landmarks?.[0]).toEqual<Point3D>({
        x: mock33[0].x,
        y: mock33[0].y,
        z: mock33[0].z,
        visibility: mock33[0].visibility,
        presence: mock33[0].presence,
      });
      expect(frame1.landmarks?.[32]).toEqual<Point3D>({
        x: mock33[32].x,
        y: mock33[32].y,
        z: mock33[32].z,
        visibility: mock33[32].visibility,
        presence: mock33[32].presence,
      });
      expect(closeSpy1).toHaveBeenCalledTimes(1);

      const frame2 = engine.detectFrame(video, 133);
      expect(frame2.landmarks).toBeNull();
      expect(frame2.inferenceMs).toBeCloseTo(3.5, 5);
      expect(frame2.delegate).toBe("GPU");
      expect(closeSpy2).toHaveBeenCalledTimes(1);
      expect(nowSpy).toHaveBeenCalledTimes(4);
    });

    it("catches runtime detectForVideo errors on GPU and triggers background CPU fallback", async () => {
      const gpuInst = createMockLandmarker();
      const cpuInst = createMockLandmarker();
      gpuInst.detectForVideo.mockImplementationOnce(() => {
        throw new Error("WebGL context lost");
      });

      mediaPipeMocks.createFromOptions
        .mockResolvedValueOnce(gpuInst)
        .mockResolvedValueOnce(cpuInst);

      const engine = trackEngine();
      const video = createMockVideo();

      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      const failedFrame = engine.detectFrame(video, 100);
      expect(failedFrame).toEqual({
        landmarks: null,
        inferenceMs: 0,
        delegate: "GPU",
      });
      expect(engine.getLastError()?.message).toBe("WebGL context lost");

      await flushMicrotasks();
      expect(gpuInst.close).toHaveBeenCalledTimes(1);
      expect(engine.isReady()).toBe(true);
      expect(engine.getActiveDelegate()).toBe("CPU");

      engine.detectFrame(video, 133);
      expect(cpuInst.detectForVideo).toHaveBeenCalledTimes(1);
    });
  });

  describe("dispose() & dependency injection", () => {
    it("calls close(), resets ready state, is idempotent, and supports re-initialization", async () => {
      const inst1 = createMockLandmarker();
      const inst2 = createMockLandmarker();
      mediaPipeMocks.createFromOptions
        .mockResolvedValueOnce(inst1)
        .mockResolvedValueOnce(inst2);

      const engine = trackEngine();
      const video = createMockVideo();

      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      engine.dispose();
      engine.dispose();

      expect(inst1.close).toHaveBeenCalledTimes(1);
      expect(engine.isReady()).toBe(false);
      expect(engine.getActiveDelegate()).toBeNull();

      const skipped = engine.detectFrame(video, 200);
      expect(skipped.landmarks).toBeNull();
      expect(inst1.detectForVideo).not.toHaveBeenCalled();

      // Re-init after StrictMode dispose
      const delegate = await engine.init({
        model: "full",
        minPoseDetectionConfidence: 0.6,
        minTrackingConfidence: 0.6,
      });
      expect(delegate).toBe("GPU");
      expect(engine.isReady()).toBe(true);
      expect(engine.getCurrentModel()).toBe("full");

      engine.detectFrame(video, 0);
      expect(inst2.detectForVideo).toHaveBeenCalledWith(video, 0);
    });

    it("supports optional PoseEngineDeps injection without relying on module mocks", async () => {
      const injectedFileset: VisionWasmFileset = {
        wasmLoaderPath: "/di/loader.js",
        wasmBinaryPath: "/di/binary.wasm",
      };
      const injectedLandmarker = createMockLandmarker();
      const nowValues = [500, 508.5];
      let nowIdx = 0;

      const forVisionTasks = vi.fn(async () => injectedFileset);
      const createFromOptions = vi.fn(async () => injectedLandmarker);
      const now = vi.fn(() => nowValues[nowIdx++] ?? 600);

      const engine = trackEngine(
        new PoseEngine({
          forVisionTasks,
          createFromOptions,
          now,
        }),
      );

      await engine.init({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      const video = createMockVideo();
      const res = engine.detectFrame(video, 50);

      expect(forVisionTasks).toHaveBeenCalledWith("/wasm");
      expect(createFromOptions).toHaveBeenCalledTimes(1);
      expect(res.inferenceMs).toBeCloseTo(8.5, 5);
    });
  });
});
