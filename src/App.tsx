import {
  Activity,
  Cpu,
  Eye,
  Gauge,
  Hand,
  PanelRightClose,
  PanelRightOpen,
  ScanFace,
  Timer,
  Users,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactElement,
} from "react";
import ControlPanel, { type TelemetryMetrics } from "./components/ControlPanel";
import SkeletonCanvas, {
  drawSkeletonFrame,
} from "./components/SkeletonCanvas";
import {
  JOINT_DEFINITIONS,
  MultiPersonTracker,
  PoseSmoother,
  computeFaceHeadMetrics,
  computeHandMetrics,
  computeJointAngles,
  serializePoseSnapshot,
  type AppStatus,
  type DelegateType,
  type DetailModulesConfig,
  type FaceHeadMetrics,
  type HandTelemetry,
  type JointId,
  type ModelVariant,
  type Point3D,
  type SmoothingConfig,
  type TrackedHand,
  type TrackedPerson,
  type ViewMode,
} from "./engine/geometry";
import {
  PoseEngine,
  isPoseEngineCancelledError,
} from "./engine/poseEngine";

const DEFAULT_SELECTED_JOINTS: readonly JointId[] = JOINT_DEFINITIONS.map(
  (d) => d.id,
);

const DEFAULT_SMOOTHING: SmoothingConfig = {
  enabled: true,
  strength: 0.5,
};

const DEFAULT_DETAIL_MODULES: DetailModulesConfig = {
  enableHands: true,
  enableFace: true,
  showFaceMesh: true,
  maxPoses: 2,
};

function classifyCameraError(error: unknown): {
  status: AppStatus;
  message: string;
} {
  const errName =
    error instanceof Error ||
    (typeof error === "object" && error !== null && "name" in error)
      ? String((error as { name?: unknown }).name ?? "")
      : "";
  const errMsg =
    error instanceof Error && error.message.trim().length > 0
      ? error.message.trim()
      : "";

  if (
    errName === "NotAllowedError" ||
    errName === "PermissionDeniedError" ||
    errName === "SecurityError"
  ) {
    return {
      status: "permission-denied",
      message:
        errMsg ||
        "Camera permission was denied. Please allow camera access in your browser settings and click Retry.",
    };
  }

  if (errName === "NotFoundError" || errName === "DevicesNotFoundError") {
    return {
      status: "no-camera",
      message:
        errMsg ||
        "No camera device was found. Please connect a webcam and click Retry.",
    };
  }

  if (errName === "NotReadableError" || errName === "TrackStartError") {
    return {
      status: "error",
      message:
        errMsg ||
        "Camera device is currently in use or unreadable. Close other applications using the camera and click Retry.",
    };
  }

  if (
    errName === "OverconstrainedError" ||
    errName === "ConstraintNotSatisfiedError"
  ) {
    return {
      status: "error",
      message:
        errMsg ||
        "Requested camera constraints could not be satisfied by your webcam. Click Retry to re-initialize.",
    };
  }

  return {
    status: "error",
    message:
      errMsg ||
      "Unable to initialize camera stream in this environment. Check device permissions and click Retry.",
  };
}

async function waitForVideoMetadata(
  videoEl: HTMLVideoElement,
  isCancelled: () => boolean,
  maxWaitMs = 1500,
): Promise<void> {
  if (videoEl.readyState >= 2 && videoEl.videoWidth > 0) {
    return;
  }

  if (
    typeof navigator !== "undefined" &&
    /jsdom/i.test(navigator.userAgent ?? "")
  ) {
    return;
  }

  await new Promise<void>((resolve) => {
    const startMs = Date.now();
    let settled = false;

    const finish = (): void => {
      if (settled) return;
      settled = true;
      videoEl.removeEventListener("loadeddata", onReady);
      videoEl.removeEventListener("loadedmetadata", onReady);
      videoEl.removeEventListener("canplay", onReady);
      clearInterval(pollId);
      resolve();
    };

    const onReady = (): void => {
      if (videoEl.readyState >= 2 && videoEl.videoWidth > 0) {
        finish();
      }
    };

    videoEl.addEventListener("loadeddata", onReady);
    videoEl.addEventListener("loadedmetadata", onReady);
    videoEl.addEventListener("canplay", onReady);

    const pollId = setInterval(() => {
      if (
        isCancelled() ||
        (videoEl.readyState >= 2 && videoEl.videoWidth > 0) ||
        Date.now() - startMs >= maxWaitMs
      ) {
        finish();
      }
    }, 25);
  });
}

function countVisibleLandmarks(
  landmarks: readonly Point3D[] | null,
  threshold = 0.5,
): number {
  if (!landmarks || landmarks.length === 0) {
    return 0;
  }
  let count = 0;
  const limit = Math.min(33, landmarks.length);
  for (let i = 0; i < limit; i++) {
    const vis = landmarks[i]?.visibility;
    if (vis !== undefined && Number.isFinite(vis) && vis >= threshold) {
      count += 1;
    }
  }
  return count;
}

export default function App(): ReactElement {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const engineRef = useRef<PoseEngine | null>(null);
  const multiTrackerRef = useRef<MultiPersonTracker>(new MultiPersonTracker());
  const smootherRef = useRef<PoseSmoother>(new PoseSmoother());
  const leftHandSmootherRef = useRef<PoseSmoother>(new PoseSmoother());
  const rightHandSmootherRef = useRef<PoseSmoother>(new PoseSmoother());
  const faceSmootherRef = useRef<PoseSmoother>(new PoseSmoother());
  const streamRef = useRef<MediaStream | null>(null);
  const sessionTokenRef = useRef<number>(0);

  const [status, setStatus] = useState<AppStatus>("requesting-camera");
  const [statusMessage, setStatusMessage] = useState<string | undefined>(
    undefined,
  );
  const [isPanelOpen, setIsPanelOpen] = useState<boolean>(true);
  const [viewMode, setViewMode] = useState<ViewMode>("overlay");
  const [mirrored, setMirrored] = useState<boolean>(false);
  const [model, setModel] = useState<ModelVariant>("lite");
  const [minPoseDetectionConfidence, setMinPoseDetectionConfidence] =
    useState<number>(0.5);
  const [minTrackingConfidence, setMinTrackingConfidence] =
    useState<number>(0.5);
  const [smoothing, setSmoothing] =
    useState<SmoothingConfig>(DEFAULT_SMOOTHING);
  const [detailModules, setDetailModules] = useState<DetailModulesConfig>(
    DEFAULT_DETAIL_MODULES,
  );
  const [selectedJoints, setSelectedJoints] = useState<readonly JointId[]>(
    DEFAULT_SELECTED_JOINTS,
  );

  const [landmarks, setLandmarks] = useState<readonly Point3D[] | null>(null);
  const [hands, setHands] = useState<readonly TrackedHand[]>([]);
  const [faceLandmarks, setFaceLandmarks] = useState<readonly Point3D[] | null>(
    null,
  );
  const [persons, setPersons] = useState<readonly TrackedPerson[]>([]);
  const [handMetrics, setHandMetrics] = useState<readonly HandTelemetry[]>([]);
  const [faceHeadMetrics, setFaceHeadMetrics] =
    useState<FaceHeadMetrics | null>(null);
  const [angles, setAngles] = useState<Partial<Record<JointId, number | null>>>(
    () => computeJointAngles(null, DEFAULT_SELECTED_JOINTS),
  );
  const [telemetry, setTelemetry] = useState<TelemetryMetrics>({
    fps: 0,
    inferenceMs: 0,
    visibleLandmarks: 0,
    totalLandmarks: 33,
    delegate: "GPU",
    detectedHandsCount: 0,
    faceLandmarksCount: 0,
    detectedPersonsCount: 0,
  });

  // Live refs for the requestAnimationFrame loop & callbacks
  const statusRef = useRef<AppStatus>(status);
  const viewModeRef = useRef<ViewMode>(viewMode);
  const mirroredRef = useRef<boolean>(mirrored);
  const modelRef = useRef<ModelVariant>(model);
  const detectionConfRef = useRef<number>(minPoseDetectionConfidence);
  const trackingConfRef = useRef<number>(minTrackingConfidence);
  const smoothingRef = useRef<SmoothingConfig>(smoothing);
  const detailModulesRef = useRef<DetailModulesConfig>(detailModules);
  const selectedJointsRef = useRef<readonly JointId[]>(selectedJoints);

  const latestLandmarksRef = useRef<readonly Point3D[] | null>(null);
  const latestHandsRef = useRef<readonly TrackedHand[]>([]);
  const latestFaceLandmarksRef = useRef<readonly Point3D[] | null>(null);
  const latestPersonsRef = useRef<readonly TrackedPerson[]>([]);
  const latestAnglesRef = useRef<Partial<Record<JointId, number | null>>>(
    angles,
  );
  const modelSwitchInFlightRef = useRef<boolean>(false);

  useEffect(() => {
    statusRef.current = status;
    viewModeRef.current = viewMode;
    mirroredRef.current = mirrored;
    modelRef.current = model;
    detectionConfRef.current = minPoseDetectionConfidence;
    trackingConfRef.current = minTrackingConfidence;
    smoothingRef.current = smoothing;
    detailModulesRef.current = detailModules;
    selectedJointsRef.current = selectedJoints;
  }, [
    status,
    viewMode,
    mirrored,
    model,
    minPoseDetectionConfidence,
    minTrackingConfidence,
    smoothing,
    detailModules,
    selectedJoints,
  ]);

  const resetAllSmoothers = useCallback((): void => {
    multiTrackerRef.current.reset();
    smootherRef.current.reset();
    leftHandSmootherRef.current.reset();
    rightHandSmootherRef.current.reset();
    faceSmootherRef.current.reset();
  }, []);

  const stopActiveStream = useCallback((): void => {
    const activeStream = streamRef.current;
    streamRef.current = null;
    if (activeStream) {
      for (const track of activeStream.getTracks()) {
        try {
          track.stop();
        } catch {
          /* ignore track stop errors */
        }
      }
    }
    const videoEl = videoRef.current;
    if (videoEl) {
      try {
        videoEl.srcObject = null;
      } catch {
        /* ignore srcObject clear errors in synthetic DOMs */
      }
    }
  }, []);

  const getOrCreateEngine = useCallback((): PoseEngine => {
    if (!engineRef.current) {
      engineRef.current = new PoseEngine();
    }
    return engineRef.current;
  }, []);

  const startSession = useCallback(async (): Promise<void> => {
    const token = ++sessionTokenRef.current;
    const isCancelled = (): boolean => token !== sessionTokenRef.current;

    stopActiveStream();
    resetAllSmoothers();
    latestLandmarksRef.current = null;
    latestHandsRef.current = [];
    latestFaceLandmarksRef.current = null;
    latestPersonsRef.current = [];
    latestAnglesRef.current = computeJointAngles(
      null,
      selectedJointsRef.current,
    );
    setLandmarks(null);
    setHands([]);
    setFaceLandmarks(null);
    setPersons([]);
    setHandMetrics([]);
    setFaceHeadMetrics(null);
    setAngles(latestAnglesRef.current);
    setStatus("requesting-camera");
    setStatusMessage(undefined);

    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices ||
      typeof navigator.mediaDevices.getUserMedia !== "function"
    ) {
      if (!isCancelled()) {
        setStatus("error");
        setStatusMessage(
          "MediaDevices camera API is not available in this browser or context.",
        );
      }
      return;
    }

    let mediaStream: MediaStream;
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
    } catch (cameraError) {
      if (isCancelled()) {
        return;
      }
      const classified = classifyCameraError(cameraError);
      setStatus(classified.status);
      setStatusMessage(classified.message);
      return;
    }

    if (isCancelled()) {
      for (const track of mediaStream.getTracks()) {
        try {
          track.stop();
        } catch {
          /* ignore */
        }
      }
      return;
    }

    streamRef.current = mediaStream;
    const videoEl = videoRef.current;
    if (videoEl) {
      try {
        videoEl.srcObject = mediaStream;
        const playPromise = videoEl.play();
        if (playPromise && typeof playPromise.catch === "function") {
          await playPromise.catch(() => undefined);
        }
      } catch {
        /* ignore synthetic play errors */
      }
    }

    if (isCancelled()) {
      stopActiveStream();
      return;
    }

    setStatus("loading-model");
    setStatusMessage(undefined);

    const engine = getOrCreateEngine();
    let activeDelegate: DelegateType;
    try {
      activeDelegate = await engine.init({
        model: modelRef.current,
        minPoseDetectionConfidence: detectionConfRef.current,
        minTrackingConfidence: trackingConfRef.current,
        enableHands: detailModulesRef.current.enableHands,
        enableFace: detailModulesRef.current.enableFace,
        maxPoses: detailModulesRef.current.maxPoses ?? 2,
      });
    } catch (modelError) {
      if (isCancelled() || isPoseEngineCancelledError(modelError)) {
        return;
      }
      setStatus("error");
      setStatusMessage(
        modelError instanceof Error && modelError.message.trim().length > 0
          ? `Failed to load MediaPipe Pose model: ${modelError.message}`
          : "Failed to download or initialize MediaPipe Pose model. Please check your connection and click Retry.",
      );
      return;
    }

    if (isCancelled()) {
      return;
    }

    if (videoEl) {
      await waitForVideoMetadata(videoEl, isCancelled);
    }

    if (isCancelled()) {
      return;
    }

    setTelemetry((prev) => ({
      ...prev,
      delegate: activeDelegate,
    }));
    statusRef.current = "ready";
    setStatus("ready");
    setStatusMessage(undefined);
  }, [getOrCreateEngine, resetAllSmoothers, stopActiveStream]);

  // Mount & unmount lifecycle
  useEffect(() => {
    const initTimer = window.setTimeout(() => {
      void startSession();
    }, 0);
    return () => {
      window.clearTimeout(initTimer);
      sessionTokenRef.current += 1;
      stopActiveStream();
      if (engineRef.current) {
        engineRef.current.dispose();
        engineRef.current = null;
      }
    };
  }, [startSession, stopActiveStream]);

  // Synchronized requestAnimationFrame loop
  useEffect(() => {
    let rafId: number | null = null;
    let lastFrameMs: number | null = null;
    let smoothedFps = 30;
    let lastHudCommitMs = 0;
    let lastVideoTime = -1;
    let lastInferenceEndMs = 0;
    let latestInferenceMs = 0;

    const handleVisibilityChange = (): void => {
      lastFrameMs = null;
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);

    const tick = (timestampMs: number): void => {
      const nowMs =
        Number.isFinite(timestampMs) && timestampMs > 0
          ? timestampMs
          : performance.now();

      if (lastFrameMs !== null) {
        const dtMs = nowMs - lastFrameMs;
        if (dtMs > 1 && dtMs < 1000) {
          const instantFps = 1000 / dtMs;
          smoothedFps = smoothedFps * 0.85 + instantFps * 0.15;
        }
      }
      lastFrameMs = nowMs;

      let frameDelegate: DelegateType | null =
        engineRef.current?.getActiveDelegate() ?? "GPU";

      const videoEl = videoRef.current;
      const engine = engineRef.current;

      const minGapMs = Math.min(250, Math.max(24, latestInferenceMs * 1.25));
      const canRunInference =
        statusRef.current === "ready" &&
        videoEl !== null &&
        engine !== null &&
        engine.isReady() &&
        videoEl.readyState >= 2 &&
        videoEl.videoWidth > 0 &&
        videoEl.videoHeight > 0 &&
        (videoEl.currentTime !== lastVideoTime || lastVideoTime < 0) &&
        performance.now() - lastInferenceEndMs >= minGapMs;

      if (canRunInference && videoEl && engine) {
        lastVideoTime = videoEl.currentTime;
        const result = engine.detectFrame(videoEl, nowMs);
        lastInferenceEndMs = performance.now();
        latestInferenceMs = Number.isFinite(result.inferenceMs)
          ? Math.max(0, result.inferenceMs)
          : 0;
        frameDelegate = result.delegate;

        const maxPoses = detailModulesRef.current.maxPoses === 1 ? 1 : 2;
        const rawPoses = (
          result.poses ?? (result.landmarks ? [result.landmarks] : [])
        ).slice(0, maxPoses);
        const rawHands = detailModulesRef.current.enableHands
          ? (result.hands ?? [])
          : [];
        const rawFaces = detailModulesRef.current.enableFace
          ? (
              result.faces ??
              (result.faceLandmarks ? [result.faceLandmarks] : [])
            ).slice(0, maxPoses)
          : [];

        if (rawPoses.length > 0) {
          const trackedPersons = multiTrackerRef.current.update({
            poses: rawPoses,
            hands: rawHands,
            faces: rawFaces,
            selectedJoints: selectedJointsRef.current,
            timestampMs: nowMs,
            smoothing: smoothingRef.current,
          });
          const primary = trackedPersons[0] ?? null;
          latestPersonsRef.current = trackedPersons;
          latestLandmarksRef.current = primary ? primary.landmarks : null;
          latestAnglesRef.current = primary
            ? primary.angles
            : computeJointAngles(null, selectedJointsRef.current);
          latestHandsRef.current = trackedPersons.flatMap((p) => p.hands);
          latestFaceLandmarksRef.current =
            primary?.faceLandmarks ??
            trackedPersons[1]?.faceLandmarks ??
            null;
        } else {
          multiTrackerRef.current.reset();
          latestPersonsRef.current = [];
          latestLandmarksRef.current = null;
          latestAnglesRef.current = computeJointAngles(
            null,
            selectedJointsRef.current,
          );

          const smoothedHands: TrackedHand[] =
            detailModulesRef.current.enableHands
              ? rawHands.map((h) => ({
                  handedness: h.handedness,
                  score: h.score,
                  landmarks: (h.handedness === "Left"
                    ? leftHandSmootherRef.current
                    : rightHandSmootherRef.current
                  ).filter(h.landmarks, nowMs, smoothingRef.current),
                }))
              : [];

          const rawFace0 = rawFaces[0] ?? null;
          const smoothedFace =
            detailModulesRef.current.enableFace && rawFace0
              ? faceSmootherRef.current.filter(
                  rawFace0,
                  nowMs,
                  smoothingRef.current,
                )
              : null;

          latestHandsRef.current = smoothedHands;
          latestFaceLandmarksRef.current = smoothedFace;
        }
      }

      // Synchronized canvas draw
      const canvas = canvasRef.current;
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const cssWidth = rect.width > 0 ? rect.width : canvas.clientWidth;
        const cssHeight = rect.height > 0 ? rect.height : canvas.clientHeight;
        if (cssWidth > 0 && cssHeight > 0) {
          const dpr =
            Number.isFinite(window.devicePixelRatio) &&
            window.devicePixelRatio > 0
              ? window.devicePixelRatio
              : 1;
          const backingW = Math.max(1, Math.round(cssWidth * dpr));
          const backingH = Math.max(1, Math.round(cssHeight * dpr));
          if (canvas.width !== backingW || canvas.height !== backingH) {
            canvas.width = backingW;
            canvas.height = backingH;
          }
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            drawSkeletonFrame(ctx, cssWidth, cssHeight, {
              landmarks: latestLandmarksRef.current,
              hands: detailModulesRef.current.enableHands
                ? latestHandsRef.current
                : [],
              faceLandmarks: detailModulesRef.current.enableFace
                ? latestFaceLandmarksRef.current
                : null,
              persons: latestPersonsRef.current,
              showFaceMesh: detailModulesRef.current.showFaceMesh,
              angles: latestAnglesRef.current,
              selectedJoints: selectedJointsRef.current,
              viewMode: viewModeRef.current,
              mirrored: mirroredRef.current,
            });
          }
        }
      }

      // Commit React HUD / telemetry state at ~10 Hz when ready
      if (
        statusRef.current === "ready" &&
        nowMs - lastHudCommitMs >= 100
      ) {
        lastHudCommitMs = nowMs;
        const currentLandmarks = latestLandmarksRef.current;
        const currentPersons = latestPersonsRef.current;
        const currentHands = detailModulesRef.current.enableHands
          ? latestHandsRef.current
          : [];
        const currentFace = detailModulesRef.current.enableFace
          ? latestFaceLandmarksRef.current
          : null;
        const currentAngles = latestAnglesRef.current;
        const visibleCount = countVisibleLandmarks(currentLandmarks);
        const totalFacePoints = currentPersons.reduce(
          (acc, p) => acc + (p.faceLandmarks?.length ?? 0),
          0,
        ) || (currentFace?.length ?? 0);
        const safeFps = Number.isFinite(smoothedFps)
          ? Math.max(1, Math.min(240, Math.round(smoothedFps)))
          : 30;

        setLandmarks(currentLandmarks);
        setHands(currentHands);
        setFaceLandmarks(currentFace);
        setPersons(currentPersons);
        setHandMetrics(currentHands.map((h) => computeHandMetrics(h)));
        setFaceHeadMetrics(computeFaceHeadMetrics(currentFace));
        setAngles(currentAngles);
        setTelemetry({
          fps: safeFps,
          inferenceMs: latestInferenceMs,
          visibleLandmarks: visibleCount,
          totalLandmarks: 33,
          delegate: frameDelegate,
          detectedHandsCount: currentHands.length,
          faceLandmarksCount: totalFacePoints,
          detectedPersonsCount: currentPersons.length,
        });
      }

      rafId = window.requestAnimationFrame(tick);
    };

    rafId = window.requestAnimationFrame(tick);

    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      if (rafId !== null) {
        window.cancelAnimationFrame(rafId);
      }
    };
  }, []);

  const handleModelChange = useCallback(
    async (nextModel: ModelVariant): Promise<void> => {
      const engine = getOrCreateEngine();
      if (
        nextModel === modelRef.current &&
        !modelSwitchInFlightRef.current &&
        statusRef.current === "ready" &&
        engine.isReady() &&
        engine.getCurrentModel() === nextModel
      ) {
        return;
      }

      setModel(nextModel);
      modelRef.current = nextModel;

      if (
        statusRef.current === "requesting-camera" ||
        statusRef.current === "permission-denied" ||
        statusRef.current === "no-camera"
      ) {
        return;
      }

      modelSwitchInFlightRef.current = true;
      setStatus("loading-model");
      setStatusMessage(undefined);

      try {
        const nextDelegate = await engine.switchModel(nextModel);
        if (modelRef.current !== nextModel) {
          return;
        }
        modelSwitchInFlightRef.current = false;
        resetAllSmoothers();
        setTelemetry((prev) => ({
          ...prev,
          delegate: nextDelegate,
        }));
        setStatus("ready");
        setStatusMessage(undefined);
      } catch (err) {
        if (isPoseEngineCancelledError(err) || modelRef.current !== nextModel) {
          return;
        }
        modelSwitchInFlightRef.current = false;
        setStatus("error");
        setStatusMessage(
          err instanceof Error && err.message.trim().length > 0
            ? `Failed to switch Pose model: ${err.message}`
            : "Failed to load selected Pose model. Click Retry to try again.",
        );
      }
    },
    [getOrCreateEngine, resetAllSmoothers],
  );

  const handleDetectionConfidenceChange = useCallback(
    (nextDetection: number): void => {
      const clamped = Number.isFinite(nextDetection)
        ? Math.max(0.1, Math.min(0.95, Math.round(nextDetection * 100) / 100))
        : 0.5;
      setMinPoseDetectionConfidence(clamped);
      detectionConfRef.current = clamped;
      engineRef.current?.updateThresholds(clamped, trackingConfRef.current);
    },
    [],
  );

  const handleTrackingConfidenceChange = useCallback(
    (nextTracking: number): void => {
      const clamped = Number.isFinite(nextTracking)
        ? Math.max(0.1, Math.min(0.95, Math.round(nextTracking * 100) / 100))
        : 0.5;
      setMinTrackingConfidence(clamped);
      trackingConfRef.current = clamped;
      engineRef.current?.updateThresholds(detectionConfRef.current, clamped);
    },
    [],
  );

  const handleSmoothingChange = useCallback(
    (nextConfig: SmoothingConfig): void => {
      const safeStrength = Number.isFinite(nextConfig.strength)
        ? Math.max(
            0,
            Math.min(1, Math.round(nextConfig.strength * 100) / 100),
          )
        : smoothingRef.current.strength;
      const normalized: SmoothingConfig = {
        enabled: Boolean(nextConfig.enabled),
        strength: safeStrength === 0 ? 0 : safeStrength,
      };
      setSmoothing(normalized);
      smoothingRef.current = normalized;
      if (!normalized.enabled) {
        resetAllSmoothers();
      }
    },
    [resetAllSmoothers],
  );

  const handleDetailModulesChange = useCallback(
    (nextModules: DetailModulesConfig): void => {
      const nextMaxPoses: 1 | 2 = nextModules.maxPoses === 1 ? 1 : 2;
      const normalized: DetailModulesConfig = {
        enableHands: Boolean(nextModules.enableHands),
        enableFace: Boolean(nextModules.enableFace),
        showFaceMesh: Boolean(nextModules.showFaceMesh),
        maxPoses: nextMaxPoses,
      };
      setDetailModules(normalized);
      detailModulesRef.current = normalized;
      engineRef.current?.setDetailModules({
        enableHands: normalized.enableHands,
        enableFace: normalized.enableFace,
        maxPoses: nextMaxPoses,
      });
      if (nextMaxPoses === 1 && latestPersonsRef.current.length > 1) {
        latestPersonsRef.current = latestPersonsRef.current.slice(0, 1);
        setPersons(latestPersonsRef.current);
      }
      if (!normalized.enableHands) {
        latestHandsRef.current = [];
        setHands([]);
        setHandMetrics([]);
        leftHandSmootherRef.current.reset();
        rightHandSmootherRef.current.reset();
      }
      if (!normalized.enableFace) {
        latestFaceLandmarksRef.current = null;
        setFaceLandmarks(null);
        setFaceHeadMetrics(null);
        faceSmootherRef.current.reset();
      }
    },
    [],
  );

  const handleToggleJoint = useCallback((jointId: JointId): void => {
    setSelectedJoints((prev) => {
      const set = new Set<JointId>(prev);
      if (set.has(jointId)) {
        set.delete(jointId);
      } else {
        set.add(jointId);
      }
      const next = JOINT_DEFINITIONS.map((d) => d.id).filter((id) =>
        set.has(id),
      );
      selectedJointsRef.current = next;
      const updatedAngles = computeJointAngles(
        latestLandmarksRef.current,
        next,
      );
      latestAnglesRef.current = updatedAngles;
      setAngles(updatedAngles);
      return next;
    });
  }, []);

  const handleExportSnapshot = useCallback((): void => {
    const nowMs = Date.now();
    const currentLandmarks = latestLandmarksRef.current;
    const currentSelected = selectedJointsRef.current;
    const currentAngles = computeJointAngles(currentLandmarks, currentSelected);

    const jsonText = serializePoseSnapshot({
      landmarks: currentLandmarks,
      selectedJoints: currentSelected,
      angles: currentAngles,
      model: modelRef.current,
      mirrored: mirroredRef.current,
      smoothing: smoothingRef.current,
      timestampMs: nowMs,
      hands: detailModulesRef.current.enableHands
        ? latestHandsRef.current
        : [],
      faceLandmarks: detailModulesRef.current.enableFace
        ? latestFaceLandmarksRef.current
        : null,
      persons: latestPersonsRef.current,
    });

    const blob = new Blob([jsonText], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `pose-snapshot-${nowMs}.json`;
    anchor.style.display = "none";
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    window.setTimeout(() => {
      URL.revokeObjectURL(url);
    }, 5000);
  }, []);

  const safeFps = Number.isFinite(telemetry.fps)
    ? Math.max(0, Math.round(telemetry.fps))
    : 0;
  const safeInferenceMs = Number.isFinite(telemetry.inferenceMs)
    ? Math.max(0, telemetry.inferenceMs)
    : 0;
  const safeVisibleJoints = Number.isFinite(telemetry.visibleLandmarks)
    ? Math.max(0, Math.min(33, Math.round(telemetry.visibleLandmarks)))
    : 0;
  const activeDelegate = telemetry.delegate ?? "GPU";
  const detectedHandsCount = telemetry.detectedHandsCount ?? 0;
  const facePointsCount = telemetry.faceLandmarksCount ?? 0;
  const detectedPersonsCount = telemetry.detectedPersonsCount ?? persons.length;
  const maxPoses: 1 | 2 = detailModules.maxPoses === 1 ? 1 : 2;

  return (
    <div className="flex h-dvh w-full max-w-full flex-col overflow-hidden bg-[#05080f] text-slate-100 select-none">
      {/* Persistent Top Cybernetic Performance HUD Header */}
      <header className="z-30 flex shrink-0 flex-wrap items-center justify-between gap-1.5 border-b border-cyan-500/20 bg-slate-950/95 px-2.5 py-1.5 backdrop-blur-md">
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded border border-cyan-400/40 bg-cyan-500/10">
            <Activity className="h-3.5 w-3.5 text-cyan-400" aria-hidden="true" />
          </div>
          <h1 className="font-mono text-xs font-bold tracking-wider text-cyan-300 uppercase sm:text-sm">
            Pose Skeleton HUD <span className="text-pink-400">V2</span>
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
          {/* FPS Readout */}
          <div className="inline-flex items-center gap-1 rounded border border-slate-800 bg-slate-900/90 px-2 py-0.5 font-mono text-[11px]">
            <Gauge className="h-3 w-3 text-cyan-400" aria-hidden="true" />
            <span className="text-slate-400">FPS</span>
            <span
              data-testid="hud-fps"
              className="font-semibold text-cyan-300"
            >
              {safeFps}
            </span>
          </div>

          {/* Inference Latency Readout */}
          <div className="inline-flex items-center gap-1 rounded border border-slate-800 bg-slate-900/90 px-2 py-0.5 font-mono text-[11px]">
            <Timer className="h-3 w-3 text-pink-400" aria-hidden="true" />
            <span
              data-testid="hud-inference-ms"
              className="font-semibold text-pink-300"
            >
              {safeInferenceMs.toFixed(1)} ms
            </span>
          </div>

          {/* Visible Body Landmarks Readout */}
          <div className="inline-flex items-center gap-1 rounded border border-slate-800 bg-slate-900/90 px-2 py-0.5 font-mono text-[11px]">
            <Eye className="h-3 w-3 text-emerald-400" aria-hidden="true" />
            <span
              data-testid="hud-visible-joints"
              className="font-semibold text-emerald-300"
            >
              {safeVisibleJoints} / 33
            </span>
          </div>

          {/* V2 Dual-Person Readout */}
          <div className="hidden items-center gap-1 rounded border border-slate-800 bg-slate-900/90 px-2 py-0.5 font-mono text-[11px] sm:inline-flex">
            <Users className="h-3 w-3 text-emerald-400" aria-hidden="true" />
            <span
              data-testid="hud-persons-count"
              className="font-semibold text-emerald-200"
            >
              {detectedPersonsCount} / {maxPoses} P
            </span>
          </div>

          {/* V2 Hands Readout */}
          <div className="hidden items-center gap-1 rounded border border-slate-800 bg-slate-900/90 px-2 py-0.5 font-mono text-[11px] sm:inline-flex">
            <Hand className="h-3 w-3 text-cyan-400" aria-hidden="true" />
            <span
              data-testid="hud-hands-count"
              className="font-semibold text-cyan-200"
            >
              {detectedHandsCount} / {maxPoses * 2}
            </span>
          </div>

          {/* V2 Face/Head Mesh Readout */}
          <div className="hidden items-center gap-1 rounded border border-slate-800 bg-slate-900/90 px-2 py-0.5 font-mono text-[11px] sm:inline-flex">
            <ScanFace className="h-3 w-3 text-pink-400" aria-hidden="true" />
            <span
              data-testid="hud-face-points"
              className="font-semibold text-pink-200"
            >
              {facePointsCount} pts
            </span>
          </div>

          {/* Active Delegate Badge */}
          <div
            data-testid="hud-delegate"
            className="inline-flex items-center gap-1 rounded border border-cyan-500/30 bg-cyan-500/10 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-cyan-300"
          >
            <Cpu className="h-3 w-3" aria-hidden="true" />
            <span>{activeDelegate}</span>
          </div>

          {/* Panel Collapse / Expand Toggle */}
          <button
            type="button"
            data-testid="panel-toggle"
            aria-expanded={isPanelOpen}
            aria-label={
              isPanelOpen ? "Collapse control panel" : "Expand control panel"
            }
            onClick={() => setIsPanelOpen((prev) => !prev)}
            className="inline-flex items-center gap-1 rounded border border-cyan-500/40 bg-cyan-500/15 px-2 py-0.5 font-mono text-[11px] font-semibold text-cyan-200 transition hover:border-cyan-400 hover:bg-cyan-500/25 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400"
          >
            {isPanelOpen ? (
              <PanelRightClose className="h-3.5 w-3.5" aria-hidden="true" />
            ) : (
              <PanelRightOpen className="h-3.5 w-3.5" aria-hidden="true" />
            )}
            <span className="hidden sm:inline">
              {isPanelOpen ? "Hide Panel" : "Controls"}
            </span>
          </button>
        </div>
      </header>

      {/* Main Responsive Stage + Collapsible Control Panel */}
      <main className="flex min-h-0 flex-1 flex-col overflow-hidden lg:flex-row">
        <div className="relative flex min-h-[220px] min-w-0 flex-1 items-center justify-center overflow-hidden p-2">
          <SkeletonCanvas
            videoRef={videoRef}
            canvasRef={canvasRef}
            landmarks={landmarks}
            hands={detailModules.enableHands ? hands : []}
            faceLandmarks={detailModules.enableFace ? faceLandmarks : null}
            persons={persons}
            showFaceMesh={detailModules.showFaceMesh}
            angles={angles}
            selectedJoints={selectedJoints}
            viewMode={viewMode}
            mirrored={mirrored}
            status={status}
            statusMessage={statusMessage}
            onRetry={() => {
              void startSession();
            }}
          />
        </div>

        <ControlPanel
          isOpen={isPanelOpen}
          onToggleOpen={() => setIsPanelOpen((prev) => !prev)}
          telemetry={telemetry}
          viewMode={viewMode}
          onViewModeChange={setViewMode}
          mirrored={mirrored}
          onMirroredChange={setMirrored}
          model={model}
          onModelChange={(nextModel) => {
            void handleModelChange(nextModel);
          }}
          minPoseDetectionConfidence={minPoseDetectionConfidence}
          onMinPoseDetectionConfidenceChange={handleDetectionConfidenceChange}
          minTrackingConfidence={minTrackingConfidence}
          onMinTrackingConfidenceChange={handleTrackingConfidenceChange}
          smoothing={smoothing}
          onSmoothingChange={handleSmoothingChange}
          detailModules={detailModules}
          onDetailModulesChange={handleDetailModulesChange}
          handMetrics={handMetrics}
          faceHeadMetrics={faceHeadMetrics}
          persons={persons}
          selectedJoints={selectedJoints}
          onToggleJoint={handleToggleJoint}
          angles={angles}
          onExportSnapshot={handleExportSnapshot}
        />
      </main>

      {/* Persistent Cybernetic Copyright Footer */}
      <footer
        data-testid="app-copyright-footer"
        className="z-20 shrink-0 border-t border-cyan-500/15 bg-slate-950/95 px-3 py-1 text-center font-mono text-[10px] text-slate-400"
      >
        © 2025–{Math.max(2026, new Date().getFullYear())}{" "}
        <a
          href="https://hamzafpv.com.tr"
          target="_blank"
          rel="noopener noreferrer"
          className="font-semibold text-cyan-400 transition hover:text-cyan-300 hover:underline"
        >
          hamzafpv.com.tr
        </a>
        {" • "}
        <span>All rights reserved.</span>
      </footer>
    </div>
  );
}
