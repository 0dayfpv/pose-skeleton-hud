import {
  Activity,
  Camera,
  ChevronRight,
  Cpu,
  Download,
  Eye,
  FlipHorizontal,
  Grid,
  Hand,
  ScanFace,
  Sliders,
  Sparkles,
  Users,
} from "lucide-react";
import type { ReactElement } from "react";
import {
  JOINT_DEFINITIONS,
  type DelegateType,
  type DetailModulesConfig,
  type FaceHeadMetrics,
  type HandTelemetry,
  type JointId,
  type ModelVariant,
  type SmoothingConfig,
  type TrackedPerson,
  type ViewMode,
} from "../engine/geometry";

export interface TelemetryMetrics {
  fps: number;
  inferenceMs: number;
  visibleLandmarks: number;
  totalLandmarks: 33;
  delegate: DelegateType | null;
  detectedHandsCount?: number;
  faceLandmarksCount?: number;
  detectedPersonsCount?: number;
}

export interface ControlPanelProps {
  isOpen: boolean;
  onToggleOpen: () => void;
  telemetry: TelemetryMetrics;
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  mirrored: boolean;
  onMirroredChange: (mirrored: boolean) => void;
  model: ModelVariant;
  onModelChange: (model: ModelVariant) => void;
  minPoseDetectionConfidence: number;
  onMinPoseDetectionConfidenceChange: (value: number) => void;
  minTrackingConfidence: number;
  onMinTrackingConfidenceChange: (value: number) => void;
  smoothing: SmoothingConfig;
  onSmoothingChange: (config: SmoothingConfig) => void;
  detailModules?: DetailModulesConfig;
  onDetailModulesChange?: (config: DetailModulesConfig) => void;
  handMetrics?: readonly HandTelemetry[];
  faceHeadMetrics?: FaceHeadMetrics | null;
  persons?: readonly TrackedPerson[];
  selectedJoints: readonly JointId[];
  onToggleJoint: (jointId: JointId) => void;
  angles: Partial<Record<JointId, number | null>>;
  onExportSnapshot: () => void;
}

const DEFAULT_DETAIL_MODULES: DetailModulesConfig = {
  enableHands: true,
  enableFace: true,
  showFaceMesh: true,
  maxPoses: 2,
};

const MODEL_OPTIONS: readonly {
  id: ModelVariant;
  label: string;
  badge: string;
}[] = [
  { id: "lite", label: "Lite", badge: "Fast" },
  { id: "full", label: "Full", badge: "Balanced" },
  { id: "heavy", label: "Heavy", badge: "Precise" },
];

function clampConfidenceInput(raw: number, fallback: number): number {
  if (!Number.isFinite(raw)) {
    return fallback;
  }
  const clamped = Math.max(0.1, Math.min(0.95, raw));
  return Math.round(clamped * 100) / 100;
}

function clampSmoothingStrength(raw: number, fallback: number): number {
  if (!Number.isFinite(raw)) {
    return fallback;
  }
  const clamped = Math.max(0, Math.min(1, raw));
  return Math.round(clamped * 100) / 100;
}

function formatAngleReadout(
  degrees: number | null | undefined,
  isSelected: boolean,
): string {
  if (
    !isSelected ||
    degrees === null ||
    degrees === undefined ||
    !Number.isFinite(degrees)
  ) {
    return "--°";
  }
  const clamped = Math.max(0, Math.min(180, degrees));
  return `${clamped.toFixed(1)}°`;
}

function formatFingerDeg(deg: number | null | undefined): string {
  if (deg === null || deg === undefined || !Number.isFinite(deg)) {
    return "--°";
  }
  return `${Math.round(Math.max(0, Math.min(180, deg)))}°`;
}

export function ControlPanel({
  isOpen,
  onToggleOpen,
  telemetry,
  viewMode,
  onViewModeChange,
  mirrored,
  onMirroredChange,
  model,
  onModelChange,
  minPoseDetectionConfidence,
  onMinPoseDetectionConfidenceChange,
  minTrackingConfidence,
  onMinTrackingConfidenceChange,
  smoothing,
  onSmoothingChange,
  detailModules = DEFAULT_DETAIL_MODULES,
  onDetailModulesChange,
  handMetrics = [],
  faceHeadMetrics = null,
  persons = [],
  selectedJoints,
  onToggleJoint,
  angles,
  onExportSnapshot,
}: ControlPanelProps): ReactElement {
  const selectedSet = new Set<JointId>(selectedJoints);

  const safeDetection = clampConfidenceInput(minPoseDetectionConfidence, 0.5);
  const safeTracking = clampConfidenceInput(minTrackingConfidence, 0.5);
  const safeStrength = clampSmoothingStrength(smoothing.strength, 0.5);
  const activeDelegate = telemetry.delegate ?? "GPU";
  const maxPoses: 1 | 2 = detailModules.maxPoses === 1 ? 1 : 2;
  const detectedPersonsCount =
    telemetry.detectedPersonsCount ?? persons.length;

  return (
    <aside
      data-testid="control-panel"
      aria-label="Telemetry and Pose Controls"
      className={
        isOpen
          ? "flex max-h-[45dvh] w-full shrink-0 flex-col overflow-y-auto border-t border-cyan-500/20 bg-slate-950/95 p-3 text-slate-100 shadow-[0_0_30px_rgba(6,182,212,0.08)] lg:h-full lg:max-h-none lg:w-84 lg:border-t-0 lg:border-l xl:w-92"
          : "hidden"
      }
    >
      {/* Panel Header */}
      <div className="mb-3 flex items-center justify-between border-b border-slate-800/90 pb-2.5">
        <div className="flex items-center gap-2">
          <Sliders className="h-4 w-4 text-cyan-400" aria-hidden="true" />
          <h2 className="font-mono text-xs font-semibold tracking-wider text-cyan-300 uppercase">
            Lab Controls &amp; Telemetry V2
          </h2>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="inline-flex items-center gap-1 rounded border border-cyan-500/30 bg-cyan-500/10 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-cyan-300">
            <Cpu className="h-3 w-3" aria-hidden="true" />
            {activeDelegate}
          </span>
          <button
            type="button"
            onClick={onToggleOpen}
            aria-label="Collapse control panel"
            className="rounded border border-slate-700/80 bg-slate-900/80 p-1 text-slate-400 transition hover:border-cyan-500/40 hover:text-cyan-300"
          >
            <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="space-y-3.5">
        {/* Display Mode & Mirror Section */}
        <section className="rounded-lg border border-slate-800/90 bg-slate-900/60 p-2.5">
          <div className="mb-2 flex items-center justify-between">
            <span className="font-mono text-[11px] font-semibold tracking-wider text-slate-300 uppercase">
              Display Mode
            </span>
            <span className="font-mono text-[10px] text-cyan-400 uppercase">
              {viewMode}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-1.5">
            <button
              type="button"
              data-testid="mode-overlay"
              aria-pressed={viewMode === "overlay"}
              onClick={() => onViewModeChange("overlay")}
              className={`flex items-center justify-center gap-1.5 rounded border px-2.5 py-1.5 font-mono text-xs font-medium transition ${
                viewMode === "overlay"
                  ? "border-cyan-400/70 bg-cyan-500/20 text-cyan-200 shadow-[0_0_12px_rgba(34,211,238,0.18)]"
                  : "border-slate-700/80 bg-slate-950/60 text-slate-400 hover:border-slate-600 hover:text-slate-200"
              }`}
            >
              <Camera className="h-3.5 w-3.5" aria-hidden="true" />
              Overlay
            </button>

            <button
              type="button"
              data-testid="mode-blackout"
              aria-pressed={viewMode === "blackout"}
              onClick={() => onViewModeChange("blackout")}
              className={`flex items-center justify-center gap-1.5 rounded border px-2.5 py-1.5 font-mono text-xs font-medium transition ${
                viewMode === "blackout"
                  ? "border-cyan-400/70 bg-cyan-500/20 text-cyan-200 shadow-[0_0_12px_rgba(34,211,238,0.18)]"
                  : "border-slate-700/80 bg-slate-950/60 text-slate-400 hover:border-slate-600 hover:text-slate-200"
              }`}
            >
              <Grid className="h-3.5 w-3.5" aria-hidden="true" />
              Blackout
            </button>
          </div>

          <div className="mt-2.5 flex items-center justify-between border-t border-slate-800/80 pt-2">
            <span className="flex items-center gap-1.5 font-mono text-xs text-slate-300">
              <FlipHorizontal
                className="h-3.5 w-3.5 text-cyan-400"
                aria-hidden="true"
              />
              Mirror Mode
            </span>
            <button
              type="button"
              data-testid="mirror-toggle"
              aria-pressed={mirrored}
              onClick={() => onMirroredChange(!mirrored)}
              className={`rounded border px-2.5 py-1 font-mono text-[11px] font-semibold tracking-wider uppercase transition ${
                mirrored
                  ? "border-cyan-400/70 bg-cyan-500/20 text-cyan-200"
                  : "border-slate-700 bg-slate-950/80 text-slate-400 hover:text-slate-200"
              }`}
            >
              {mirrored ? "ON" : "OFF"}
            </button>
          </div>
        </section>

        {/* V2 Detail Modules: Multi-Person (2P), Hands (21x4) & Face/Eyes/Head (478x2) */}
        <section className="rounded-lg border border-cyan-500/25 bg-slate-900/70 p-2.5">
          <div className="mb-2 flex items-center justify-between">
            <span className="flex items-center gap-1.5 font-mono text-[11px] font-semibold tracking-wider text-cyan-300 uppercase">
              <ScanFace className="h-3.5 w-3.5 text-cyan-400" aria-hidden="true" />
              V2 Multi-Subject Tracking
            </span>
            <span className="rounded bg-cyan-500/15 px-1.5 py-0.5 font-mono text-[9px] font-semibold text-cyan-300">
              {maxPoses === 2 ? "2P • 1106 PTS" : "1P • 553 PTS"}
            </span>
          </div>

          <div className="space-y-1.5">
            <div className="flex items-center justify-between rounded border border-slate-800/90 bg-slate-950/70 px-2 py-1.5">
              <span className="flex items-center gap-1.5 font-mono text-xs text-slate-200">
                <Users className="h-3.5 w-3.5 text-emerald-400" aria-hidden="true" />
                Dual-Person Mode ({detectedPersonsCount}/{maxPoses} Active)
              </span>
              <button
                type="button"
                data-testid="toggle-multi-person"
                aria-pressed={maxPoses === 2}
                onClick={() =>
                  onDetailModulesChange?.({
                    ...detailModules,
                    maxPoses: maxPoses === 2 ? 1 : 2,
                  })
                }
                className={`rounded border px-2 py-0.5 font-mono text-[10px] font-semibold uppercase transition ${
                  maxPoses === 2
                    ? "border-emerald-400/70 bg-emerald-500/20 text-emerald-200"
                    : "border-slate-700 bg-slate-900 text-slate-400"
                }`}
              >
                {maxPoses === 2 ? "2 PERSONS" : "1 PERSON"}
              </button>
            </div>

            <div className="flex items-center justify-between rounded border border-slate-800/90 bg-slate-950/70 px-2 py-1.5">
              <span className="flex items-center gap-1.5 font-mono text-xs text-slate-200">
                <Hand className="h-3.5 w-3.5 text-cyan-400" aria-hidden="true" />
                Hands &amp; 5-Fingers (21×{maxPoses * 2})
              </span>
              <button
                type="button"
                data-testid="toggle-hands"
                aria-pressed={detailModules.enableHands}
                onClick={() =>
                  onDetailModulesChange?.({
                    ...detailModules,
                    enableHands: !detailModules.enableHands,
                  })
                }
                className={`rounded border px-2 py-0.5 font-mono text-[10px] font-semibold uppercase transition ${
                  detailModules.enableHands
                    ? "border-cyan-400/70 bg-cyan-500/20 text-cyan-200"
                    : "border-slate-700 bg-slate-900 text-slate-400"
                }`}
              >
                {detailModules.enableHands ? "ON" : "OFF"}
              </button>
            </div>

            <div className="flex items-center justify-between rounded border border-slate-800/90 bg-slate-950/70 px-2 py-1.5">
              <span className="flex items-center gap-1.5 font-mono text-xs text-slate-200">
                <ScanFace
                  className="h-3.5 w-3.5 text-pink-400"
                  aria-hidden="true"
                />
                Face, Eyes, Iris &amp; Head (478)
              </span>
              <button
                type="button"
                data-testid="toggle-face"
                aria-pressed={detailModules.enableFace}
                onClick={() =>
                  onDetailModulesChange?.({
                    ...detailModules,
                    enableFace: !detailModules.enableFace,
                  })
                }
                className={`rounded border px-2 py-0.5 font-mono text-[10px] font-semibold uppercase transition ${
                  detailModules.enableFace
                    ? "border-cyan-400/70 bg-cyan-500/20 text-cyan-200"
                    : "border-slate-700 bg-slate-900 text-slate-400"
                }`}
              >
                {detailModules.enableFace ? "ON" : "OFF"}
              </button>
            </div>

            <div className="flex items-center justify-between rounded border border-slate-800/90 bg-slate-950/70 px-2 py-1.5">
              <span className="flex items-center gap-1.5 font-mono text-xs text-slate-300">
                <Grid className="h-3.5 w-3.5 text-sky-400" aria-hidden="true" />
                3D Face Tesselation Mesh
              </span>
              <button
                type="button"
                data-testid="toggle-face-mesh"
                aria-pressed={detailModules.showFaceMesh}
                onClick={() =>
                  onDetailModulesChange?.({
                    ...detailModules,
                    showFaceMesh: !detailModules.showFaceMesh,
                  })
                }
                className={`rounded border px-2 py-0.5 font-mono text-[10px] font-semibold uppercase transition ${
                  detailModules.showFaceMesh
                    ? "border-cyan-400/70 bg-cyan-500/20 text-cyan-200"
                    : "border-slate-700 bg-slate-900 text-slate-400"
                }`}
              >
                {detailModules.showFaceMesh ? "ON" : "OFF"}
              </button>
            </div>
          </div>

          {/* Live Head, Eye & Hand Biometric Readouts */}
          <div
            data-testid="v2-biometrics-summary"
            className="mt-2.5 space-y-2 border-t border-slate-800/80 pt-2 font-mono text-[11px]"
          >
            {/* Head & Eye Telemetry */}
            <div className="rounded border border-slate-800/90 bg-slate-950/80 p-2">
              <div className="mb-1 flex items-center justify-between text-[10px] text-slate-400 uppercase">
                <span>Head &amp; Eye Geometry</span>
                <span className="text-cyan-300">
                  {faceHeadMetrics ? "LOCKED (478)" : "SCANNING"}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-1 text-center">
                <div className="rounded bg-slate-900/90 px-1 py-1">
                  <div className="text-[9px] text-slate-400">YAW</div>
                  <div
                    data-testid="head-yaw-deg"
                    className="font-semibold text-cyan-300"
                  >
                    {faceHeadMetrics ? `${faceHeadMetrics.headYawDeg}°` : "--°"}
                  </div>
                </div>
                <div className="rounded bg-slate-900/90 px-1 py-1">
                  <div className="text-[9px] text-slate-400">PITCH</div>
                  <div
                    data-testid="head-pitch-deg"
                    className="font-semibold text-cyan-300"
                  >
                    {faceHeadMetrics
                      ? `${faceHeadMetrics.headPitchDeg}°`
                      : "--°"}
                  </div>
                </div>
                <div className="rounded bg-slate-900/90 px-1 py-1">
                  <div className="text-[9px] text-slate-400">ROLL</div>
                  <div
                    data-testid="head-roll-deg"
                    className="font-semibold text-cyan-300"
                  >
                    {faceHeadMetrics
                      ? `${faceHeadMetrics.headRollDeg}°`
                      : "--°"}
                  </div>
                </div>
              </div>
              <div className="mt-1.5 grid grid-cols-3 gap-1 text-center text-[10px]">
                <div className="rounded bg-slate-900/60 px-1 py-0.5">
                  <span className="text-slate-400">L.Eye </span>
                  <span className="text-cyan-300">
                    {faceHeadMetrics
                      ? `${faceHeadMetrics.leftEyeOpenness}%`
                      : "--%"}
                  </span>
                </div>
                <div className="rounded bg-slate-900/60 px-1 py-0.5">
                  <span className="text-slate-400">R.Eye </span>
                  <span className="text-pink-300">
                    {faceHeadMetrics
                      ? `${faceHeadMetrics.rightEyeOpenness}%`
                      : "--%"}
                  </span>
                </div>
                <div className="rounded bg-slate-900/60 px-1 py-0.5">
                  <span className="text-slate-400">Mouth </span>
                  <span className="text-rose-300">
                    {faceHeadMetrics
                      ? `${faceHeadMetrics.mouthOpenness}%`
                      : "--%"}
                  </span>
                </div>
              </div>
            </div>

            {/* Hand & Finger Articulation Telemetry */}
            <div className="rounded border border-slate-800/90 bg-slate-950/80 p-2">
              <div className="mb-1 flex items-center justify-between text-[10px] text-slate-400 uppercase">
                <span>Finger Articulation</span>
                <span className="text-cyan-300">
                  {handMetrics.length} / {maxPoses * 2} HANDS
                </span>
              </div>
              {handMetrics.length === 0 ? (
                <div className="py-1 text-center text-[10px] text-slate-500">
                  Raise left or right hand into view for 21-joint finger tracking
                </div>
              ) : (
                <div className="space-y-1.5">
                  {handMetrics.map((hm, idx) => (
                    <div
                      key={`${hm.handedness}-${idx}`}
                      className="rounded border border-slate-800 bg-slate-900/70 p-1.5"
                    >
                      <div className="flex items-center justify-between text-[10px]">
                        <span
                          className={
                            hm.handedness === "Left"
                              ? "font-semibold text-cyan-300"
                              : "font-semibold text-pink-300"
                          }
                        >
                          {hm.handedness.toUpperCase()} HAND ({hm.extendedFingers}
                          /5 EXT)
                        </span>
                        <span className="text-slate-300">
                          {hm.isPinching
                            ? "PINCH ACTIVE"
                            : `Open ${hm.opennessPercent}%`}
                        </span>
                      </div>
                      <div className="mt-1 grid grid-cols-5 gap-1 text-center text-[9px]">
                        <div className="rounded bg-slate-950/80 py-0.5">
                          <div className="text-slate-500">THM</div>
                          <div className="text-slate-200">
                            {formatFingerDeg(hm.fingerAngles.thumb)}
                          </div>
                        </div>
                        <div className="rounded bg-slate-950/80 py-0.5">
                          <div className="text-slate-500">IDX</div>
                          <div className="text-slate-200">
                            {formatFingerDeg(hm.fingerAngles.index)}
                          </div>
                        </div>
                        <div className="rounded bg-slate-950/80 py-0.5">
                          <div className="text-slate-500">MID</div>
                          <div className="text-slate-200">
                            {formatFingerDeg(hm.fingerAngles.middle)}
                          </div>
                        </div>
                        <div className="rounded bg-slate-950/80 py-0.5">
                          <div className="text-slate-500">RNG</div>
                          <div className="text-slate-200">
                            {formatFingerDeg(hm.fingerAngles.ring)}
                          </div>
                        </div>
                        <div className="rounded bg-slate-950/80 py-0.5">
                          <div className="text-slate-500">PNK</div>
                          <div className="text-slate-200">
                            {formatFingerDeg(hm.fingerAngles.pinky)}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Model & Confidence Section */}
        <section className="rounded-lg border border-slate-800/90 bg-slate-900/60 p-2.5">
          <div className="mb-2 flex items-center justify-between">
            <span className="font-mono text-[11px] font-semibold tracking-wider text-slate-300 uppercase">
              Pose Model
            </span>
            <span className="font-mono text-[10px] text-cyan-400 uppercase">
              {model}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-1.5">
            {MODEL_OPTIONS.map((opt) => {
              const active = model === opt.id;
              return (
                <button
                  key={opt.id}
                  type="button"
                  data-testid={`model-${opt.id}`}
                  aria-pressed={active}
                  onClick={() => onModelChange(opt.id)}
                  className={`flex flex-col items-center rounded border px-2 py-1.5 font-mono transition ${
                    active
                      ? "border-cyan-400/70 bg-cyan-500/20 text-cyan-200 shadow-[0_0_12px_rgba(34,211,238,0.18)]"
                      : "border-slate-700/80 bg-slate-950/60 text-slate-400 hover:border-slate-600 hover:text-slate-200"
                  }`}
                >
                  <span className="text-xs font-semibold">{opt.label}</span>
                  <span className="text-[9px] opacity-75">{opt.badge}</span>
                </button>
              );
            })}
          </div>

          <div className="mt-3 space-y-2.5 border-t border-slate-800/80 pt-2.5">
            <div>
              <div className="mb-1 flex items-center justify-between font-mono text-[11px]">
                <label
                  htmlFor="slider-detection-confidence"
                  className="text-slate-300"
                >
                  Detection Confidence
                </label>
                <span className="text-cyan-300">
                  {safeDetection.toFixed(2)} ({Math.round(safeDetection * 100)}
                  %)
                </span>
              </div>
              <input
                id="slider-detection-confidence"
                data-testid="slider-detection-confidence"
                type="range"
                min="0.1"
                max="0.95"
                step="0.05"
                value={safeDetection}
                onInput={(e) =>
                  onMinPoseDetectionConfidenceChange(
                    clampConfidenceInput(
                      Number.parseFloat(e.currentTarget.value),
                      safeDetection,
                    ),
                  )
                }
                onChange={(e) =>
                  onMinPoseDetectionConfidenceChange(
                    clampConfidenceInput(
                      Number.parseFloat(e.currentTarget.value),
                      safeDetection,
                    ),
                  )
                }
                className="h-1.5 w-full cursor-pointer accent-cyan-400"
              />
            </div>

            <div>
              <div className="mb-1 flex items-center justify-between font-mono text-[11px]">
                <label
                  htmlFor="slider-tracking-confidence"
                  className="text-slate-300"
                >
                  Tracking Confidence
                </label>
                <span className="text-cyan-300">
                  {safeTracking.toFixed(2)} ({Math.round(safeTracking * 100)}%)
                </span>
              </div>
              <input
                id="slider-tracking-confidence"
                data-testid="slider-tracking-confidence"
                type="range"
                min="0.1"
                max="0.95"
                step="0.05"
                value={safeTracking}
                onInput={(e) =>
                  onMinTrackingConfidenceChange(
                    clampConfidenceInput(
                      Number.parseFloat(e.currentTarget.value),
                      safeTracking,
                    ),
                  )
                }
                onChange={(e) =>
                  onMinTrackingConfidenceChange(
                    clampConfidenceInput(
                      Number.parseFloat(e.currentTarget.value),
                      safeTracking,
                    ),
                  )
                }
                className="h-1.5 w-full cursor-pointer accent-cyan-400"
              />
            </div>
          </div>
        </section>

        {/* Smoothing Section */}
        <section className="rounded-lg border border-slate-800/90 bg-slate-900/60 p-2.5">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1.5 font-mono text-[11px] font-semibold tracking-wider text-slate-300 uppercase">
              <Sparkles
                className="h-3.5 w-3.5 text-cyan-400"
                aria-hidden="true"
              />
              Jitter Smoothing
            </span>
            <button
              type="button"
              data-testid="smoothing-toggle"
              aria-pressed={smoothing.enabled}
              onClick={() =>
                onSmoothingChange({
                  enabled: !smoothing.enabled,
                  strength: safeStrength,
                })
              }
              className={`rounded border px-2.5 py-1 font-mono text-[11px] font-semibold tracking-wider uppercase transition ${
                smoothing.enabled
                  ? "border-cyan-400/70 bg-cyan-500/20 text-cyan-200"
                  : "border-slate-700 bg-slate-950/80 text-slate-400 hover:text-slate-200"
              }`}
            >
              {smoothing.enabled ? "ON" : "OFF"}
            </button>
          </div>

          <div className="mt-2.5">
            <div className="mb-1 flex items-center justify-between font-mono text-[11px]">
              <label
                htmlFor="slider-smoothing-strength"
                className="text-slate-300"
              >
                Filter Strength
              </label>
              <span className="text-cyan-300">
                {safeStrength.toFixed(2)} ({Math.round(safeStrength * 100)}%)
              </span>
            </div>
            <input
              id="slider-smoothing-strength"
              data-testid="slider-smoothing-strength"
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={safeStrength}
              onInput={(e) =>
                onSmoothingChange({
                  enabled: smoothing.enabled,
                  strength: clampSmoothingStrength(
                    Number.parseFloat(e.currentTarget.value),
                    safeStrength,
                  ),
                })
              }
              onChange={(e) =>
                onSmoothingChange({
                  enabled: smoothing.enabled,
                  strength: clampSmoothingStrength(
                    Number.parseFloat(e.currentTarget.value),
                    safeStrength,
                  ),
                })
              }
              className="h-1.5 w-full cursor-pointer accent-cyan-400"
            />
          </div>
        </section>

        {/* Joint Angles Section */}
        <section className="rounded-lg border border-slate-800/90 bg-slate-900/60 p-2.5">
          <div className="mb-2 flex items-center justify-between">
            <span className="flex items-center gap-1.5 font-mono text-[11px] font-semibold tracking-wider text-slate-300 uppercase">
              <Activity
                className="h-3.5 w-3.5 text-cyan-400"
                aria-hidden="true"
              />
              Joint Angles ({selectedJoints.length}/8)
            </span>
            <Eye className="h-3.5 w-3.5 text-slate-500" aria-hidden="true" />
          </div>

          <div className="grid grid-cols-1 gap-1.5">
            {JOINT_DEFINITIONS.map((joint) => {
              const isSelected = selectedSet.has(joint.id);
              const rawAngle = angles[joint.id];
              const formattedAngle = formatAngleReadout(rawAngle, isSelected);
              const accentClass =
                joint.side === "left"
                  ? "text-cyan-300 border-cyan-500/40"
                  : "text-pink-300 border-pink-500/40";

              return (
                <div
                  key={joint.id}
                  className={`flex items-center justify-between rounded border px-2 py-1.5 transition ${
                    isSelected
                      ? "border-slate-700/90 bg-slate-950/80"
                      : "border-slate-800/60 bg-slate-950/30 opacity-65"
                  }`}
                >
                  <button
                    type="button"
                    data-testid={`joint-toggle-${joint.id}`}
                    aria-pressed={isSelected}
                    onClick={() => onToggleJoint(joint.id)}
                    className="flex flex-1 items-center gap-2 text-left font-mono text-xs focus:outline-none"
                  >
                    <span
                      className={`inline-block h-2 w-2 rounded-full ${
                        isSelected
                          ? joint.side === "left"
                            ? "bg-cyan-400 shadow-[0_0_6px_#22d3ee]"
                            : "bg-pink-400 shadow-[0_0_6px_#f472b6]"
                          : "bg-slate-600"
                      }`}
                    />
                    <span className="font-medium text-slate-200">
                      {joint.label}
                    </span>
                  </button>

                  <span
                    data-testid={`joint-angle-${joint.id}`}
                    className={`rounded border bg-slate-900/90 px-2 py-0.5 font-mono text-xs font-semibold ${accentClass}`}
                  >
                    {formattedAngle}
                  </span>
                </div>
              );
            })}
          </div>
        </section>

        {/* Export Snapshot Button */}
        <button
          type="button"
          data-testid="snapshot-json-button"
          onClick={onExportSnapshot}
          className="flex w-full items-center justify-center gap-2 rounded-lg border border-cyan-400/60 bg-cyan-500/15 px-4 py-2.5 font-mono text-xs font-semibold tracking-wider text-cyan-200 uppercase shadow-[0_0_20px_rgba(34,211,238,0.12)] transition hover:border-cyan-300 hover:bg-cyan-500/25 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400"
        >
          <Download className="h-4 w-4" aria-hidden="true" />
          Snapshot JSON
        </button>

        {/* Panel Copyright Attribution */}
        <div className="border-t border-slate-800/80 pt-2 text-center font-mono text-[10px] text-slate-500">
          © 2025–{Math.max(2026, new Date().getFullYear())}{" "}
          <a
            href="https://hamzafpv.com.tr"
            target="_blank"
            rel="noopener noreferrer"
            className="font-semibold text-cyan-400 transition hover:text-cyan-300 hover:underline"
          >
            hamzafpv.com.tr
          </a>{" "}
          • All rights reserved.
        </div>
      </div>
    </aside>
  );
}

export default ControlPanel;
