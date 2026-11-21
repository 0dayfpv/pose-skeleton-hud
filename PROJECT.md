# Project: Pose Skeleton HUD (`pose_skeleton_hud`)

## Architecture
Client-side real-time human pose estimation and cybernetic telemetry HUD built with **Vite 8 + React 19 + TypeScript (~5.8) + Tailwind CSS v4 + `@mediapipe/tasks-vision` (v1.2.3)**.

### System Data Flow
1. **Camera Capture (`src/App.tsx`)**: Requests `navigator.mediaDevices.getUserMedia({ video: ... })` with strict lifecycle state tracking (`requesting-camera`, `ready`, `permission-denied`, `no-camera`, `error`) and React 19 StrictMode cleanup.
2. **ML Inference (`src/engine/poseEngine.ts`)**: Manages `FilesetResolver.forVisionTasks('/wasm')` and `PoseLandmarker` (`Lite` / `Full` / `Heavy` via CORS-enabled HuggingFace `litert-community` `.task` URLs). Attempts `delegate: "GPU"` first and falls back to `"CPU"` on failure. Runs `detectForVideo` inside the `requestAnimationFrame` loop with strictly monotonic timestamps and `video.readyState >= 2` guards; supports live `setOptions` threshold updates and clean `close()` disposal on model switches.
3. **Geometry, Smoothing & Export (`src/engine/geometry.ts`)**: Pure TypeScript math module computing 3D vector joint angles (`calculateAngle3D`, `computeJointAngles`) for 8 anatomical joints (L/R Elbow, Shoulder, Hip, Knee), adaptive One-Euro smoothing (`PoseSmoother`) with exact zero-mutation pass-through when disabled, and `PoseSnapshot` JSON serialization guaranteeing 33 numeric `{ index, name, x, y, z, visibility }` landmarks.
4. **Canvas Rendering & Status Overlays (`src/components/SkeletonCanvas.tsx`)**: Aspect-locked `<video>` + HiDPI `<canvas>` stage supporting **Overlay** (semi-transparent neon skeleton over video) and **Blackout** (dark cybernetic grid + `z`-depth scaling/brightness/glow) modes, smart coloring (Left=Cyan, Right=Magenta/Orange, Torso/Midline=White/Gray, visibility-based alpha dimming), **Mirror** mode (`scale-x-[-1]` on `<video>` only + `1 - x` canvas coordinate flip so angle pills remain unflipped and readable), and actionable status/error overlays with Retry.
5. **Telemetry & Controls (`src/components/ControlPanel.tsx`)**: Collapsible dark cybernetic HUD right panel displaying Performance HUD (`FPS`, `inference ms`, `visible / 33`, active delegate), View Mode toggle, Mirror toggle, Model selector (`Lite`/`Full`/`Heavy`), `minPoseDetectionConfidence` & `minTrackingConfidence` sliders, Smoothing toggle + strength slider, 8 Joint Angle selectors + live `°` readouts, and `"Snapshot JSON"` download button.

---

## Feature Inventory
Every feature from the Survey phase is assigned a milestone. No feature is unassigned.

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F-01 | Project Scaffolding & Tooling | Vite + React 19 + TS `<6.1` + Tailwind v4 + ESLint + Vitest + Playwright setup | M1 | R1, AC-Build |
| F-02 | 3D Vector Joint Angle Math | `calculateAngle3D` & `computeJointAngles` for 8 joints (`±0.5°`, degenerate/non-finite safe) | M1 | R1, R3, AC-Unit |
| F-03 | One-Euro Smoothing Filter | `PoseSmoother` reducing jitter variance when on; bit-exact raw pass-through when off | M1 | R1, R3, AC-Unit |
| F-04 | Snapshot JSON Serializer | `buildPoseSnapshot` & `serializePoseSnapshot` with 33 `{index,name,x,y,z,visibility}` items, ISO timestamp, active angles, zeroed fallback | M1 | R3, AC-Unit |
| F-05 | MediaPipe Pose Engine | `PoseLandmarker` (`Lite`/`Full`/`Heavy`), HuggingFace CORS URLs, local `/wasm`, GPU-first + CPU fallback | M2 | R1, R3 |
| F-06 | Live Options & Clean Disposal | Live `setOptions` confidence updates, `close()` disposal on model swap/unmount, monotonic timestamp + StrictMode guards | M2 | R1, R3 |
| F-07 | Overlay & Blackout Modes | Aspect-locked `<video>` + HiDPI `<canvas>`; Overlay semi-transparent neon; Blackout dark grid + `z`-depth scale/brightness/glow | M3 | R2 |
| F-08 | Smart Coloring & Visibility Dimming | Left=Cyan, Right=Magenta/Orange, Torso/Head=White/Gray (explicit face index handling), low-`visibility` alpha dimming | M3 | R2 |
| F-09 | Mirror Mode (Readable Text) | `<video>` CSS `scale-x-[-1]` + canvas `1 - x` landmark projection; unflipped upright angle pills | M3 | R2 |
| F-10 | Status & Error Overlay Screens | Actionable HUD screens for `requesting-camera`, `loading-model`, `no-camera`, `permission-denied`, `error` (`NotSupportedError` etc.) + Retry | M3 | R4, AC-Browser |
| F-11 | Live Joint Angle Telemetry | On-canvas angle pills + ControlPanel joint selectors (8 joints) & live `°` readouts | M4 | R3 |
| F-12 | Performance HUD | Live FPS, `inference ms`, `visible / 33` joint count, active delegate badge | M4 | R3 |
| F-13 | Responsive Collapsible ControlPanel | Dark cybernetic HUD layout, collapsible right panel (`aria-expanded`), all interactive controls + Snapshot JSON download | M4 | R3, R4, AC-Browser |
| F-14 | Synchronized `rAF` App Loop | `App.tsx` webcam capture, `rAF` inference→smooth→angle→draw loop, state machine & StrictMode cleanup | M4 | R1, R4 |
| F-15 | Opaque-Box E2E Test Suite | Requirement-driven Playwright suite (Tiers 1–4) + `TEST_INFRA.md` + `TEST_READY.md` | E2E_TRACK | Dual Track, AC-Browser |
| F-16 | Final Verification & Tier 5 Hardening | 100% Tier 1–4 E2E pass + build/tsc/lint/vitest pass + Tier 5 white-box adversarial coverage hardening | M5 | Project Pattern |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|--------------|--------|
| M1 | Scaffolding & Geometry Engine | `package.json` (pinned `typescript ~5.8.3`, `@playwright/test 1.65.1`, `@mediapipe/tasks-vision ^1.2.3`, Vitest, ESLint, Tailwind v4), `public/wasm/` copy, `src/engine/geometry.ts`, unit tests `src/engine/geometry.test.ts` | none | DONE (`19/19` tests, `tsc`/`lint`/`build` pass) |
| M2 | MediaPipe Pose Engine | `src/engine/poseEngine.ts` + `src/engine/poseEngine.test.ts` (mocked MediaPipe tests for GPU→CPU fallback, model switch + `close()`, `setOptions`, monotonic `detectForVideo`, race cancellation) | M1 | DONE (`26/26` tests) |
| M3 | Skeleton Canvas & Status UI | `src/components/SkeletonCanvas.tsx` + `src/components/SkeletonCanvas.test.tsx` (Overlay/Blackout, z-depth, smart coloring, Mirror `1 - x` math + readable pills, status/error screens + Retry) | M1 | DONE (`14/14` tests, Chromium mirror CSS verified, Auditor CLEAN) |
| M4 | Control Panel & App Integration | `src/components/ControlPanel.tsx`, `src/App.tsx`, `src/main.tsx`, `src/index.css`, unit/component tests for ControlPanel & App error/control wiring | M2, M3 | DONE (`10/10` tests, `69/69` total unit tests) |
| E2E_TRACK | Opaque-Box E2E Testing Track | `TEST_INFRA.md`, `playwright.config.ts`, `e2e/` test suite (Tiers 1–4), `TEST_READY.md` | M1 (needs `@playwright/test` installed; file-isolated from `src/`) | DONE (`59/59` specs, `161/161` cases) |
| M5 | Final Milestone (E2E Pass + Tier 5) | Phase 1: pass 100% E2E Tiers 1–4 + build/tsc/lint/vitest; Phase 2: Tier 5 adversarial coverage hardening | M4, E2E_TRACK | DONE (`59/59` E2E specs, `69/69` unit tests, `0` lint/tsc/build errors) |

---

## Code Layout (Strict File Write Ownership)
```
pose_skeleton_hud/
├── package.json                  # M1 owns
├── tsconfig.json                 # M1 owns
├── tsconfig.app.json             # M1 owns
├── tsconfig.node.json            # M1 owns
├── vite.config.ts                # M1 owns
├── vitest.config.ts              # M1 owns
├── eslint.config.js              # M1 owns
├── index.html                    # M1 owns
├── playwright.config.ts          # E2E_TRACK owns
├── TEST_INFRA.md                 # E2E_TRACK owns
├── TEST_READY.md                 # E2E_TRACK owns
├── PROJECT.md                    # Orchestrator owns
├── public/
│   └── wasm/                     # M1 copies from @mediapipe/tasks-vision/wasm
├── e2e/                          # E2E_TRACK owns (Tiers 1-4), M5 Phase 2 adds Tier 5
│   └── ...
└── src/
    ├── main.tsx                  # M1 creates placeholder; M4 may refine if needed
    ├── index.css                 # M1 creates Tailwind v4 entry; M4 styles cybernetic theme
    ├── App.tsx                   # M1 creates minimal placeholder; M4 owns full implementation
    ├── App.test.tsx              # M4 owns
    ├── engine/
    │   ├── geometry.ts           # M1 owns (shared types + math + smoother + snapshot)
    │   ├── geometry.test.ts      # M1 owns
    │   ├── poseEngine.ts         # M2 owns
    │   └── poseEngine.test.ts    # M2 owns
    └── components/
        ├── SkeletonCanvas.tsx    # M3 owns
        ├── SkeletonCanvas.test.tsx # M3 owns
        ├── ControlPanel.tsx      # M4 owns
        └── ControlPanel.test.tsx # M4 owns
```

---

## Interface Contracts

### 1. Shared Domain Types (`src/engine/geometry.ts`)
```ts
export type ModelVariant = "lite" | "full" | "heavy";
export type DelegateType = "GPU" | "CPU";
export type ViewMode = "overlay" | "blackout";

export type AppStatus =
  | "requesting-camera"
  | "loading-model"
  | "ready"
  | "no-camera"
  | "permission-denied"
  | "error";

export type JointId =
  | "left_elbow"
  | "right_elbow"
  | "left_shoulder"
  | "right_shoulder"
  | "left_hip"
  | "right_hip"
  | "left_knee"
  | "right_knee";

export interface Point3D {
  x: number;
  y: number;
  z: number;
  visibility?: number;
  presence?: number;
}

export interface SmoothingConfig {
  enabled: boolean;
  strength: number; // [0, 1]
}

export interface JointDefinition {
  id: JointId;
  label: string;
  shortLabel: string;
  side: "left" | "right";
  indices: readonly [number, number, number]; // [A, Vertex B, C]
}

export interface SnapshotLandmark {
  index: number;
  name: string;
  x: number;
  y: number;
  z: number;
  visibility: number;
}

export interface SnapshotJointAngle {
  joint: JointId;
  label: string;
  degrees: number | null;
}

export interface PoseSnapshot {
  schemaVersion: "1.0.0";
  timestamp: string;
  timestampMs: number;
  poseDetected: boolean;
  model: ModelVariant;
  mirrored: boolean;
  smoothing: SmoothingConfig;
  landmarks: SnapshotLandmark[]; // ALWAYS length 33 (indices 0..32)
  activeJointAngles: SnapshotJointAngle[];
}
```

### 2. `src/engine/geometry.ts` ↔ Consumers (`M2`, `M3`, `M4`)
- `POSE_LANDMARK_NAMES: readonly string[]` (length 33, indices `0..32`).
- `JOINT_DEFINITIONS: readonly JointDefinition[]` (all 8 joints):
  - `left_elbow`: `[11, 13, 15]`
  - `right_elbow`: `[12, 14, 16]`
  - `left_shoulder`: `[23, 11, 13]`
  - `right_shoulder`: `[24, 12, 14]`
  - `left_hip`: `[11, 23, 25]`
  - `right_hip`: `[12, 24, 26]`
  - `left_knee`: `[23, 25, 27]`
  - `right_knee`: `[24, 26, 28]`
- `calculateAngle3D(a: Point3D, b: Point3D, c: Point3D): number | null` — returns `[0, 180]` degrees (`±0.5°` accuracy), or `null` if non-finite or either ray magnitude `< 1e-8`.
- `computeJointAngles(landmarks: readonly Point3D[] | null, selectedJoints: readonly JointId[], minVisibility?: number): Record<JointId, number | null>` — default `minVisibility = 0.4`.
- `class PoseSmoother`:
  - `filter(landmarks: readonly Point3D[], timestampMs: number, config: SmoothingConfig): Point3D[]` — returns bit-exact `(x, y, z, visibility)` copies when `config.enabled === false`; applies adaptive One-Euro filter per coordinate when `true`.
  - `reset(): void`
- `buildPoseSnapshot(params): PoseSnapshot` & `serializePoseSnapshot(params): string`.

### 3. `src/engine/poseEngine.ts` ↔ `src/App.tsx` (`M2` ↔ `M4`)
```ts
export interface PoseEngineOptions {
  model: ModelVariant;
  minPoseDetectionConfidence: number;
  minTrackingConfidence: number;
  minPosePresenceConfidence?: number;
  wasmBasePath?: string; // defaults to "/wasm"
}

export interface InferenceFrameResult {
  landmarks: Point3D[] | null;
  inferenceMs: number;
  delegate: DelegateType;
}

export class PoseEngine {
  init(options: PoseEngineOptions): Promise<DelegateType>;
  switchModel(model: ModelVariant): Promise<DelegateType>;
  updateThresholds(minPoseDetectionConfidence: number, minTrackingConfidence: number): void;
  detectFrame(video: HTMLVideoElement, timestampMs: number): InferenceFrameResult;
  dispose(): void;
  isReady(): boolean;
  getActiveDelegate(): DelegateType | null;
  getCurrentModel(): ModelVariant;
}
```
- Model URLs (HuggingFace `litert-community`, verified CORS `*`):
  - `lite`: `https://huggingface.co/litert-community/MediaPipe-PoseLandmarker-Lite/resolve/main/pose_landmarker_lite.task`
  - `full`: `https://huggingface.co/litert-community/MediaPipe-PoseLandmarker-Full/resolve/main/pose_landmarker_full.task`
  - `heavy`: `https://huggingface.co/litert-community/MediaPipe-PoseLandmarker-Heavy/resolve/main/pose_landmarker_heavy.task`
- Must NEVER pass the 2D HUD `<canvas>` to `PoseLandmarker.createFromOptions`.

### 4. `src/components/SkeletonCanvas.tsx` ↔ `src/App.tsx` (`M3` ↔ `M4`)
```ts
export interface SkeletonCanvasProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  landmarks: readonly Point3D[] | null;
  angles: Partial<Record<JointId, number | null>>;
  selectedJoints: readonly JointId[];
  viewMode: ViewMode;
  mirrored: boolean;
  status: AppStatus;
  statusMessage?: string;
  onRetry?: () => void;
  videoAspectRatio?: number;
}
```
- Also export pure/testable drawing helper `drawSkeletonFrame(ctx, width, height, params)` so unit tests can verify Mirror math (`1 - x` vs unflipped text), Overlay vs Blackout background, `z`-depth scaling, and smart coloring directly!

### 5. `src/components/ControlPanel.tsx` ↔ `src/App.tsx` (`M4`)
```ts
export interface TelemetryMetrics {
  fps: number;
  inferenceMs: number;
  visibleLandmarks: number;
  totalLandmarks: 33;
  delegate: DelegateType | null;
}
```
Required DOM `data-testid` & accessible controls (shared contract with `E2E_TRACK`):
- Performance HUD:
  - `data-testid="hud-fps"` (renders numeric FPS)
  - `data-testid="hud-inference-ms"` (renders `ms`)
  - `data-testid="hud-visible-joints"` (renders `<n> / 33`)
- Stage elements (`SkeletonCanvas.tsx`):
  - `data-testid="pose-video"` (`<video>`)
  - `data-testid="pose-canvas"` (`<canvas>`)
  - `data-testid="status-overlay"` (when `status !== "ready"`, showing descriptive status message)
  - `data-testid="retry-button"` (visible on `permission-denied`, `no-camera`, `error`)
- ControlPanel controls:
  - `data-testid="panel-toggle"` (`aria-expanded="true"|"false"`, collapses/expands `data-testid="control-panel"`)
  - `data-testid="mode-overlay"` & `data-testid="mode-blackout"` (or role radio/button with `aria-pressed`)
  - `data-testid="mirror-toggle"` (`aria-pressed` or checkbox)
  - `data-testid="model-lite"`, `data-testid="model-full"`, `data-testid="model-heavy"`
  - `data-testid="slider-detection-confidence"` (`<input type="range">`, min `0.1`, max `0.95`, step `0.05`)
  - `data-testid="slider-tracking-confidence"` (`<input type="range">`, min `0.1`, max `0.95`, step `0.05`)
  - `data-testid="smoothing-toggle"` (`aria-pressed` or checkbox)
  - `data-testid="slider-smoothing-strength"` (`<input type="range">`, min `0`, max `1`, step `0.05`)
  - `data-testid="joint-toggle-<jointId>"` for all 8 `JointId`s + `data-testid="joint-angle-<jointId>"` showing live `°` (or `--°`)
  - `data-testid="snapshot-json-button"` (`"Snapshot JSON"`) — triggers browser download of `pose-snapshot-<timestamp>.json`
