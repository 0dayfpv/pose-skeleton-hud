# Pose Skeleton HUD — Opaque-Box E2E Test Infrastructure (`TEST_INFRA.md`)

> **Track**: `E2E_TRACK` (`F-15`) | **Target URL**: `http://localhost:4173` | **Mode**: Strictly Opaque-Box
> **Sources**: `.agents/teamwork/ORIGINAL_REQUEST.md` (`R1`–`R4`, Acceptance Criteria) & `PROJECT.md` (`F-01`–`F-14`, `§ Interface Contracts`)

---

## 1. Opaque-Box Testing Principles & Contracts

1. **Zero Internal Coupling**: No file under `e2e/` or `playwright.config.ts` may import from `src/` (not even TypeScript types), call internal hooks/classes (`PoseEngine`, `PoseSmoother`, `calculateAngle3D`), or inspect internal React state.
2. **Browser-Only Observation**: All assertions operate exclusively through `http://localhost:4173`, DOM/ARIA attributes, `data-testid` contracts (`PROJECT.md § Interface Contracts`), computed CSS styles, HTML5 `<canvas>`/`<video>` web APIs, network responses, browser events (`pageerror`, `console`), and downloaded `pose-snapshot-*.json` files.
3. **Global Runtime Invariants** (enforced across all non-error-injection tests):
   - `pageerror` count === `0` (listeners attached **before** `page.goto('/')`).
   - `console.error` count === `0` (listeners attached **before** `page.goto('/')`).
   - Note for simulated error tests (`NotAllowedError`, `NotFoundError`, `NotSupportedError`, network failure): uncaught `pageerror` count must still remain `0`; controlled UI error states must never crash React.

---

## 2. Playwright & Environment Infrastructure

### 2.1 Server & Browser Projects (`playwright.config.ts`)
- **Web Server**: `npm run dev -- --port 4173 --strictPort`, `url: 'http://localhost:4173'`, `reuseExistingServer: !process.env.CI`.
  - `http://localhost:4173` satisfies `window.isSecureContext === true` required by `navigator.mediaDevices.getUserMedia`.
- **Browser Binary**: `@playwright/test@1.65.1` using pre-cached headless `chromium-1255` (`~/.cache/ms-playwright`).
- **Projects**:
  1. `chromium-fake-webcam` (default):
     - `permissions: ['camera']`
     - `launchOptions.args`:
       - `--use-fake-device-for-media-stream`
       - `--use-fake-ui-for-media-stream`
       - `--enable-webgl`
       - `--use-gl=angle`
       - `--use-angle=swiftshader`
       - `--enable-unsafe-swiftshader`
     - Produces a synthetic `640x480@20fps` camera stream (`fake_device_0`). Because Chromium's synthetic pattern contains no human body, default stream inference yields `0 / 33` visible joints and `poseDetected: false` — tests accept `0 / 33` as valid steady-state telemetry while verifying `[0, 33]` numeric invariants.
  2. `chromium-no-device` (isolated helper launch `launchNoFakeDeviceSession` without fake-device flags):
     - Launches headless Chromium **without** `--use-fake-device-for-media-stream` or `--use-fake-ui-for-media-stream` to verify real headless `NotSupportedError` handling.
  3. **Deterministic Camera Error Injection (`page.addInitScript`)**:
     - Overrides `navigator.mediaDevices.getUserMedia` (and `enumerateDevices` when needed) via mutable `window.__e2eMediaMockError` state so React 19 `<StrictMode>` double-mounts do not prematurely consume the error before `retry-button` is clicked.

### 2.2 Assets, CDN & Timeout Budget
- **Local WASM**: `/wasm/vision_wasm_internal.{js,wasm}` and `/wasm/vision_wasm_module_internal.{js,wasm}` served from `public/wasm/`.
- **Remote Models** (HuggingFace `litert-community`, CORS `*`):
  - `Lite` (~5.5 MB): default startup model.
  - `Full` (~9.0 MB): requires `>= 45s` timeout budget on first download.
  - `Heavy` (~29.2 MB): requires `>= 90s` timeout budget on first download; Tiers 2–4 use an opaque-box worker-level Playwright `page.route('**/*.task*')` cache (preserving CORS `*` and `application/octet-stream`) to avoid redundant 29 MB CDN downloads while supporting deterministic slow/failure/race injection.

### 2.3 DOM / `data-testid` Contract Reference (`PROJECT.md § Interface Contracts`)
| Surface | `data-testid` / Selector | Contracted Behavior / Attributes |
|---|---|---|
| Stage Video | `pose-video` | `<video>` element; horizontally flipped via CSS `transform` (`m11 === -1`) OR CSS `scale` (`-1 ...`) only when Mirror is ON; hidden (`opacity-0` / `visibility: hidden` / `display: none`) in Blackout mode |
| Stage Canvas | `pose-canvas` | `<canvas>` element; HiDPI backing store; NEVER horizontally flipped in Mirror mode on either `transform` or `scale` |
| Status Overlay | `status-overlay` | Visible when `status !== "ready"` (`requesting-camera`, `loading-model`, `no-camera`, `permission-denied`, `error`) |
| Retry Button | `retry-button` | Visible on `permission-denied`, `no-camera`, `error`; re-triggers camera/engine init |
| Performance HUD | `hud-fps`, `hud-inference-ms`, `hud-visible-joints` | Numeric FPS; numeric `ms`; `<n> / 33` (`0 <= n <= 33`); active delegate (`GPU` or `CPU`) visible in HUD/panel |
| Panel Toggle | `panel-toggle`, `control-panel` | `aria-expanded="true"\|"false"`; collapses/expands `control-panel` |
| View Mode | `mode-overlay`, `mode-blackout` | Radio/toggle buttons (`aria-pressed` or `aria-checked`) switching Overlay ↔ Blackout |
| Mirror | `mirror-toggle` | Toggle/checkbox (`aria-pressed` or `:checked`) |
| Model Selector | `model-lite`, `model-full`, `model-heavy` | Buttons/radios switching active `PoseLandmarker` variant |
| Confidence Sliders | `slider-detection-confidence`, `slider-tracking-confidence` | `<input type="range" min="0.1" max="0.95" step="0.05">` |
| Smoothing | `smoothing-toggle`, `slider-smoothing-strength` | Toggle (`aria-pressed`/checkbox) + `<input type="range" min="0" max="1" step="0.05">` |
| Joint Angles (x8) | `joint-toggle-<jointId>`, `joint-angle-<jointId>` | `left_elbow`, `right_elbow`, `left_shoulder`, `right_shoulder`, `left_hip`, `right_hip`, `left_knee`, `right_knee`; live `°` or `--°` |
| Snapshot Export | `snapshot-json-button` | Triggers download `pose-snapshot-<timestamp>.json` matching `PoseSnapshot` schema (`33` landmarks) |

---

## 3. Opaque-Box Test Feature Inventory (`N = 14`)

Every feature from `PROJECT.md` (`F-01`..`F-14`) and `ORIGINAL_REQUEST.md` (`R1`–`R4`) is mapped below to an opaque-box E2E testable feature unit.

| ID | Feature Name | Source | Opaque-Box Verification Surface | Primary Tier 1 Pass Criterion |
|---|---|---|---|---|
| **F-01** | Project Scaffolding, Dark Theme & WASM Boot | R1, R4, AC-Build, AC-Browser | `http://localhost:4173`, `<html class="dark">` / dark bg, `/wasm/*`, `pageerror`, `console.error` | App boots with HTTP 200, dark HUD palette, local WASM 200s, and `0` console/page errors |
| **F-02** | 3D Vector Joint Angle Telemetry Integrity | R1, R3, AC-Unit | `joint-angle-<jointId>`, Snapshot JSON `activeJointAngles` | Angle values format cleanly as `<deg>°` (`0..180`) or `--°` with zero `NaN`/`Infinity` |
| **F-03** | One-Euro Smoothing Toggle & Strength | R1, R3, AC-Unit | `smoothing-toggle`, `slider-smoothing-strength`, Snapshot JSON `smoothing` | Toggle & `[0, 1]` strength slider update DOM state and Snapshot JSON `smoothing` fields accurately |
| **F-04** | Snapshot JSON Export & 33-Landmark Schema | R3, AC-Unit | `snapshot-json-button`, Playwright `download` event | Downloads `pose-snapshot-*.json` with valid ISO timestamp, `schemaVersion: "1.0.0"`, 33 finite `{index,name,x,y,z,visibility}` items |
| **F-05** | MediaPipe Pose Engine & Delegate Fallback | R1, R3 | Model init (`Lite`/`Full`/`Heavy`), delegate badge (`GPU`/`CPU`), network `.task`/`.wasm` | Initializes `PoseLandmarker` cleanly under headless WebGL/SwiftShader (GPU or safe CPU fallback) with `0` errors |
| **F-06** | Live Confidence Options & Model Disposal | R1, R3 | `slider-detection-confidence`, `slider-tracking-confidence`, `model-lite`/`full`/`heavy` | Sliders update live without re-downloading `.task`; model switches dispose old instance cleanly |
| **F-07** | Overlay & Blackout Rendering Modes | R2 | `mode-overlay`, `mode-blackout`, `pose-video`, `pose-canvas` | Overlay shows `<video>` + `<canvas>`; Blackout hides `<video>` and renders dark grid onto `<canvas>` |
| **F-08** | Smart Coloring & Visibility Dimming Canvas | R2 | `pose-canvas` 2D pixel/rendering health across Overlay & Blackout | Canvas maintains crisp HiDPI backing store, renders without context/render errors, and handles visibility states |
| **F-09** | Mirror Mode (Video-Only Flip, Upright Text) | R2 | `mirror-toggle`, computed CSS `transform`/`scale` on `pose-video` vs `pose-canvas`, Snapshot `mirrored` | Mirror ON flips `pose-video` (`transform` `m11 === -1` or `scale` `-1...`) while keeping `pose-canvas` unflipped on both |
| **F-10** | Status & Error Overlay Screens + Retry | R4, AC-Browser | `status-overlay`, `retry-button`, camera/model lifecycle states | Clear actionable screens for `requesting-camera`, `loading-model`, `permission-denied`, `no-camera`, `error` + Retry recovery |
| **F-11** | Selectable 8-Joint Angle Controls | R3 | `joint-toggle-<jointId>` (x8), `joint-angle-<jointId>` (x8), Snapshot `activeJointAngles` | All 8 anatomical joints can be toggled on/off independently; UI readouts & Snapshot JSON reflect exact selection |
| **F-12** | Performance HUD Telemetry | R3 | `hud-fps`, `hud-inference-ms`, `hud-visible-joints`, delegate text | Displays numeric FPS, finite `inference ms`, `0..33 / 33` visible joints, and active delegate (`GPU`/`CPU`) |
| **F-13** | Responsive Collapsible Cybernetic ControlPanel | R3, R4, AC-Browser | `panel-toggle`, `control-panel`, desktop (`1280x720`) & mobile (`390x844`) viewports | `panel-toggle` updates `aria-expanded` & collapses/expands `control-panel`; stage expands cleanly without overflow |
| **F-14** | Synchronized `rAF` Stream & Lifecycle Loop | R1, R4, AC-Browser | `pose-video` (`readyState >= 2`), `pose-canvas` sizing, sustained `rAF` HUD ticks | Live video stream plays continuously, aspect-locked canvas tracks video geometry, and HUD updates over time |

---

## 4. Coverage Threshold Verification

- **Inventoried Features (`N`)**: `14` (`F-01`..`F-14`)
- **Formula Threshold**: `~11 * N + max(5, N / 2)` = `11 * 14 + max(5, 7)` = **`161` test cases**

| Tier | Description | Spec File | Formula Minimum | Designed Count | Status |
|---|---|---|---|---|---|
| **Tier 1** | Feature Coverage (Happy Path) | `e2e/tier1-features.spec.ts` | `>= 5` total (`5 * N` = 70 for full parity) | **70** (`5` per feature `F-01`..`F-14`) | Meets (`70 >= 5`) |
| **Tier 2** | Boundary & Corner Cases | `e2e/tier2-boundaries.spec.ts` | `>= 5` per feature (`5 * 14` = 70) | **70** (`5` per feature `F-01`..`F-14`) | Meets (`70 >= 70`) |
| **Tier 3** | Cross-Feature Pairwise Combinations | `e2e/tier3-combinations.spec.ts` | `>= N` (`14`) | **14** | Meets (`14 >= 14`) |
| **Tier 4** | Real-World Application Scenarios | `e2e/tier4-scenarios.spec.ts` | `>= max(5, N/2)` (`7`) | **7** | Meets (`7 >= 7`) |
| **Total** | **Full 4-Tier Matrix** | `e2e/tier*.spec.ts` | **`161`** (`11*14 + 7`) | **`161`** | **Exact Match (`161/161`)** |

> **Execution Strategy (Parameterized Playwright Tests)**: To avoid 161 redundant Vite/Chromium/WASM boots while preserving 100% assertion coverage, specs use structured `test(...)` blocks with explicit `test.step(...)` case IDs inside grouped `test.describe` suites. Every Case ID below maps 1-to-1 to an implemented assertion step in the spec files.

---

## 5. 4-Tier Playwright Test Matrix

### 5.1 Tier 1 — Feature Coverage (`e2e/tier1-features.spec.ts` — 70 Cases)

| Case ID | Feature | Test Case Description & Opaque-Box Assertion |
|---|---|---|
| `T1-F01-1` | F-01 | Root page loads at `http://localhost:4173` with HTTP 200 and non-empty title (`Pose Skeleton HUD`). |
| `T1-F01-2` | F-01 | Dark cybernetic root styling active (computed body/root background RGB luminance is dark `< 30`). |
| `T1-F01-3` | F-01 | Local WASM assets (`/wasm/vision_wasm_internal.js` & `.wasm`) respond with HTTP 200 (zero 404s). |
| `T1-F01-4` | F-01 | Zero uncaught `pageerror` events from before `goto('/')` through ready state. |
| `T1-F01-5` | F-01 | Zero `console.error` logs from before `goto('/')` through ready state. |
| `T1-F02-1` | F-02 | When all 8 joints are enabled, all 8 `joint-angle-<jointId>` readouts render in DOM on ready state. |
| `T1-F02-2` | F-02 | Active angle readout text strictly matches `/^(\d+(\.\d+)?\|--)\s*°$/` (never `NaN`/`undefined`). |
| `T1-F02-3` | F-02 | When no human pose is present on fake stream, active readouts display `--°` cleanly without error. |
| `T1-F02-4` | F-02 | Snapshot JSON `activeJointAngles` lists objects with `{ joint, label, degrees }` where `degrees` is `null` or `0 <= degrees <= 180`. |
| `T1-F02-5` | F-02 | Toggling any joint off removes its entry from Snapshot JSON `activeJointAngles`. |
| `T1-F03-1` | F-03 | `smoothing-toggle` is present, interactive, and reflects boolean state (`aria-pressed` or `:checked`). |
| `T1-F03-2` | F-03 | `slider-smoothing-strength` is present (`type="range"`, `min="0"`, `max="1"`, `step="0.05"`). |
| `T1-F03-3` | F-03 | Adjusting `slider-smoothing-strength` (e.g. `0.75`) updates displayed strength value in ControlPanel. |
| `T1-F03-4` | F-03 | Snapshot JSON `smoothing.enabled` matches `smoothing-toggle` state (both `true` and `false`). |
| `T1-F03-5` | F-03 | Snapshot JSON `smoothing.strength` matches `slider-smoothing-strength` numeric value (`0.75`). |
| `T1-F04-1` | F-04 | `snapshot-json-button` is visible, enabled, and accessible (`"Snapshot JSON"`). |
| `T1-F04-2` | F-04 | Clicking `snapshot-json-button` triggers a browser `download` matching `pose-snapshot-*.json`. |
| `T1-F04-3` | F-04 | Downloaded JSON parses cleanly and contains `schemaVersion: "1.0.0"`, valid ISO-8601 `timestamp`, finite `timestampMs`, and boolean `poseDetected`. |
| `T1-F04-4` | F-04 | Downloaded JSON `landmarks` array has exact length `33` (`index` `0..32`, non-empty `name`, finite numeric `x, y, z, visibility`). |
| `T1-F04-5` | F-04 | Downloaded JSON metadata (`model`, `mirrored`, `smoothing`, `activeJointAngles`) matches current UI control state. |
| `T1-F05-1` | F-05 | Default model (`Lite`) initializes cleanly and transitions app out of `loading-model` into `ready`. |
| `T1-F05-2` | F-05 | `model-lite`, `model-full`, `model-heavy` selectors are all rendered and selectable in `control-panel`. |
| `T1-F05-3` | F-05 | Switching `Lite` → `Full` fetches `pose_landmarker_full.task` (200), updates active model state, and preserves stream (`0` errors). |
| `T1-F05-4` | F-05 | Switching to `Heavy` fetches `pose_landmarker_heavy.task` (200), updates active model state, and preserves stream (`0` errors). |
| `T1-F05-5` | F-05 | Active delegate badge reports `"GPU"` or `"CPU"` in HUD/panel without unhandled WebGL/SwiftShader failures. |
| `T1-F06-1` | F-06 | `slider-detection-confidence` exists (`min="0.1"`, `max="0.95"`, `step="0.05"`) and displays numeric readout. |
| `T1-F06-2` | F-06 | `slider-tracking-confidence` exists (`min="0.1"`, `max="0.95"`, `step="0.05"`) and displays numeric readout. |
| `T1-F06-3` | F-06 | Changing `slider-detection-confidence` (e.g. `0.7`) updates UI value without re-fetching `.task` model files. |
| `T1-F06-4` | F-06 | Changing `slider-tracking-confidence` (e.g. `0.65`) updates UI value without re-fetching `.task` model files. |
| `T1-F06-5` | F-06 | Switching `Full` → `Lite` after model load leaves only one active model running and HUD FPS continuing smoothly. |
| `T1-F07-1` | F-07 | `mode-overlay` and `mode-blackout` controls exist and default to `Overlay` active. |
| `T1-F07-2` | F-07 | In `Overlay` mode, `pose-video` is visible (`opacity > 0`, non-zero bounding box) beneath `pose-canvas`. |
| `T1-F07-3` | F-07 | Switching to `Blackout` (`mode-blackout`) hides `pose-video` visually (`opacity === '0'` or `visibility === 'hidden'` or `display === 'none'`). |
| `T1-F07-4` | F-07 | In `Blackout` mode, `pose-canvas` renders a non-blank dark cybernetic grid backdrop (sampled canvas pixels contain opaque dark background + grid strokes). |
| `T1-F07-5` | F-07 | Switching `Blackout` → `Overlay` restores `pose-video` visibility and removes opaque blackout fill from `pose-canvas`. |
| `T1-F08-1` | F-08 | `pose-canvas` uses a 2D rendering context (`getContext('2d') !== null`) separate from MediaPipe's offscreen WebGL canvas. |
| `T1-F08-2` | F-08 | `pose-canvas` backing store dimensions (`canvas.width`, `canvas.height`) match CSS box × `devicePixelRatio` (`>= clientWidth`). |
| `T1-F08-3` | F-08 | Continuous `rAF` redraw loop in both Overlay and Blackout modes executes with `0` canvas runtime errors. |
| `T1-F08-4` | F-08 | Snapshot `landmarks[*].visibility` values are clamped within `[0, 1]` for all 33 indices. |
| `T1-F08-5` | F-08 | Blackout mode grid rendering preserves crisp dark-background contrast without clipping stage borders. |
| `T1-F09-1` | F-09 | `mirror-toggle` exists in `control-panel` and toggles active state cleanly. |
| `T1-F09-2` | F-09 | When Mirror is OFF, `pose-video` is unflipped on both `transform` (`m11 === 1` or `'none'`) and `scale`. |
| `T1-F09-3` | F-09 | When Mirror is ON, `pose-video` is horizontally flipped via CSS `transform` (`m11 === -1`) OR CSS `scale` (`startsWith('-1')`). |
| `T1-F09-4` | F-09 | When Mirror is ON, `pose-canvas` remains unflipped on BOTH `transform` (`m11 === 1` or `'none'`) and `scale` (upright on-canvas angle pills). |
| `T1-F09-5` | F-09 | Mirror state (`true`/`false`) is faithfully exported in Snapshot JSON `mirrored` field. |
| `T1-F10-1` | F-10 | `status-overlay` displays descriptive status text during initial camera/model loading before transitioning to `ready`. |
| `T1-F10-2` | F-10 | When `getUserMedia` rejects with `NotAllowedError`, `status-overlay` shows permission-denied guidance and `retry-button`. |
| `T1-F10-3` | F-10 | When `getUserMedia` rejects with `NotFoundError`, `status-overlay` shows missing-camera guidance and `retry-button`. |
| `T1-F10-4` | F-10 | In real headless Chromium without fake-device flags (`NotSupportedError`), `status-overlay` renders an actionable error screen + `retry-button` without crashing. |
| `T1-F10-5` | F-10 | Clicking `retry-button` after camera availability is restored transitions out of `status-overlay` to active `ready` video/HUD. |
| `T1-F11-1` | F-11 | All 8 `joint-toggle-<jointId>` controls (`left_elbow`..`right_knee`) exist in `control-panel`. |
| `T1-F11-2` | F-11 | When enabled via `joint-toggle-<jointId>`, each of the 8 joints displays its corresponding `joint-angle-<jointId>` readout. |
| `T1-F11-3` | F-11 | Toggling an active joint off via `joint-toggle-<jointId>` updates its pressed/checked state to `false`. |
| `T1-F11-4` | F-11 | Toggling an inactive joint on via `joint-toggle-<jointId>` sets its pressed/checked state to `true` and includes it in `activeJointAngles`. |
| `T1-F11-5` | F-11 | Selecting specific upper-body vs lower-body subsets updates Snapshot JSON `activeJointAngles` length & IDs to match. |
| `T1-F12-1` | F-12 | `hud-fps` renders a valid non-negative integer/numeric FPS value. |
| `T1-F12-2` | F-12 | `hud-inference-ms` renders a finite non-negative latency string containing `ms`. |
| `T1-F12-3` | F-12 | `hud-visible-joints` strictly matches `/^(\d+)\s*\/\s*33$/` with `0 <= n <= 33`. |
| `T1-F12-4` | F-12 | `hud-fps` and `hud-inference-ms` stay live (remain numeric / update across multiple rendered frames). |
| `T1-F12-5` | F-12 | Performance HUD remains visible even when `control-panel` is collapsed. |
| `T1-F13-1` | F-13 | `panel-toggle` defaults to `aria-expanded="true"` on desktop (`1280x720`) with `control-panel` visible. |
| `T1-F13-2` | F-13 | Clicking `panel-toggle` collapses `control-panel` (`aria-expanded="false"`) and increases stage width. |
| `T1-F13-3` | F-13 | Clicking `panel-toggle` again re-expands `control-panel` (`aria-expanded="true"`) with all controls accessible. |
| `T1-F13-4` | F-13 | Desktop viewport (`1280x720`) has zero unwanted horizontal document overflow (`scrollWidth <= innerWidth`). |
| `T1-F13-5` | F-13 | Mobile viewport (`390x844`) stacks stage and `control-panel` cleanly with zero horizontal overflow (`scrollWidth <= innerWidth`). |
| `T1-F14-1` | F-14 | `pose-video` receives an active `MediaStream` (`readyState >= 2`, `videoWidth > 0`, `videoHeight > 0`). |
| `T1-F14-2` | F-14 | `pose-video` and `pose-canvas` share aligned bounding boxes (`|videoRect.width - canvasRect.width| <= 2` and height `<= 2`). |
| `T1-F14-3` | F-14 | Sustained `rAF` execution over `2s` produces no monotonic timestamp errors or frame stalls. |
| `T1-F14-4` | F-14 | Page reload (`page.reload()`) cleanly tears down previous camera stream/engine and re-initializes with `0` errors. |
| `T1-F14-5` | F-14 | Navigating away (`page.goto('about:blank')`) after active streaming triggers zero unload/cleanup errors. |

---

### 5.2 Tier 2 — Boundary & Corner Cases (`e2e/tier2-boundaries.spec.ts` — 70 Cases)

| Case ID | Feature | Boundary / Corner Condition & Assertion |
|---|---|---|
| `T2-F01-1` | F-01 | Deep-link / query params (`/?debug=1`) boot cleanly with HTTP 200 and `0` console/page errors. |
| `T2-F01-2` | F-01 | High-DPI (`deviceScaleFactor: 2`) browser context renders without layout clipping or console errors. |
| `T2-F01-3` | F-01 | Reduced-motion media preference (`prefers-reduced-motion: reduce`) loads without UI breakage. |
| `T2-F01-4` | F-01 | Blocked third-party favicon/unrelated request does not crash root shell; app mounts cleanly. |
| `T2-F01-5` | F-01 | Double-reload stress (2 consecutive `reload()` calls) leaves zero lingering error listeners or broken WASM state. |
| `T2-F02-1` | F-02 | Zero-pose synthetic stream produces zero `NaN` strings anywhere inside `control-panel` or `hud-*`. |
| `T2-F02-2` | F-02 | Snapshot JSON with `poseDetected: false` still preserves valid `null` degrees for every selected joint (never `NaN`/`undefined`). |
| `T2-F02-3` | F-02 | Snapshot JSON when `0` joints are selected produces `activeJointAngles: []` (empty array, valid schema). |
| `T2-F02-4` | F-02 | Snapshot JSON when all `8` joints are selected produces `activeJointAngles.length === 8` with unique joint IDs. |
| `T2-F02-5` | F-02 | Toggling joints rapidly 10 times produces valid final angle array with zero duplicate joint entries. |
| `T2-F03-1` | F-03 | `slider-smoothing-strength` clamped at minimum boundary `0` (`0.00`) is accepted and exported as `strength: 0`. |
| `T2-F03-2` | F-03 | `slider-smoothing-strength` clamped at maximum boundary `1` (`1.00`) is accepted and exported as `strength: 1`. |
| `T2-F03-3` | F-03 | Setting out-of-range value (`-0.5` or `1.5`) via DOM input clamps cleanly to `[0, 1]` without errors. |
| `T2-F03-4` | F-03 | Rapid `smoothing-toggle` clicks (6x in succession) leave UI and Snapshot `smoothing.enabled` synchronized. |
| `T2-F03-5` | F-03 | Adjusting `slider-smoothing-strength` while smoothing is disabled preserves strength value cleanly in Snapshot JSON. |
| `T2-F04-1` | F-04 | Clicking `snapshot-json-button` immediately when `poseDetected === false` downloads valid zeroed 33-landmark fallback (`x:0,y:0,z:0,visibility:0`). |
| `T2-F04-2` | F-04 | Clicking `snapshot-json-button` twice within 200ms yields two distinct valid downloads without blob URL errors. |
| `T2-F04-3` | F-04 | Every one of the 33 landmarks in downloaded JSON has exact sequential `index === i` (`0..32`) and valid non-empty `name`. |
| `T2-F04-4` | F-04 | Clicking `snapshot-json-button` when `0` joints are selected still produces valid 33 landmarks and `activeJointAngles: []`. |
| `T2-F04-5` | F-04 | Downloaded `pose-snapshot-*.json` filename contains no spaces or illegal path characters (`/`, `\`, `:`). |
| `T2-F05-1` | F-05 | Rapid model button clicks (`Lite` → `Heavy` → `Lite` with delayed `Heavy`) settle on final selection (`Lite`) without race condition or `console.error`. |
| `T2-F05-2` | F-05 | Clicking the already-active model button (`Lite` while `Lite` active) is a safe no-op (no duplicate `.task` network fetch or crash). |
| `T2-F05-3` | F-05 | Model download failure (intercepted `.task` returning HTTP 500) transitions cleanly to actionable `error` `status-overlay` + `retry-button` (zero uncaught `pageerror`). |
| `T2-F05-4` | F-05 | Slow model fetch (`500ms+` route delay) keeps `loading-model` status visible until model resolves, then enters `ready`. |
| `T2-F05-5` | F-05 | Model recovery via `retry-button` after transient `.task` HTTP 500 succeeds cleanly when network recovers. |
| `T2-F06-1` | F-06 | `slider-detection-confidence` at minimum boundary (`0.1`) updates UI and engine safely (`0` errors). |
| `T2-F06-2` | F-06 | `slider-detection-confidence` at maximum boundary (`0.95`) updates UI and engine safely (`0` errors). |
| `T2-F06-3` | F-06 | `slider-tracking-confidence` at minimum boundary (`0.1`) updates UI and engine safely (`0` errors). |
| `T2-F06-4` | F-06 | `slider-tracking-confidence` at maximum boundary (`0.95`) updates UI and engine safely (`0` errors). |
| `T2-F06-5` | F-06 | Rapid step-sweeping both confidence sliders (`0.1 -> 0.5 -> 0.95`) triggers zero `setOptions` runtime exceptions. |
| `T2-F07-1` | F-07 | Clicking `mode-overlay` when already in `Overlay` mode is idempotent (no flicker/error). |
| `T2-F07-2` | F-07 | Clicking `mode-blackout` when already in `Blackout` mode is idempotent. |
| `T2-F07-3` | F-07 | Rapidly toggling `Overlay` ↔ `Blackout` 8 times leaves final `<video>` visibility & `<canvas>` mode consistent with active button. |
| `T2-F07-4` | F-07 | Switching to `Blackout` before or immediately after window resize maintains grid coverage across full resized canvas. |
| `T2-F07-5` | F-07 | In `Blackout` mode, `pose-video` remains in DOM (`pose-video` count === 1) so stream playback is not destroyed on switch back. |
| `T2-F08-1` | F-08 | Resizing viewport from `1280x720` down to `640x480` dynamically updates `pose-canvas` dimensions without `0x0` collapse. |
| `T2-F08-2` | F-08 | Resizing viewport up to `1920x1080` preserves aspect-locked `pose-canvas` without stretching outside stage. |
| `T2-F08-3` | F-08 | Collapsing `control-panel` triggers `pose-canvas` resize cleanly without clearing Blackout grid. |
| `T2-F08-4` | F-08 | At `deviceScaleFactor: 2`, canvas drawing buffer scales proportionally without overflowing container bounds. |
| `T2-F08-5` | F-08 | Zero-landmark frames (`0 / 33`) render clean Overlay (transparent) / Blackout (grid only) without stray artifact dots. |
| `T2-F09-1` | F-09 | Rapid `mirror-toggle` clicks (7x odd count) result in mirrored `<video>` and unflipped `<canvas>` (`transform` & `scale`). |
| `T2-F09-2` | F-09 | Rapid `mirror-toggle` clicks (8x even count) restore unmirrored `<video>` and unflipped `<canvas>`. |
| `T2-F09-3` | F-09 | Keyboard activation (`Space` / `Enter`) on focused `mirror-toggle` toggles Mirror mode reliably. |
| `T2-F09-4` | F-09 | Toggling Mirror ON while `control-panel` is expanded vs collapsed preserves `<video>` flip & `<canvas>` upright state. |
| `T2-F09-5` | F-09 | Snapshot JSON after rapid Mirror toggles reflects exact final UI `mirrored` boolean. |
| `T2-F10-1` | F-10 | `NotReadableError` from `getUserMedia` (camera busy) displays descriptive `error` screen + `retry-button` without crash. |
| `T2-F10-2` | F-10 | `OverconstrainedError` from `getUserMedia` displays descriptive `error` screen + `retry-button` without crash. |
| `T2-F10-3` | F-10 | Missing `navigator.mediaDevices` (`undefined` simulated via `addInitScript`) renders graceful error screen + `retry-button`. |
| `T2-F10-4` | F-10 | Multiple consecutive failed `retry-button` clicks (`NotAllowedError` persisting) stay stably in `permission-denied` without duplicate error listeners. |
| `T2-F10-5` | F-10 | Recovery from `NotFoundError` → Retry success clears `status-overlay` and hides `retry-button`. |
| `T2-F11-1` | F-11 | Deselecting all 8 joints leaves `0` active joints without UI crash or layout collapse. |
| `T2-F11-2` | F-11 | Selecting all 8 joints simultaneously displays all 8 `joint-angle-*` readouts cleanly within panel scroll bounds. |
| `T2-F11-3` | F-11 | Toggling only left-side joints (`left_elbow`, `left_shoulder`, `left_hip`, `left_knee`) updates Snapshot JSON `activeJointAngles` to exact 4 left IDs. |
| `T2-F11-4` | F-11 | Toggling only right-side joints (`right_elbow`, `right_shoulder`, `right_hip`, `right_knee`) updates Snapshot JSON `activeJointAngles` to exact 4 right IDs. |
| `T2-F11-5` | F-11 | Keyboard interaction (`Space`/`Enter`) on `joint-toggle-left_elbow` toggles selection cleanly. |
| `T2-F12-1` | F-12 | Initial startup frame before first inference settles does not display `NaN` or negative FPS in `hud-fps`. |
| `T2-F12-2` | F-12 | `hud-inference-ms` never shows negative numbers or `Infinity ms`. |
| `T2-F12-3` | F-12 | `hud-visible-joints` count stays strictly `>= 0` and `<= 33` across model switches. |
| `T2-F12-4` | F-12 | At narrow mobile viewport (`390x844`), `hud-fps`, `hud-inference-ms`, and `hud-visible-joints` remain visible without overlapping `panel-toggle`. |
| `T2-F12-5` | F-12 | During `Blackout` mode, Performance HUD remains readable above dark canvas (`contrast` / non-hidden). |
| `T2-F13-1` | F-13 | Ultra-narrow mobile viewport (`320x568`) renders with zero horizontal page scroll (`scrollWidth <= innerWidth`). |
| `T2-F13-2` | F-13 | Tablet portrait viewport (`768x1024`) keeps stage and controls accessible without clipping. |
| `T2-F13-3` | F-13 | Rapid `panel-toggle` clicks (5x) keep `aria-expanded` synchronized with `control-panel` visibility. |
| `T2-F13-4` | F-13 | Collapsing `control-panel` then resizing window maintains collapsed state (`aria-expanded="false"`). |
| `T2-F13-5` | F-13 | All interactive buttons/sliders inside `control-panel` are reachable via scroll/keyboard when panel height is constrained (`1024x600`). |
| `T2-F14-1` | F-14 | Delayed `getUserMedia` resolution (`500ms`) holds `requesting-camera` cleanly before entering `loading-model` / `ready`. |
| `T2-F14-2` | F-14 | StrictMode / fast reload mid-initialization (`goto` then immediate `reload`) completes cleanly with `0` uncaught promise rejections. |
| `T2-F14-3` | F-14 | Viewport resize while streaming keeps `pose-video` and `pose-canvas` aligned (`<= 2px` delta). |
| `T2-F14-4` | F-14 | Tab visibility change simulation (`visibilitychange`) resumes `rAF` loop with `0` monotonic timestamp errors. |
| `T2-F14-5` | F-14 | Sustained 3-second idle stream keeps `hud-fps` > 0 (or non-stalled) and zero console/page errors. |

---

### 5.3 Tier 3 — Cross-Feature Pairwise Combinations (`e2e/tier3-combinations.spec.ts` — 14 Cases)

| Case ID | Features Combined | Pairwise Interaction & Opaque-Box Assertion |
|---|---|---|
| `T3-01` | F-07 × F-09 | **Blackout + Mirror ON**: `pose-video` hidden, `pose-canvas` renders dark grid (`transform` & `scale` unflipped), `<video>` horizontally flipped, Snapshot JSON records `mirrored: true`. |
| `T3-02` | F-05 × F-03 × F-04 | **Model (`Full`) + Smoothing OFF + Snapshot**: Switch to `Full`, disable smoothing, export Snapshot JSON → `model: "full"`, `smoothing.enabled: false`, 33 landmarks valid. |
| `T3-03` | F-13 × F-07 × F-11 | **Panel Collapse + Blackout + Joint Subset**: Toggle custom joint subset, switch to `Blackout`, collapse `control-panel` (`aria-expanded="false"`), verify HUD & stage remain active and re-expand preserves joint selection. |
| `T3-04` | F-06 × F-05 | **Confidence Sliders + Model Switch**: Set detection `0.8` & tracking `0.75`, switch `Lite` → `Full` → `Lite`, verify slider values remain preserved across model disposal/reload. |
| `T3-05` | F-10 × F-07 × F-09 | **Error Recovery + User Settings**: Recover from `NotAllowedError` via `retry-button` → confirm `Blackout`, `Mirror`, and all HUD controls remain interactive after recovery. |
| `T3-06` | F-03 × F-06 × F-12 | **Smoothing Strength Max (`1.0`) + Low Confidence (`0.1`) + Performance HUD**: Verify `hud-fps` and `hud-inference-ms` stay live without lag spikes or errors. |
| `T3-07` | F-11 × F-04 × F-09 | **8-Joint Toggle + Mirror ON + Snapshot JSON**: Select lower-body joints (`left_hip`, `right_hip`, `left_knee`, `right_knee`), enable Mirror, export Snapshot → `mirrored: true` and exact 4 lower-body joints in `activeJointAngles`. |
| `T3-08` | F-13 × F-09 × F-14 | **Mobile Viewport (`390x844`) + Mirror ON + Aspect Alignment**: Verify `<video>` flipped, `<canvas>` unflipped, and bounding boxes aligned within mobile layout. |
| `T3-09` | F-05 × F-04 (`Heavy`) | **Model `Heavy` + Snapshot JSON**: Switch to `Heavy`, wait for ready, export Snapshot JSON → `model: "heavy"`, 33 finite landmarks. |
| `T3-10` | F-07 × F-08 × F-13 | **Overlay ↔ Blackout across HiDPI Resize + Panel Toggle**: Toggle panel & viewport size while switching Overlay ↔ Blackout; verify canvas pixel buffers remain valid. |
| `T3-11` | F-10 × F-13 | **Permission Denied / No-Camera on Mobile (`390x844`)**: Verify `status-overlay` and `retry-button` remain centered, unobstructed, and clickable on mobile viewport. |
| `T3-12` | F-06 × F-03 × F-04 | **Confidence (`0.95`) + Smoothing (`0.0`, disabled) + Snapshot**: Verify JSON accurately reflects `smoothing: { enabled: false, strength: 0 }` alongside strict threshold state. |
| `T3-13` | F-12 × F-07 × F-05 | **Performance HUD across Overlay/Blackout & Model Switch (`Lite`→`Full`)**: Verify `hud-fps`, `hud-inference-ms`, `hud-visible-joints` remain properly formatted before, during, and after switch. |
| `T3-14` | F-11 × F-13 × F-14 | **Collapse Panel + Resize + Re-expand + Joint Angle Inspection**: Verify state persistence of selected joints and live `°`/`--°` formatting across layout transitions. |

---

### 5.4 Tier 4 — Real-World Application Scenarios (`e2e/tier4-scenarios.spec.ts` — 7 Cases)

| Case ID | Scenario Title | End-to-End User Workflow |
|---|---|---|
| `T4-01` | **Full Biomechanics Lab Calibration & Snapshot Export** | Boot app → verify ready stream & `hud-fps` → tune `detectionConfidence = 0.65`, `trackingConfidence = 0.60` → enable smoothing at `0.70` → select upper-body joints (`left_elbow`, `right_elbow`, `left_shoulder`, `right_shoulder`) → toggle Mirror ON → switch to Blackout mode → export Snapshot JSON → validate schema (`33` landmarks, `mirrored: true`, `smoothing`, `4` upper-body angles) with `0` console/page errors. |
| `T4-02` | **Camera Permission Denied → User Grants Permission → Retry Recovery** | Start session with `NotAllowedError` mock → verify `status-overlay` explains denial and shows `retry-button` → user grants permission (mock switches to live fake stream) → click `retry-button` → verify `status-overlay` disappears, `pose-video` streams (`readyState >= 2`), `hud-fps` ticks, and Snapshot JSON succeeds. |
| `T4-03` | **Missing Webcam (`NotFoundError`) → Plug-In → Retry → Full Pose Inspection** | Start session with `NotFoundError` mock → verify no-camera screen + `retry-button` → simulate webcam attachment → click `retry-button` → switch to `Blackout` → toggle all 8 joints → verify `hud-visible-joints` (`X / 33`) & zero uncaught errors. |
| `T4-04` | **Progressive Model Upgrade (`Lite` → `Full` → `Heavy`) & Cross-Model Snapshot Audit** | Start on `Lite` and export baseline Snapshot (`model: "lite"`) → upgrade to `Full` (`model: "full"`) → upgrade to `Heavy` (`model: "heavy"`) → confirm smooth video/canvas continuity, active delegate badge (`GPU`/`CPU`), and valid Snapshot JSON on `Heavy`. |
| `T4-05` | **Presenter Mode (Collapsed Panel + Mirror + Blackout 3D Inspection)** | Configure lower-body tracking (`left_hip`, `right_hip`, `left_knee`, `right_knee`) → enable Mirror (`pose-video` flipped, `pose-canvas` unflipped) → switch to `Blackout` → collapse `control-panel` (`aria-expanded="false"`) for full-stage presentation → verify Performance HUD remains visible → re-expand panel and export Snapshot JSON. |
| `T4-06` | **Mobile Field Assessment (`390x844`) Workflow** | Open app on mobile viewport → verify no horizontal overflow → inspect camera stream & HUD → open/collapse panel → toggle smoothing & `Overlay`↔`Blackout` → export Snapshot JSON cleanly. |
| `T4-07` | **Sustained Monitoring & Reload Resilience Session** | Stream continuously for `3s` while interacting with confidence/smoothing sliders → verify zero `pageerror`/`console.error` → perform `page.reload()` → verify fresh camera/model re-initialization and functional Snapshot JSON export post-reload. |

---

## 6. Spec Layout & Verification Commands

- **Owned Files (`E2E_TRACK`)**:
  - `playwright.config.ts` — Web server (`http://localhost:4173`), `chromium-fake-webcam` project, timeouts, reporters.
  - `e2e/helpers.ts` — Opaque-box error listener (`pageerror` + `console.error`), ready-state helpers, `addInitScript` camera error/recovery injectors, `launchNoFakeDeviceSession`, Snapshot JSON validator (`33` landmarks, ISO timestamp, finite numbers), computed `transform`/`scale` flip helpers, `.task` route cache.
  - `e2e/tier1-features.spec.ts` — Cases `T1-F01-1` .. `T1-F14-5` (70 cases).
  - `e2e/tier2-boundaries.spec.ts` — Cases `T2-F01-1` .. `T2-F14-5` (70 cases).
  - `e2e/tier3-combinations.spec.ts` — Cases `T3-01` .. `T3-14` (14 cases).
  - `e2e/tier4-scenarios.spec.ts` — Cases `T4-01` .. `T4-07` (7 cases).
- **Verification Commands**:
  - `npx playwright test --list` (verify Playwright parses all specs cleanly)
  - `npm run lint` (verify `e2e/**/*.ts` and `playwright.config.ts` pass ESLint)
  - `npx tsc --noEmit --strict --target ES2022 --module ESNext --moduleResolution bundler --skipLibCheck --lib ES2022,DOM,DOM.Iterable playwright.config.ts e2e/*.ts`
  - `npx vitest run` (verify Vitest ignores `e2e/` and passes unit tests)
  - `npm run test:e2e` (executed in M5 once M4 completes `App.tsx` & `ControlPanel.tsx`)
