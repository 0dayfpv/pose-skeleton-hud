<div align="center">

# 🦾 Pose Skeleton HUD V2 — 553-Landmark Cybernetic Biometrics & Motion Lab

**Real-Time, 100% Client-Side (WebGL / WASM) Full-Body Pose (33), Articulated Hands & 5-Fingers (42), and 3D Face, Gaze-Tracking Iris & Cranial Dome (478) Telemetry Platform**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-pose.hamzafpv.com.tr-06B6D4?style=for-the-badge)](https://pose.hamzafpv.com.tr)
[![Author](https://img.shields.io/badge/🌐_Website-hamzafpv.com.tr-0F172A?style=for-the-badge)](https://hamzafpv.com.tr)

[![React](https://img.shields.io/badge/React_19-20232A?style=flat-square&logo=react&logoColor=61DAFB)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript_5.8-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite_8-646CFF?style=flat-square&logo=vite&logoColor=white)](https://vitejs.dev/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe_Tasks_Vision-553_3D_Landmarks-0097A7?style=flat-square&logo=google&logoColor=white)](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS_v4-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![Unit Tests](https://img.shields.io/badge/Vitest-73%2F73_Passed-22C55E?style=flat-square&logo=vitest&logoColor=white)](#testing--verification)
[![E2E Tests](https://img.shields.io/badge/Playwright_E2E-161%2F161_Passed-2EAD33?style=flat-square&logo=playwright&logoColor=white)](#testing--verification)
[![License: MIT](https://img.shields.io/badge/License-MIT-3B82F6?style=flat-square)](LICENSE)

</div>

---

## 📌 Overview

**Pose Skeleton HUD V2** is a high-FPS, dark cybernetic biomechanics and motion-capture laboratory running **100% locally inside the user's browser** using **WebGL / WebAssembly (`@mediapipe/tasks-vision`)** and a synchronized HiDPI HTML5 `<canvas>` rendering pipeline. Zero video frames ever leave the user's device.

Building upon the V1 33-point full-body skeleton engine, **V2** fuses **three synchronized MediaPipe Vision engines** (`PoseLandmarker` + `HandLandmarker` + `FaceLandmarker`) to track up to **553 3D anatomical landmarks simultaneously**:
- 🦴 **33 Body Pose Landmarks** (`Lite`, `Full`, and `Heavy` models with live GPU/CPU delegate fallback)
- 🖐️ **42 Hand & Finger Landmarks** (`21 × 2` hands with 5-finger phalanx articulation, fingertip target halos, and thumb-index pinch laser detection)
- 👁️ **478 3D Face, Eye, Iris & Cranial Head Landmarks** (3D facial tesselation wireframe, face oval + synthesized 3D cranial skull dome, upper/lower eyelids, gaze-tracking iris rings & pupils, eyebrows, nasal bridge, and inner/outer lips)

---

## ✨ Key Features

### 🎨 Dual Cybernetic Display Modes & Smart Coloring
- 📷 **Overlay Mode:** Semi-transparent neon skeleton, hand phalanges, and 3D face mesh rendered directly over the live webcam stream.
- 🌌 **Blackout (Pure 3D Skeleton) Mode:** Hides the camera feed and renders the 553-point biomechanics skeleton over a dark cybernetic reticle grid with **`z`-depth scaling, brightness modulation, and neon glow**.
- 🎯 **Anatomical Smart Coloring:**
  - **Left Side:** Neon Cyan (`#22d3ee`)
  - **Right Side:** Cybernetic Magenta / Coral (`#f472b6` / `#fb923c`)
  - **Torso, Spine & Midline:** Crisp Slate / Silver (`#e2e8f0`)
  - **Visibility Dimming:** Automatically fades low-confidence landmarks based on real-time visibility scores.
- 🪞 **Synchronized Mirror Mode:** Horizontally flips the `<video>` feed and `<canvas>` coordinates (`1 - x`) while keeping all on-canvas joint angle pills and hand telemetry tags **upright and readable**.

### 🖐️ V2 Articulated Hand & 5-Finger Tracking (`21 × 2 = 42` Landmarks)
- Tracks both **Left** and **Right** hands with all 5 articulated finger chains (**Thumb**, **Index**, **Middle**, **Ring**, **Pinky**).
- Automatically suppresses the coarse 3-point body pose hand triangle when high-detail 21-point hand tracking is locked, seamlessly bridging the forearm wrist to the palm.
- **Live Finger Telemetry:** Computes individual flexion/curl angles (`°`) for all 5 fingers, extended finger count (`0..5`), hand openness percentage (`0..100%`), and **Thumb–Index Pinch Detection** with a glowing neon beam.

### 👁️ V2 3D Face Mesh, Gaze-Tracking Irises & Cranial Head Shape (`478` Landmarks)
- **3D Cranial Dome & Face Oval:** Renders `FACE_LANDMARKS_FACE_OVAL` along the jawline and forehead, topped by a synthesized **3D cranial skull dome arc**, sagittal head-axis meridian, and neck spine bridge to the shoulders.
- **Gaze-Tracking Irises & Pupils (`468..477`):** Tracks upper/lower eyelids and renders glowing cyan/magenta iris rings, pupil pinpoint centers, and Blackout gaze reticles.
- **3D Face Tesselation Wireframe:** Toggleable (`toggle-face-mesh`) 2,556-edge topological surface mesh across the face.
- **Live Head & Eye Biometrics:** Real-time **3D Head Orientation (`YAW`, `PITCH`, `ROLL` in `°`)**, **Left/Right Eye Openness (`%` via Eye Aspect Ratio)**, and **Mouth Openness (`%`)**.

### 📐 Vector Joint Angle Telemetry & One-Euro Jitter Smoothing
- **8 Selectable Anatomical Body Joints:** Live 3D vector angle calculation (`±0.5°` precision) for **Left/Right Elbow, Shoulder, Hip, and Knee**, displayed both as on-canvas pills and inside the right telemetry panel.
- **Adaptive One-Euro Filter (`PoseSmoother`):** Independent velocity-adaptive low-pass filters across body, left hand, right hand, and 478-point face streams to eliminate high-frequency jitter without adding lag during fast motion.
- **One-Click Snapshot JSON Export:** Downloads a validated `pose-snapshot-<timestamp>.json` file containing all 33 body landmarks `(x, y, z, visibility)`, active joint angles, tracked 21-point hands, finger flexion telemetry, and 3D head/eye biometrics.

---

## 🏗️ Architecture & Project Structure

```text
pose_skeleton_hud_v2/
├── public/
│   └── wasm/                        # Local MediaPipe Vision WASM runtime binaries
├── src/
│   ├── engine/
│   │   ├── poseEngine.ts            # Multi-landmarker engine (Pose + Hand + Face), GPU->CPU fallback, live options & disposal
│   │   ├── poseEngine.test.ts       # 26 unit tests covering delegate fallback, model switching, race safety & monotonic timestamps
│   │   ├── geometry.ts              # 3D vector joint angles, 5-finger metrics, 3D head Yaw/Pitch/Roll, One-Euro smoother & JSON serializer
│   │   └── geometry.test.ts         # 21 unit tests for 3D math, smoothing variance reduction, hand/face biometrics & JSON schema
│   ├── components/
│   │   ├── SkeletonCanvas.tsx       # HiDPI 2D canvas renderer (Body + 21x2 Hands + 478 Face/Iris/Cranial Dome, Overlay/Blackout/Mirror)
│   │   ├── SkeletonCanvas.test.tsx  # 15 component & canvas drawing unit tests
│   │   ├── ControlPanel.tsx         # Collapsible cybernetic telemetry & control drawer
│   │   └── ControlPanel.test.tsx    # 7 component interaction unit tests
│   ├── App.tsx                      # Camera lifecycle, synchronized requestAnimationFrame loop & persistent Performance HUD
│   └── App.test.tsx                 # 4 full-app integration unit tests
└── e2e/                             # 4-tier (161 test cases) Playwright headless Chromium E2E test suite
```

---

## 🚀 Getting Started

### Prerequisites
- **Node.js** `>= 20` and **npm**

### Local Development

```bash
# Install dependencies
npm install

# Start Vite development server
npm run dev
```

### Production Build

```bash
# Run TypeScript check and build production bundle into dist/
npm run build

# Preview production build locally
npm run preview
```

---

## 🧪 Testing & Verification

```bash
# Run ESLint static analysis (0 errors)
npm run lint

# Run TypeScript compiler verification (0 errors)
npm run typecheck

# Run Vitest unit test suite (73/73 unit tests)
npx vitest run

# Run Playwright headless Chromium E2E test suite
npm run test:e2e
```

---

## 📄 License & Copyright

The source code is licensed under the [MIT License](LICENSE). All personal content, visuals, and branding remain All Rights Reserved.

---

<div align="center">

© 2025–2026 [hamzafpv.com.tr](https://hamzafpv.com.tr) • All rights reserved.

</div>
