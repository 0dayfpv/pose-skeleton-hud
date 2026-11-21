import {
  FaceLandmarker,
  HandLandmarker,
} from "@mediapipe/tasks-vision";
import {
  useEffect,
  useRef,
  useState,
  type ReactElement,
  type RefObject,
  type SyntheticEvent,
} from "react";
import {
  AlertTriangle,
  Camera,
  CameraOff,
  Loader2,
  RefreshCw,
  ShieldAlert,
} from "lucide-react";
import {
  JOINT_DEFINITION_MAP,
  computeHandMetrics,
  type AppStatus,
  type JointId,
  type PersonId,
  type Point3D,
  type TrackedHand,
  type TrackedPerson,
  type ViewMode,
} from "../engine/geometry";

export interface DrawSkeletonFrameParams {
  landmarks: readonly Point3D[] | null;
  hands?: readonly TrackedHand[];
  faceLandmarks?: readonly Point3D[] | null;
  persons?: readonly TrackedPerson[];
  showFaceMesh?: boolean;
  angles?: Partial<Record<JointId, number | null>>;
  selectedJoints?: readonly JointId[];
  viewMode: ViewMode;
  mirrored: boolean;
  minVisibility?: number;
}

export type DrawSkeletonParams = DrawSkeletonFrameParams;

export interface SkeletonCanvasProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  landmarks: readonly Point3D[] | null;
  hands?: readonly TrackedHand[];
  faceLandmarks?: readonly Point3D[] | null;
  persons?: readonly TrackedPerson[];
  showFaceMesh?: boolean;
  angles: Partial<Record<JointId, number | null>>;
  selectedJoints: readonly JointId[];
  viewMode: ViewMode;
  mirrored: boolean;
  status: AppStatus;
  statusMessage?: string;
  onRetry?: () => void;
  videoAspectRatio?: number;
}

const DEFAULT_ASPECT_RATIO = 4 / 3;
const COLOR_LEFT = "#22d3ee";
const COLOR_RIGHT = "#f472b6";
const COLOR_TORSO = "#e2e8f0";
const COLOR_MIDLINE = "#94a3b8";

const P2_COLOR_LEFT = "#10b981";
const P2_COLOR_RIGHT = "#f59e0b";
const P2_COLOR_TORSO = "#c4b5fd";
const P2_COLOR_MIDLINE = "#a78bfa";
const BLACKOUT_BG = "#05080f";

const FINGERTIP_INDICES = new Set<number>([4, 8, 12, 16, 20]);
const POSE_FACE_INDICES = new Set<number>([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const POSE_LEFT_HAND_INDICES = new Set<number>([17, 19, 21]);
const POSE_RIGHT_HAND_INDICES = new Set<number>([18, 20, 22]);

const NOSE_BRIDGE_CHAIN = [168, 6, 197, 195, 5, 4, 1, 2] as const;
const NOSE_BASE_CHAIN = [98, 97, 2, 326, 327] as const;

// Verified 35 edges from @mediapipe/tasks-vision@1.2.3 PoseLandmarker.POSE_CONNECTIONS
const POSE_CONNECTIONS: readonly (readonly [number, number])[] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 7],
  [0, 4],
  [4, 5],
  [5, 6],
  [6, 8],
  [9, 10],
  [11, 12],
  [11, 13],
  [13, 15],
  [15, 17],
  [15, 19],
  [15, 21],
  [17, 19],
  [12, 14],
  [14, 16],
  [16, 18],
  [16, 20],
  [16, 22],
  [18, 20],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [24, 26],
  [25, 27],
  [26, 28],
  [27, 29],
  [28, 30],
  [29, 31],
  [30, 32],
  [27, 31],
  [28, 32],
];

const LEFT_LANDMARKS = new Set<number>([
  1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
]);
const RIGHT_LANDMARKS = new Set<number>([
  4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
]);
const TORSO_EDGE_KEYS = new Set<string>(["11-12", "23-24", "11-23", "12-24"]);

type BodyRegion = "left" | "right" | "torso" | "midline";

interface StageDimensions {
  width: number;
  height: number;
}

function getLandmarkSide(index: number): "left" | "right" | "center" {
  if (LEFT_LANDMARKS.has(index)) return "left";
  if (RIGHT_LANDMARKS.has(index)) return "right";
  return "center";
}

function getConnectionRegion(a: number, b: number): BodyRegion {
  const min = Math.min(a, b);
  const max = Math.max(a, b);
  if (TORSO_EDGE_KEYS.has(`${min}-${max}`)) return "torso";
  const sideA = getLandmarkSide(a);
  const sideB = getLandmarkSide(b);
  if (sideA === "left" && sideB === "left") return "left";
  if (sideA === "right" && sideB === "right") return "right";
  if (
    (sideA === "center" && sideB === "left") ||
    (sideB === "center" && sideA === "left")
  ) {
    return "left";
  }
  if (
    (sideA === "center" && sideB === "right") ||
    (sideB === "center" && sideA === "right")
  ) {
    return "right";
  }
  return "midline";
}

function regionColor(region: BodyRegion, personId: PersonId = 0): string {
  if (personId === 1) {
    switch (region) {
      case "left":
        return P2_COLOR_LEFT;
      case "right":
        return P2_COLOR_RIGHT;
      case "torso":
        return P2_COLOR_TORSO;
      case "midline":
        return P2_COLOR_MIDLINE;
    }
  }
  switch (region) {
    case "left":
      return COLOR_LEFT;
    case "right":
      return COLOR_RIGHT;
    case "torso":
      return COLOR_TORSO;
    case "midline":
      return COLOR_MIDLINE;
  }
}

function landmarkColor(index: number, personId: PersonId = 0): string {
  const side = getLandmarkSide(index);
  if (personId === 1) {
    if (side === "left") return P2_COLOR_LEFT;
    if (side === "right") return P2_COLOR_RIGHT;
    return P2_COLOR_TORSO;
  }
  if (side === "left") return COLOR_LEFT;
  if (side === "right") return COLOR_RIGHT;
  return COLOR_TORSO;
}

function getVisibility(lm: Point3D): number {
  if (lm.visibility === undefined) return 1;
  if (!Number.isFinite(lm.visibility)) return 0;
  return Math.max(0, Math.min(1, lm.visibility));
}

function visibilityAlpha(vis: number, dimThreshold: number): number {
  const threshold = Math.max(0.05, Math.min(0.95, dimThreshold));
  if (vis < threshold) {
    return 0.12 + (vis / threshold) * 0.23;
  }
  return 0.45 + ((vis - threshold) / (1 - threshold)) * 0.55;
}

function computeDepthFactor(z: number): number {
  const Z_RANGE = 0.6;
  const safeZ = Number.isFinite(z) ? z : 0;
  const clamped = Math.max(-Z_RANGE, Math.min(Z_RANGE, safeZ));
  return (-clamped + Z_RANGE) / (2 * Z_RANGE);
}

function isValidPoint2D(p: Point3D | undefined | null): p is Point3D {
  return Boolean(p) && Number.isFinite(p!.x) && Number.isFinite(p!.y);
}

function drawBlackoutBackdrop(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
): void {
  ctx.save();
  ctx.shadowBlur = 0;
  ctx.globalAlpha = 1;
  ctx.fillStyle = BLACKOUT_BG;
  ctx.fillRect(0, 0, width, height);

  const minorStep = 24;
  const majorStep = 96;

  // Minor grid
  ctx.strokeStyle = "rgba(34, 211, 238, 0.06)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = minorStep; x < width; x += minorStep) {
    if (x % majorStep === 0) continue;
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
  }
  for (let y = minorStep; y < height; y += minorStep) {
    if (y % majorStep === 0) continue;
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
  }
  ctx.stroke();

  // Major grid
  ctx.strokeStyle = "rgba(34, 211, 238, 0.14)";
  ctx.beginPath();
  for (let x = majorStep; x < width; x += majorStep) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
  }
  for (let y = majorStep; y < height; y += majorStep) {
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
  }
  ctx.stroke();

  // Center reticle
  const cx = width / 2;
  const cy = height / 2;
  ctx.strokeStyle = "rgba(34, 211, 238, 0.26)";
  ctx.beginPath();
  ctx.moveTo(cx - 18, cy);
  ctx.lineTo(cx - 6, cy);
  ctx.moveTo(cx + 6, cy);
  ctx.lineTo(cx + 18, cy);
  ctx.moveTo(cx, cy - 18);
  ctx.lineTo(cx, cy - 6);
  ctx.moveTo(cx, cy + 6);
  ctx.lineTo(cx, cy + 18);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(cx, cy, 28, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawConnectionList(
  ctx: CanvasRenderingContext2D,
  landmarks: readonly Point3D[],
  connections: readonly { start: number; end: number }[],
  projectX: (nx: number) => number,
  projectY: (ny: number) => number,
  strokeStyle: string,
  lineWidth: number,
  alpha: number,
  shadowBlur = 0,
): void {
  ctx.save();
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidth;
  ctx.globalAlpha = alpha;
  ctx.shadowBlur = shadowBlur;
  if (shadowBlur > 0) {
    ctx.shadowColor = strokeStyle;
  }
  ctx.beginPath();
  for (let i = 0; i < connections.length; i++) {
    const conn = connections[i];
    if (!conn) continue;
    const a = landmarks[conn.start];
    const b = landmarks[conn.end];
    if (!isValidPoint2D(a) || !isValidPoint2D(b)) continue;
    ctx.moveTo(projectX(a.x), projectY(a.y));
    ctx.lineTo(projectX(b.x), projectY(b.y));
  }
  ctx.stroke();
  ctx.restore();
}

function drawPointChain(
  ctx: CanvasRenderingContext2D,
  landmarks: readonly Point3D[],
  chain: readonly number[],
  projectX: (nx: number) => number,
  projectY: (ny: number) => number,
  strokeStyle: string,
  lineWidth: number,
  alpha: number,
  shadowBlur = 0,
): void {
  ctx.save();
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidth;
  ctx.globalAlpha = alpha;
  ctx.shadowBlur = shadowBlur;
  if (shadowBlur > 0) {
    ctx.shadowColor = strokeStyle;
  }
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < chain.length; i++) {
    const pt = landmarks[chain[i]!];
    if (!isValidPoint2D(pt)) continue;
    const px = projectX(pt.x);
    const py = projectY(pt.y);
    if (!started) {
      ctx.moveTo(px, py);
      started = true;
    } else {
      ctx.lineTo(px, py);
    }
  }
  if (started) {
    ctx.stroke();
  }
  ctx.restore();
}

function drawIrisAndPupil(
  ctx: CanvasRenderingContext2D,
  faceLandmarks: readonly Point3D[],
  centerIndex: number,
  ringIndices: readonly [number, number, number, number],
  projectX: (nx: number) => number,
  projectY: (ny: number) => number,
  color: string,
  isBlackout: boolean,
): void {
  const centerPt = faceLandmarks[centerIndex];
  if (!isValidPoint2D(centerPt)) return;

  const cx = projectX(centerPt.x);
  const cy = projectY(centerPt.y);

  let radiusSum = 0;
  let validCount = 0;
  for (const idx of ringIndices) {
    const rPt = faceLandmarks[idx];
    if (isValidPoint2D(rPt)) {
      const rx = projectX(rPt.x);
      const ry = projectY(rPt.y);
      radiusSum += Math.hypot(rx - cx, ry - cy);
      validCount += 1;
    }
  }
  const irisRadius = Math.max(
    2.2,
    validCount > 0 ? radiusSum / validCount : 4.0,
  );

  ctx.save();
  // Outer glowing iris ring
  ctx.strokeStyle = color;
  ctx.fillStyle = isBlackout
    ? "rgba(34, 211, 238, 0.18)"
    : "rgba(34, 211, 238, 0.12)";
  ctx.lineWidth = 1.6;
  ctx.shadowColor = color;
  ctx.shadowBlur = isBlackout ? 10 : 6;
  ctx.globalAlpha = 0.95;
  ctx.beginPath();
  ctx.arc(cx, cy, irisRadius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Inner bright pupil core
  ctx.shadowBlur = 0;
  ctx.fillStyle = "#f8fafc";
  ctx.globalAlpha = 0.98;
  ctx.beginPath();
  ctx.arc(cx, cy, Math.max(1.2, irisRadius * 0.36), 0, Math.PI * 2);
  ctx.fill();

  // Subtle iris gaze crosshair in Blackout mode
  if (isBlackout) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 0.8;
    ctx.globalAlpha = 0.55;
    const tick = irisRadius + 2.5;
    ctx.beginPath();
    ctx.moveTo(cx - tick, cy);
    ctx.lineTo(cx + tick, cy);
    ctx.moveTo(cx, cy - tick);
    ctx.lineTo(cx, cy + tick);
    ctx.stroke();
  }
  ctx.restore();
}

/**
 * Draws the 478-point FaceLandmarker mesh, 3D cranial dome/head contour,
 * eyebrows, nose ridge, lips, eyelids, and gaze-tracking irises.
 */
function drawFaceAndHeadLayer(
  ctx: CanvasRenderingContext2D,
  faceLandmarks: readonly Point3D[],
  poseLandmarks: readonly Point3D[] | null,
  projectX: (nx: number) => number,
  projectY: (ny: number) => number,
  isBlackout: boolean,
  showFaceMesh: boolean,
  personId: PersonId = 0,
): void {
  if (faceLandmarks.length < 468) return;

  const leftColor = personId === 1 ? P2_COLOR_LEFT : COLOR_LEFT;
  const rightColor = personId === 1 ? P2_COLOR_RIGHT : COLOR_RIGHT;
  const contourColor = personId === 1 ? "#34d399" : "#38bdf8";
  const torsoColor = personId === 1 ? P2_COLOR_TORSO : COLOR_TORSO;

  // 0. Neck spine bridge from Pose shoulders midpoint (11 & 12) to Chin (152)
  const chin = faceLandmarks[152];
  const leftShoulder = poseLandmarks?.[11];
  const rightShoulder = poseLandmarks?.[12];
  if (
    isValidPoint2D(chin) &&
    isValidPoint2D(leftShoulder) &&
    isValidPoint2D(rightShoulder)
  ) {
    const neckX = (projectX(leftShoulder.x) + projectX(rightShoulder.x)) * 0.5;
    const neckY = (projectY(leftShoulder.y) + projectY(rightShoulder.y)) * 0.5;
    const chinX = projectX(chin.x);
    const chinY = projectY(chin.y);

    ctx.save();
    ctx.strokeStyle = torsoColor;
    ctx.shadowColor = leftColor;
    ctx.shadowBlur = isBlackout ? 10 : 6;
    ctx.lineWidth = isBlackout ? 2.6 : 2.2;
    ctx.globalAlpha = 0.78;
    if (typeof ctx.setLineDash === "function") {
      ctx.setLineDash([4, 3]);
    }
    ctx.beginPath();
    ctx.moveTo(neckX, neckY);
    ctx.lineTo(chinX, chinY);
    ctx.stroke();
    ctx.restore();
  }

  // 1. Optional 3D Face Tesselation Wireframe
  if (showFaceMesh && FaceLandmarker.FACE_LANDMARKS_TESSELATION) {
    drawConnectionList(
      ctx,
      faceLandmarks,
      FaceLandmarker.FACE_LANDMARKS_TESSELATION,
      projectX,
      projectY,
      contourColor,
      0.55,
      isBlackout ? 0.19 : 0.12,
      0,
    );
  }

  // 2. 3D Head Shape: Face Oval + Synthesized Cranial Crown Dome & Axis Meridians
  if (FaceLandmarker.FACE_LANDMARKS_FACE_OVAL) {
    drawConnectionList(
      ctx,
      faceLandmarks,
      FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
      projectX,
      projectY,
      contourColor,
      isBlackout ? 2.1 : 1.7,
      0.88,
      isBlackout ? 10 : 5,
    );
  }

  // Cranial dome & 3D head-axis guide above forehead
  const forehead = faceLandmarks[10];
  const leftTemple = faceLandmarks[234];
  const rightTemple = faceLandmarks[454];
  const noseBridgeTop = faceLandmarks[168];
  if (
    isValidPoint2D(forehead) &&
    isValidPoint2D(chin) &&
    isValidPoint2D(leftTemple) &&
    isValidPoint2D(rightTemple)
  ) {
    const fx = projectX(forehead.x);
    const fy = projectY(forehead.y);
    const cx = projectX(chin.x);
    const cy = projectY(chin.y);
    const lx = projectX(leftTemple.x);
    const ly = projectY(leftTemple.y);
    const rx = projectX(rightTemple.x);
    const ry = projectY(rightTemple.y);

    // Vector from chin to forehead defines head vertical axis
    const vx = fx - cx;
    const vy = fy - cy;
    // Crown apex sits ~26% above the top forehead landmark along the cranial axis
    const crownX = fx + vx * 0.26;
    const crownY = fy + vy * 0.26;

    ctx.save();
    ctx.strokeStyle = contourColor;
    ctx.shadowColor = contourColor;
    ctx.shadowBlur = isBlackout ? 10 : 5;
    ctx.lineWidth = isBlackout ? 1.8 : 1.4;
    ctx.globalAlpha = 0.72;

    // Outer cranial dome curve from left temple -> crown apex -> right temple
    ctx.beginPath();
    ctx.moveTo(lx, ly);
    if (typeof ctx.quadraticCurveTo === "function") {
      ctx.quadraticCurveTo(
        lx + vx * 0.38,
        ly + vy * 0.38,
        crownX,
        crownY,
      );
      ctx.quadraticCurveTo(
        rx + vx * 0.38,
        ry + vy * 0.38,
        rx,
        ry,
      );
    } else {
      ctx.lineTo(crownX, crownY);
      ctx.lineTo(rx, ry);
    }
    ctx.stroke();

    // Central sagittal head meridian (crown apex -> forehead -> nose bridge)
    ctx.lineWidth = 1.0;
    ctx.globalAlpha = 0.45;
    if (typeof ctx.setLineDash === "function") {
      ctx.setLineDash([3, 3]);
    }
    ctx.beginPath();
    ctx.moveTo(crownX, crownY);
    ctx.lineTo(fx, fy);
    if (isValidPoint2D(noseBridgeTop)) {
      ctx.lineTo(projectX(noseBridgeTop.x), projectY(noseBridgeTop.y));
    }
    ctx.stroke();
    ctx.restore();
  }

  // 3. Nose Bridge & Nostril Base
  drawPointChain(
    ctx,
    faceLandmarks,
    NOSE_BRIDGE_CHAIN,
    projectX,
    projectY,
    "#e2e8f0",
    1.5,
    0.82,
    isBlackout ? 6 : 3,
  );
  drawPointChain(
    ctx,
    faceLandmarks,
    NOSE_BASE_CHAIN,
    projectX,
    projectY,
    "#cbd5e1",
    1.2,
    0.7,
    0,
  );

  // 4. Eyebrows
  if (FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW) {
    drawConnectionList(
      ctx,
      faceLandmarks,
      FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
      projectX,
      projectY,
      leftColor,
      1.8,
      0.88,
      isBlackout ? 8 : 4,
    );
  }
  if (FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW) {
    drawConnectionList(
      ctx,
      faceLandmarks,
      FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
      projectX,
      projectY,
      rightColor,
      1.8,
      0.88,
      isBlackout ? 8 : 4,
    );
  }

  // 5. Lips (Inner & Outer Mouth)
  if (FaceLandmarker.FACE_LANDMARKS_LIPS) {
    drawConnectionList(
      ctx,
      faceLandmarks,
      FaceLandmarker.FACE_LANDMARKS_LIPS,
      projectX,
      projectY,
      "#fb7185",
      1.5,
      0.9,
      isBlackout ? 8 : 4,
    );
  }

  // 6. Upper & Lower Eyelid Contours
  if (FaceLandmarker.FACE_LANDMARKS_LEFT_EYE) {
    drawConnectionList(
      ctx,
      faceLandmarks,
      FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
      projectX,
      projectY,
      leftColor,
      1.7,
      0.95,
      isBlackout ? 8 : 4,
    );
  }
  if (FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE) {
    drawConnectionList(
      ctx,
      faceLandmarks,
      FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
      projectX,
      projectY,
      rightColor,
      1.7,
      0.95,
      isBlackout ? 8 : 4,
    );
  }

  // 7. Gaze-Tracking Irises & Pupils (landmarks 468..477)
  if (faceLandmarks.length >= 478) {
    // Right iris: center 468, perimeter 469..472
    drawIrisAndPupil(
      ctx,
      faceLandmarks,
      468,
      [469, 470, 471, 472],
      projectX,
      projectY,
      rightColor,
      isBlackout,
    );
    // Left iris: center 473, perimeter 474..477
    drawIrisAndPupil(
      ctx,
      faceLandmarks,
      473,
      [474, 475, 476, 477],
      projectX,
      projectY,
      leftColor,
      isBlackout,
    );
  }
}

/**
 * Draws 21-landmark articulated hand & 5-finger skeletons for up to 2 hands,
 * fingertip target halos, pinch beams, and upright hand telemetry tags.
 */
function drawHandsLayer(
  ctx: CanvasRenderingContext2D,
  hands: readonly TrackedHand[],
  poseLandmarks: readonly Point3D[] | null,
  projectX: (nx: number) => number,
  projectY: (ny: number) => number,
  width: number,
  height: number,
  isBlackout: boolean,
  personId: PersonId = 0,
  showPersonPrefix = false,
): void {
  if (!hands || hands.length === 0) return;

  const handConnections = HandLandmarker.HAND_CONNECTIONS ?? [];

  for (const hand of hands) {
    const lms = hand.landmarks;
    if (!lms || lms.length < 21) continue;

    const isLeft = hand.handedness === "Left";
    const primaryColor =
      personId === 1
        ? isLeft
          ? P2_COLOR_LEFT
          : P2_COLOR_RIGHT
        : isLeft
          ? COLOR_LEFT
          : COLOR_RIGHT;
    const secondaryColor =
      personId === 1
        ? isLeft
          ? "#6ee7b7"
          : "#fcd34d"
        : isLeft
          ? "#67e8f9"
          : "#fb923c";
    const tipColor = isLeft ? "#a5f3fc" : "#fde047";

    // Bridge from Pose wrist (15 for Left, 16 for Right) to Hand wrist (0)
    const poseWrist = poseLandmarks?.[isLeft ? 15 : 16];
    const handWrist = lms[0];
    if (isValidPoint2D(poseWrist) && isValidPoint2D(handWrist)) {
      ctx.save();
      ctx.strokeStyle = primaryColor;
      ctx.lineWidth = 2.2;
      ctx.globalAlpha = 0.75;
      ctx.beginPath();
      ctx.moveTo(projectX(poseWrist.x), projectY(poseWrist.y));
      ctx.lineTo(projectX(handWrist.x), projectY(handWrist.y));
      ctx.stroke();
      ctx.restore();
    }

    // Draw all 21 palm & phalanx bone connections
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const conn of handConnections) {
      const a = lms[conn.start];
      const b = lms[conn.end];
      if (!isValidPoint2D(a) || !isValidPoint2D(b)) continue;

      const isFingerTipSegment = FINGERTIP_INDICES.has(conn.end);
      const avgDepth =
        (computeDepthFactor(a.z) + computeDepthFactor(b.z)) * 0.5;
      ctx.strokeStyle = isFingerTipSegment ? secondaryColor : primaryColor;
      ctx.shadowColor = primaryColor;
      ctx.shadowBlur = isBlackout ? 4 + avgDepth * 12 : 6;
      ctx.lineWidth = isBlackout
        ? (isFingerTipSegment ? 1.8 : 2.3) + avgDepth * 2.0
        : isFingerTipSegment
          ? 2.2
          : 2.8;
      ctx.globalAlpha = 0.92;

      ctx.beginPath();
      ctx.moveTo(projectX(a.x), projectY(a.y));
      ctx.lineTo(projectX(b.x), projectY(b.y));
      ctx.stroke();
    }
    ctx.restore();

    // Draw 21 hand joints + glowing fingertip target rings
    ctx.save();
    for (let i = 0; i < 21; i++) {
      const pt = lms[i];
      if (!isValidPoint2D(pt)) continue;

      const px = projectX(pt.x);
      const py = projectY(pt.y);
      const isTip = FINGERTIP_INDICES.has(i);
      const isWrist = i === 0;
      const depth = computeDepthFactor(pt.z);
      const radius = isTip
        ? isBlackout
          ? 3.2 + depth * 2.5
          : 3.8
        : isWrist
          ? 4.2
          : 2.7;

      ctx.fillStyle = isTip ? tipColor : primaryColor;
      ctx.shadowColor = isTip ? tipColor : primaryColor;
      ctx.shadowBlur = isTip ? 10 : 5;
      ctx.globalAlpha = 0.95;
      ctx.beginPath();
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      ctx.fill();

      if (isTip) {
        ctx.shadowBlur = 0;
        ctx.strokeStyle = tipColor;
        ctx.lineWidth = 1.1;
        ctx.globalAlpha = 0.75;
        ctx.beginPath();
        ctx.arc(px, py, radius + 2.6, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    ctx.restore();

    // Pinch beam between Thumb Tip (4) and Index Tip (8) + Upright Wrist Tag
    const metrics = computeHandMetrics(hand);
    const thumbTip = lms[4];
    const indexTip = lms[8];
    if (
      metrics.isPinching &&
      isValidPoint2D(thumbTip) &&
      isValidPoint2D(indexTip)
    ) {
      const tx = projectX(thumbTip.x);
      const ty = projectY(thumbTip.y);
      const ix = projectX(indexTip.x);
      const iy = projectY(indexTip.y);
      const mx = (tx + ix) * 0.5;
      const my = (ty + iy) * 0.5;

      ctx.save();
      ctx.strokeStyle = "#fde047";
      ctx.shadowColor = "#fde047";
      ctx.shadowBlur = 12;
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(ix, iy);
      ctx.stroke();

      ctx.fillStyle = "#fde047";
      ctx.beginPath();
      ctx.arc(mx, my, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Upright Hand HUD Tag near wrist
    if (isValidPoint2D(handWrist)) {
      const wx = projectX(handWrist.x);
      const wy = projectY(handWrist.y);
      const prefix = showPersonPrefix ? `P${personId + 1}•` : "";
      const tagText = metrics.isPinching
        ? `${prefix}${isLeft ? "L" : "R"}.HAND • PINCH`
        : `${prefix}${isLeft ? "L" : "R"}.HAND • ${metrics.extendedFingers}F`;

      ctx.save();
      ctx.shadowBlur = 0;
      ctx.globalAlpha = 0.92;
      ctx.font =
        "600 10px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      const pillW = tagText.length * 6.4 + 12;
      const pillH = 18;
      const boxX = Math.max(
        4,
        Math.min(width - pillW - 4, wx - pillW * 0.5),
      );
      const boxY = Math.max(4, Math.min(height - pillH - 4, wy + 12));

      ctx.fillStyle = "rgba(5, 8, 15, 0.88)";
      ctx.strokeStyle = primaryColor;
      ctx.lineWidth = 1;
      if (typeof ctx.roundRect === "function") {
        ctx.beginPath();
        ctx.roundRect(boxX, boxY, pillW, pillH, 4);
        ctx.fill();
        ctx.stroke();
      } else {
        ctx.fillRect(boxX, boxY, pillW, pillH);
        ctx.strokeRect(boxX, boxY, pillW, pillH);
      }

      ctx.fillStyle = primaryColor;
      ctx.fillText(tagText, boxX + pillW * 0.5, boxY + pillH * 0.5 + 0.5);
      ctx.restore();
    }
  }
}

function drawPersonBeacon(
  ctx: CanvasRenderingContext2D,
  personId: PersonId,
  landmarks: readonly Point3D[],
  faceLandmarks: readonly Point3D[] | null,
  projectX: (nx: number) => number,
  projectY: (ny: number) => number,
  width: number,
  height: number,
): void {
  const forehead = faceLandmarks?.[10];
  const nose = landmarks[0];
  const lSh = landmarks[11];
  const rSh = landmarks[12];

  let anchorX: number | null = null;
  let anchorY: number | null = null;

  if (isValidPoint2D(forehead)) {
    anchorX = projectX(forehead.x);
    anchorY = projectY(forehead.y) - 26;
  } else if (isValidPoint2D(nose)) {
    anchorX = projectX(nose.x);
    anchorY = projectY(nose.y) - 34;
  } else if (isValidPoint2D(lSh) && isValidPoint2D(rSh)) {
    anchorX = (projectX(lSh.x) + projectX(rSh.x)) * 0.5;
    anchorY = Math.min(projectY(lSh.y), projectY(rSh.y)) - 32;
  }

  if (anchorX === null || anchorY === null) return;

  const label = personId === 0 ? "P1 • SUBJECT 01" : "P2 • SUBJECT 02";
  const accent = personId === 0 ? COLOR_LEFT : P2_COLOR_LEFT;

  ctx.save();
  ctx.shadowBlur = 8;
  ctx.shadowColor = accent;
  ctx.globalAlpha = 0.94;
  ctx.font =
    "700 10px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  const pillW = label.length * 6.5 + 14;
  const pillH = 19;
  const boxX = Math.max(4, Math.min(width - pillW - 4, anchorX - pillW * 0.5));
  const boxY = Math.max(4, Math.min(height - pillH - 4, anchorY - pillH));

  ctx.fillStyle = "rgba(5, 8, 15, 0.9)";
  ctx.strokeStyle = accent;
  ctx.lineWidth = 1.2;
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath();
    ctx.roundRect(boxX, boxY, pillW, pillH, 4);
    ctx.fill();
    ctx.stroke();
  } else {
    ctx.fillRect(boxX, boxY, pillW, pillH);
    ctx.strokeRect(boxX, boxY, pillW, pillH);
  }

  ctx.shadowBlur = 0;
  ctx.fillStyle = accent;
  ctx.fillText(label, boxX + pillW * 0.5, boxY + pillH * 0.5 + 0.5);
  ctx.restore();
}

export function drawSkeletonFrame(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  params: DrawSkeletonFrameParams,
): void {
  if (
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    width <= 0 ||
    height <= 0
  ) {
    return;
  }

  const {
    landmarks,
    hands = [],
    faceLandmarks = null,
    persons,
    showFaceMesh = true,
    viewMode,
    mirrored,
    angles = {},
    selectedJoints = [],
    minVisibility = 0.4,
  } = params;
  const dimThreshold = Number.isFinite(minVisibility) ? minVisibility : 0.4;
  const isBlackout = viewMode === "blackout";

  if (isBlackout) {
    drawBlackoutBackdrop(ctx, width, height);
  } else {
    ctx.clearRect(0, 0, width, height);
  }

  const projectX = (nx: number): number => (mirrored ? 1 - nx : nx) * width;
  const projectY = (ny: number): number => ny * height;

  // Build unified list of subjects to render (1 or 2 people)
  const subjects: Array<{
    personId: PersonId;
    landmarks: readonly Point3D[] | null;
    angles: Partial<Record<JointId, number | null>>;
    hands: readonly TrackedHand[];
    faceLandmarks: readonly Point3D[] | null;
  }> =
    persons && persons.length > 0
      ? persons.map((p) => ({
          personId: p.personId,
          landmarks: p.landmarks,
          angles: p.angles,
          hands: p.hands,
          faceLandmarks: p.faceLandmarks,
        }))
      : [
          {
            personId: 0,
            landmarks,
            angles,
            hands,
            faceLandmarks,
          },
        ];

  const isMultiPersonActive = subjects.length > 1;

  for (const subject of subjects) {
    const pId = subject.personId;
    const pLandmarks = subject.landmarks;
    const pHands = subject.hands;
    const pFace = subject.faceLandmarks;
    const pAngles = subject.angles;

    const hasDetailedFace = Boolean(pFace && pFace.length >= 468);
    const hasDetailedLeftHand = pHands.some(
      (h) => h.handedness === "Left" && h.landmarks.length >= 21,
    );
    const hasDetailedRightHand = pHands.some(
      (h) => h.handedness === "Right" && h.landmarks.length >= 21,
    );

    const shouldSkipPoseLandmark = (index: number): boolean => {
      if (hasDetailedFace && POSE_FACE_INDICES.has(index)) {
        return true;
      }
      if (hasDetailedLeftHand && POSE_LEFT_HAND_INDICES.has(index)) {
        return true;
      }
      if (hasDetailedRightHand && POSE_RIGHT_HAND_INDICES.has(index)) {
        return true;
      }
      return false;
    };

    if (pLandmarks && pLandmarks.length > 0) {
      const edges = POSE_CONNECTIONS.filter(
        ([start, end]) =>
          !shouldSkipPoseLandmark(start) && !shouldSkipPoseLandmark(end),
      ).map(([start, end]) => {
        const a = pLandmarks[start];
        const b = pLandmarks[end];
        const avgZ =
          a && b && Number.isFinite(a.z) && Number.isFinite(b.z)
            ? (a.z + b.z) / 2
            : 0;
        return { start, end, avgZ };
      });

      if (isBlackout) {
        edges.sort((e1, e2) => e2.avgZ - e1.avgZ);
      }

      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      for (const { start, end } of edges) {
        const a = pLandmarks[start];
        const b = pLandmarks[end];
        if (
          !a ||
          !b ||
          !Number.isFinite(a.x) ||
          !Number.isFinite(a.y) ||
          !Number.isFinite(b.x) ||
          !Number.isFinite(b.y)
        ) {
          continue;
        }

        const visA = getVisibility(a);
        const visB = getVisibility(b);
        const minVis = Math.min(visA, visB);
        if (minVis <= 0.02) continue;

        const color = regionColor(getConnectionRegion(start, end), pId);
        const visAlpha = visibilityAlpha(minVis, dimThreshold);

        if (isBlackout) {
          const avgDepth =
            (computeDepthFactor(a.z) + computeDepthFactor(b.z)) / 2;
          ctx.lineWidth = 1.75 + avgDepth * 4.25;
          ctx.globalAlpha = (0.55 + 0.45 * avgDepth) * visAlpha;
          ctx.shadowBlur = 3 + avgDepth * 16;
        } else {
          ctx.lineWidth = 3;
          ctx.globalAlpha = 0.8 * visAlpha;
          ctx.shadowBlur = 8;
        }

        ctx.strokeStyle = color;
        ctx.shadowColor = color;
        ctx.beginPath();
        ctx.moveTo(projectX(a.x), projectY(a.y));
        ctx.lineTo(projectX(b.x), projectY(b.y));
        ctx.stroke();
      }

      const indices = pLandmarks
        .map((_, idx) => idx)
        .filter((idx) => !shouldSkipPoseLandmark(idx));
      if (isBlackout) {
        indices.sort((i1, i2) => {
          const z1 = Number.isFinite(pLandmarks[i1]?.z)
            ? pLandmarks[i1]!.z
            : 0;
          const z2 = Number.isFinite(pLandmarks[i2]?.z)
            ? pLandmarks[i2]!.z
            : 0;
          return z2 - z1;
        });
      }

      for (const idx of indices) {
        const lm = pLandmarks[idx];
        if (!lm || !Number.isFinite(lm.x) || !Number.isFinite(lm.y)) continue;
        const vis = getVisibility(lm);
        if (vis <= 0.02) continue;

        const px = projectX(lm.x);
        const py = projectY(lm.y);
        const color = landmarkColor(idx, pId);
        const visAlpha = visibilityAlpha(vis, dimThreshold);
        const depth = computeDepthFactor(lm.z);
        const radius = isBlackout ? 3.0 + depth * 6.0 : 4.5;

        ctx.fillStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = isBlackout ? 3 + depth * 16 : 10;
        ctx.globalAlpha = isBlackout
          ? (0.55 + 0.45 * depth) * visAlpha
          : 0.9 * visAlpha;

        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fill();

        if (vis >= dimThreshold) {
          ctx.shadowBlur = 0;
          ctx.fillStyle = "#f8fafc";
          ctx.globalAlpha = isBlackout
            ? (0.65 + 0.35 * depth) * visAlpha
            : 0.9 * visAlpha;
          ctx.beginPath();
          ctx.arc(px, py, Math.max(1.4, radius * 0.38), 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.restore();
    }

    // V2 Layer: Detailed Face, Eyes, Irises & 3D Head Shape
    if (pFace && pFace.length >= 468) {
      drawFaceAndHeadLayer(
        ctx,
        pFace,
        pLandmarks,
        projectX,
        projectY,
        isBlackout,
        showFaceMesh,
        pId,
      );
    }

    // V2 Layer: Detailed Hands & 5-Finger Articulation
    if (pHands && pHands.length > 0) {
      drawHandsLayer(
        ctx,
        pHands,
        pLandmarks,
        projectX,
        projectY,
        width,
        height,
        isBlackout,
        pId,
        isMultiPersonActive,
      );
    }

    // Multi-person Subject Beacon (P1 / P2) when 2 people are detected
    if (isMultiPersonActive && pLandmarks && pLandmarks.length > 0) {
      drawPersonBeacon(
        ctx,
        pId,
        pLandmarks,
        pFace,
        projectX,
        projectY,
        width,
        height,
      );
    }

    // Joint angle pills (upright, unflipped, clamped within canvas bounds)
    if (pLandmarks && pLandmarks.length > 0 && selectedJoints.length > 0) {
      ctx.save();
      ctx.shadowBlur = 0;
      ctx.globalAlpha = 1;
      ctx.font =
        "600 11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      const margin = 4;
      const padX = 6;
      const pillH = 20;

      for (const jointId of selectedJoints) {
        const def = JOINT_DEFINITION_MAP[jointId];
        if (!def) continue;
        const deg = pAngles[jointId];
        if (deg === undefined || deg === null || !Number.isFinite(deg))
          continue;

        const vertex = pLandmarks[def.indices[1]];
        if (!vertex || !Number.isFinite(vertex.x) || !Number.isFinite(vertex.y)) {
          continue;
        }

        const vx = projectX(vertex.x);
        const vy = projectY(vertex.y);
        const rounded = Math.round(deg);
        const label = `${rounded === 0 ? 0 : rounded}°`;
        const measured =
          typeof ctx.measureText === "function"
            ? ctx.measureText(label).width
            : label.length * 7;
        const textW =
          Number.isFinite(measured) && measured > 0
            ? measured
            : label.length * 7;
        const pillW = Math.max(38, Math.ceil(textW) + padX * 2);

        const rawX =
          vx + 10 + pillW + margin <= width ? vx + 10 : vx - 10 - pillW;
        const rawY = vy - pillH - 8 >= margin ? vy - pillH - 8 : vy + 10;
        const boxX =
          width >= pillW + margin * 2
            ? Math.max(margin, Math.min(width - pillW - margin, rawX))
            : Math.max(0, (width - pillW) / 2);
        const boxY =
          height >= pillH + margin * 2
            ? Math.max(margin, Math.min(height - pillH - margin, rawY))
            : Math.max(0, (height - pillH) / 2);

        const accent =
          pId === 1
            ? def.side === "left"
              ? P2_COLOR_LEFT
              : P2_COLOR_RIGHT
            : def.side === "left"
              ? COLOR_LEFT
              : COLOR_RIGHT;
        ctx.fillStyle = "rgba(5, 8, 15, 0.86)";
        ctx.strokeStyle = accent;
        ctx.lineWidth = 1;

        if (typeof ctx.roundRect === "function") {
          ctx.beginPath();
          ctx.roundRect(boxX, boxY, pillW, pillH, 5);
          ctx.fill();
          ctx.stroke();
        } else {
          ctx.fillRect(boxX, boxY, pillW, pillH);
          ctx.strokeRect(boxX, boxY, pillW, pillH);
        }

        ctx.fillStyle = "#f8fafc";
        ctx.fillText(label, boxX + pillW / 2, boxY + pillH / 2 + 0.5);
      }
      ctx.restore();
    }
  }
}

function computeContainBox(
  outerSize: StageDimensions | null,
  aspectRatio: number,
): StageDimensions | null {
  if (
    !outerSize ||
    outerSize.width <= 0 ||
    outerSize.height <= 0 ||
    !Number.isFinite(aspectRatio) ||
    aspectRatio <= 0
  ) {
    return null;
  }

  if (outerSize.width / outerSize.height > aspectRatio) {
    const height = Math.floor(outerSize.height);
    const width = Math.floor(outerSize.height * aspectRatio);
    return width > 0 && height > 0 ? { width, height } : null;
  }

  const width = Math.floor(outerSize.width);
  const height = Math.floor(outerSize.width / aspectRatio);
  return width > 0 && height > 0 ? { width, height } : null;
}

interface StatusDisplayMeta {
  title: string;
  defaultDetail: string;
  isError: boolean;
  showRetry: boolean;
}

function getStatusMeta(status: AppStatus): StatusDisplayMeta {
  switch (status) {
    case "requesting-camera":
      return {
        title: "Requesting Camera Access",
        defaultDetail:
          "Waiting for browser camera permission to start live pose, hand & face tracking.",
        isError: false,
        showRetry: false,
      };
    case "loading-model":
      return {
        title: "Initializing Pose, Hand & Face Models",
        defaultDetail:
          "Loading MediaPipe Pose, Hand (21x2) & Face (478) Landmarker weights and WASM runtime...",
        isError: false,
        showRetry: false,
      };
    case "permission-denied":
      return {
        title: "Camera Permission Denied",
        defaultDetail:
          "Camera access was blocked. Enable camera permission in browser settings and retry.",
        isError: true,
        showRetry: true,
      };
    case "no-camera":
      return {
        title: "No Camera Detected",
        defaultDetail:
          "No video input device was found. Connect a camera and try again.",
        isError: true,
        showRetry: true,
      };
    case "error":
      return {
        title: "Pose Pipeline Error",
        defaultDetail:
          "Unable to initialize camera stream or pose inference engine.",
        isError: true,
        showRetry: true,
      };
    case "ready":
      return {
        title: "",
        defaultDetail: "",
        isError: false,
        showRetry: false,
      };
  }
}

function renderStatusIcon(status: AppStatus): ReactElement | null {
  switch (status) {
    case "requesting-camera":
      return <Camera className="h-7 w-7 text-cyan-300" aria-hidden="true" />;
    case "loading-model":
      return (
        <Loader2
          className="h-7 w-7 animate-spin text-cyan-300"
          aria-hidden="true"
        />
      );
    case "permission-denied":
      return (
        <ShieldAlert className="h-7 w-7 text-rose-400" aria-hidden="true" />
      );
    case "no-camera":
      return (
        <CameraOff className="h-7 w-7 text-amber-400" aria-hidden="true" />
      );
    case "error":
      return (
        <AlertTriangle className="h-7 w-7 text-rose-400" aria-hidden="true" />
      );
    case "ready":
      return null;
  }
}

export function SkeletonCanvas({
  videoRef,
  canvasRef,
  landmarks,
  hands = [],
  faceLandmarks = null,
  persons,
  showFaceMesh = true,
  angles,
  selectedJoints,
  viewMode,
  mirrored,
  status,
  statusMessage,
  onRetry,
  videoAspectRatio,
}: SkeletonCanvasProps): ReactElement {
  const outerRef = useRef<HTMLDivElement | null>(null);
  const latestDrawParamsRef = useRef<DrawSkeletonFrameParams>({
    landmarks,
    hands,
    faceLandmarks,
    persons,
    showFaceMesh,
    angles,
    selectedJoints,
    viewMode,
    mirrored,
  });

  const [outerSize, setOuterSize] = useState<StageDimensions | null>(null);
  const [detectedAspectRatio, setDetectedAspectRatio] = useState<number | null>(
    null,
  );

  const normalizedPropAspect =
    videoAspectRatio !== undefined &&
    Number.isFinite(videoAspectRatio) &&
    videoAspectRatio > 0
      ? videoAspectRatio
      : null;

  const effectiveAspectRatio =
    detectedAspectRatio ?? normalizedPropAspect ?? DEFAULT_ASPECT_RATIO;

  const stageBox = computeContainBox(outerSize, effectiveAspectRatio);

  const handleVideoDimensions = (
    event: SyntheticEvent<HTMLVideoElement>,
  ): void => {
    const videoEl = event.currentTarget;
    if (
      videoEl.videoWidth > 0 &&
      videoEl.videoHeight > 0 &&
      Number.isFinite(videoEl.videoWidth) &&
      Number.isFinite(videoEl.videoHeight)
    ) {
      setDetectedAspectRatio(videoEl.videoWidth / videoEl.videoHeight);
    }
  };

  useEffect(() => {
    const outer = outerRef.current;
    if (!outer || typeof ResizeObserver === "undefined") {
      return;
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      const rect = outer.getBoundingClientRect();
      const width =
        rect.width > 0 ? rect.width : (entry?.contentRect.width ?? 0);
      const height =
        rect.height > 0 ? rect.height : (entry?.contentRect.height ?? 0);
      if (width > 0 && height > 0) {
        setOuterSize((prev) =>
          prev &&
          Math.abs(prev.width - width) < 0.5 &&
          Math.abs(prev.height - height) < 0.5
            ? prev
            : { width, height },
        );
      }
    });

    observer.observe(outer);
    return () => {
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const syncCanvasResolution = (
      fallbackWidth?: number,
      fallbackHeight?: number,
    ): void => {
      const rect = canvas.getBoundingClientRect();
      const cssWidth =
        rect.width > 0
          ? rect.width
          : canvas.clientWidth > 0
            ? canvas.clientWidth
            : (fallbackWidth ?? 0);
      const cssHeight =
        rect.height > 0
          ? rect.height
          : canvas.clientHeight > 0
            ? canvas.clientHeight
            : (fallbackHeight ?? 0);

      if (
        !Number.isFinite(cssWidth) ||
        !Number.isFinite(cssHeight) ||
        cssWidth <= 0 ||
        cssHeight <= 0
      ) {
        return;
      }

      const dpr =
        Number.isFinite(window.devicePixelRatio) && window.devicePixelRatio > 0
          ? window.devicePixelRatio
          : 1;
      const backingW = Math.max(1, Math.round(cssWidth * dpr));
      const backingH = Math.max(1, Math.round(cssHeight * dpr));

      if (canvas.width !== backingW || canvas.height !== backingH) {
        canvas.width = backingW;
        canvas.height = backingH;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        return;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawSkeletonFrame(ctx, cssWidth, cssHeight, latestDrawParamsRef.current);
    };

    syncCanvasResolution();

    let observer: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver((entries) => {
        const entry = entries[0];
        syncCanvasResolution(
          entry?.contentRect.width,
          entry?.contentRect.height,
        );
      });
      observer.observe(canvas);
    }

    const handleWindowResize = (): void => {
      syncCanvasResolution();
    };
    window.addEventListener("resize", handleWindowResize);

    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", handleWindowResize);
    };
  }, [canvasRef]);

  useEffect(() => {
    latestDrawParamsRef.current = {
      landmarks,
      hands,
      faceLandmarks,
      persons,
      showFaceMesh,
      angles,
      selectedJoints,
      viewMode,
      mirrored,
    };

    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const cssWidth = rect.width > 0 ? rect.width : canvas.clientWidth;
    const cssHeight = rect.height > 0 ? rect.height : canvas.clientHeight;
    if (cssWidth <= 0 || cssHeight <= 0) {
      return;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    drawSkeletonFrame(ctx, cssWidth, cssHeight, latestDrawParamsRef.current);
  }, [
    canvasRef,
    landmarks,
    hands,
    faceLandmarks,
    persons,
    showFaceMesh,
    angles,
    selectedJoints,
    viewMode,
    mirrored,
  ]);

  const videoClassName = [
    "block h-full w-full object-fill transition-opacity duration-200",
    mirrored ? "[transform:scaleX(-1)]" : "",
    viewMode === "blackout"
      ? "pointer-events-none opacity-0"
      : "opacity-100",
  ]
    .filter(Boolean)
    .join(" ");

  const statusMeta = getStatusMeta(status);
  const customDetail = statusMessage?.trim();
  const detailText =
    customDetail && customDetail.length > 0
      ? customDetail
      : statusMeta.defaultDetail;

  return (
    <div
      ref={outerRef}
      className="relative flex h-full w-full min-h-0 min-w-0 items-center justify-center overflow-hidden bg-[#05080f]"
    >
      <div
        data-testid="pose-stage"
        className="relative overflow-hidden rounded-lg border border-cyan-500/20 bg-[#05080f] shadow-[0_0_40px_rgba(6,182,212,0.08)]"
        style={{
          aspectRatio: String(effectiveAspectRatio),
          ...(stageBox
            ? { width: `${stageBox.width}px`, height: `${stageBox.height}px` }
            : { width: "100%", maxWidth: "100%", maxHeight: "100%" }),
        }}
      >
        <video
          ref={videoRef}
          data-testid="pose-video"
          autoPlay
          playsInline
          muted
          onLoadedMetadata={handleVideoDimensions}
          onResize={handleVideoDimensions}
          className={videoClassName}
        />
        <canvas
          ref={canvasRef}
          data-testid="pose-canvas"
          className="pointer-events-none absolute inset-0 h-full w-full"
        />
      </div>

      {status !== "ready" && (
        <div
          data-testid="status-overlay"
          role={statusMeta.isError ? "alert" : "status"}
          aria-live={statusMeta.isError ? "assertive" : "polite"}
          className="absolute inset-0 z-20 flex items-center justify-center bg-[#05080f]/80 p-4 backdrop-blur-sm"
        >
          <div className="w-full max-w-md rounded-xl border border-cyan-500/30 bg-slate-950/90 p-6 text-center shadow-[0_0_32px_rgba(34,211,238,0.12)]">
            <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-full border border-slate-700/80 bg-slate-900/90">
              {renderStatusIcon(status)}
            </div>
            <h2 className="font-mono text-base font-semibold tracking-wider text-slate-100 uppercase">
              {statusMeta.title}
            </h2>
            <p className="mt-2 text-sm leading-relaxed text-slate-300">
              {detailText}
            </p>
            {statusMeta.showRetry && (
              <button
                type="button"
                data-testid="retry-button"
                onClick={() => onRetry?.()}
                className="mt-5 inline-flex items-center gap-2 rounded-md border border-cyan-400/50 bg-cyan-500/15 px-4 py-2 font-mono text-xs font-semibold tracking-wider text-cyan-200 uppercase transition hover:border-cyan-300 hover:bg-cyan-500/25 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400"
              >
                <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
                Retry
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default SkeletonCanvas;
