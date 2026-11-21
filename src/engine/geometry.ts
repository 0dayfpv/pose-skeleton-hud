export type ModelVariant = "lite" | "full" | "heavy";
export type DelegateType = "GPU" | "CPU";
export type ViewMode = "overlay" | "blackout";
export type HandednessType = "Left" | "Right";
export type PersonId = 0 | 1;

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

export type FingerId = "thumb" | "index" | "middle" | "ring" | "pinky";

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

export interface DetailModulesConfig {
  enableHands: boolean;
  enableFace: boolean;
  showFaceMesh: boolean;
  maxPoses?: 1 | 2;
}

export interface TrackedHand {
  handedness: HandednessType;
  score: number;
  landmarks: Point3D[]; // 21 landmarks (indices 0..20)
  personId?: PersonId;
}

export interface TrackedPerson {
  personId: PersonId;
  label: "P1" | "P2";
  landmarks: Point3D[]; // 33 body landmarks
  angles: Partial<Record<JointId, number | null>>;
  hands: TrackedHand[];
  handMetrics: HandTelemetry[];
  faceLandmarks: Point3D[] | null;
  faceHeadMetrics: FaceHeadMetrics | null;
  centroid: { x: number; y: number };
}

export interface FingerAngles {
  thumb: number | null;
  index: number | null;
  middle: number | null;
  ring: number | null;
  pinky: number | null;
}

export interface HandTelemetry {
  handedness: HandednessType;
  score: number;
  fingerAngles: FingerAngles;
  extendedFingers: number; // 0..5
  pinchDistance: number; // normalized 3D distance between thumb_tip (4) and index_tip (8)
  isPinching: boolean;
  opennessPercent: number; // [0, 100]
}

export interface FaceHeadMetrics {
  headRollDeg: number; // tilt left/right
  headYawDeg: number; // turn left/right
  headPitchDeg: number; // nod up/down
  leftEyeOpenness: number; // [0, 100]
  rightEyeOpenness: number; // [0, 100]
  mouthOpenness: number; // [0, 100]
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
  hands?: TrackedHand[];
  handMetrics?: HandTelemetry[];
  faceLandmarksCount?: number;
  faceHeadMetrics?: FaceHeadMetrics | null;
  personsCount?: number;
  persons?: Array<{
    personId: PersonId;
    label: "P1" | "P2";
    landmarks: SnapshotLandmark[];
    activeJointAngles: SnapshotJointAngle[];
    handsCount: number;
    faceDetected: boolean;
    faceHeadMetrics: FaceHeadMetrics | null;
  }>;
}

export interface BuildSnapshotParams {
  landmarks: readonly Point3D[] | null;
  selectedJoints: readonly JointId[];
  angles?: Partial<Record<JointId, number | null>>;
  model: ModelVariant;
  mirrored: boolean;
  smoothing: SmoothingConfig;
  timestampMs?: number;
  hands?: readonly TrackedHand[];
  faceLandmarks?: readonly Point3D[] | null;
  persons?: readonly TrackedPerson[];
}

export const POSE_LANDMARK_NAMES: readonly string[] = [
  "nose",
  "left_eye_inner",
  "left_eye",
  "left_eye_outer",
  "right_eye_inner",
  "right_eye",
  "right_eye_outer",
  "left_ear",
  "right_ear",
  "mouth_left",
  "mouth_right",
  "left_shoulder",
  "right_shoulder",
  "left_elbow",
  "right_elbow",
  "left_wrist",
  "right_wrist",
  "left_pinky",
  "right_pinky",
  "left_index",
  "right_index",
  "left_thumb",
  "right_thumb",
  "left_hip",
  "right_hip",
  "left_knee",
  "right_knee",
  "left_ankle",
  "right_ankle",
  "left_heel",
  "right_heel",
  "left_foot_index",
  "right_foot_index",
];

export const HAND_LANDMARK_NAMES: readonly string[] = [
  "wrist",
  "thumb_cmc",
  "thumb_mcp",
  "thumb_ip",
  "thumb_tip",
  "index_finger_mcp",
  "index_finger_pip",
  "index_finger_dip",
  "index_finger_tip",
  "middle_finger_mcp",
  "middle_finger_pip",
  "middle_finger_dip",
  "middle_finger_tip",
  "ring_finger_mcp",
  "ring_finger_pip",
  "ring_finger_dip",
  "ring_finger_tip",
  "pinky_mcp",
  "pinky_pip",
  "pinky_dip",
  "pinky_tip",
];

export const FINGER_CHAINS: Readonly<Record<FingerId, readonly [number, number, number, number]>> = {
  thumb: [1, 2, 3, 4],
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
};

export const JOINT_DEFINITIONS: readonly JointDefinition[] = [
  {
    id: "left_elbow",
    label: "Left Elbow",
    shortLabel: "L.ELB",
    side: "left",
    indices: [11, 13, 15],
  },
  {
    id: "right_elbow",
    label: "Right Elbow",
    shortLabel: "R.ELB",
    side: "right",
    indices: [12, 14, 16],
  },
  {
    id: "left_shoulder",
    label: "Left Shoulder",
    shortLabel: "L.SHD",
    side: "left",
    indices: [23, 11, 13],
  },
  {
    id: "right_shoulder",
    label: "Right Shoulder",
    shortLabel: "R.SHD",
    side: "right",
    indices: [24, 12, 14],
  },
  {
    id: "left_hip",
    label: "Left Hip",
    shortLabel: "L.HIP",
    side: "left",
    indices: [11, 23, 25],
  },
  {
    id: "right_hip",
    label: "Right Hip",
    shortLabel: "R.HIP",
    side: "right",
    indices: [12, 24, 26],
  },
  {
    id: "left_knee",
    label: "Left Knee",
    shortLabel: "L.KNE",
    side: "left",
    indices: [23, 25, 27],
  },
  {
    id: "right_knee",
    label: "Right Knee",
    shortLabel: "R.KNE",
    side: "right",
    indices: [24, 26, 28],
  },
];

export const JOINT_DEFINITION_MAP: Readonly<Record<JointId, JointDefinition>> =
  JOINT_DEFINITIONS.reduce(
    (acc, def) => {
      acc[def.id] = def;
      return acc;
    },
    {} as Record<JointId, JointDefinition>,
  );

const MIN_RAY_MAGNITUDE = 1e-8;
const DEFAULT_MIN_VISIBILITY = 0.4;
const LANDMARK_COUNT = 33;

function isFinitePoint(p: Point3D | undefined | null): p is Point3D {
  return (
    Boolean(p) &&
    Number.isFinite(p!.x) &&
    Number.isFinite(p!.y) &&
    Number.isFinite(p!.z)
  );
}

function clonePoint(lm: Point3D): Point3D {
  return {
    x: lm.x,
    y: lm.y,
    z: lm.z,
    ...(lm.visibility !== undefined ? { visibility: lm.visibility } : {}),
    ...(lm.presence !== undefined ? { presence: lm.presence } : {}),
  };
}

function distance3D(a: Point3D, b: Point3D): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

export function calculateAngle3D(
  a: Point3D,
  b: Point3D,
  c: Point3D,
): number | null {
  if (!isFinitePoint(a) || !isFinitePoint(b) || !isFinitePoint(c)) {
    return null;
  }

  const ux = a.x - b.x;
  const uy = a.y - b.y;
  const uz = a.z - b.z;

  const vx = c.x - b.x;
  const vy = c.y - b.y;
  const vz = c.z - b.z;

  const magU = Math.hypot(ux, uy, uz);
  const magV = Math.hypot(vx, vy, vz);

  if (
    !Number.isFinite(magU) ||
    !Number.isFinite(magV) ||
    magU < MIN_RAY_MAGNITUDE ||
    magV < MIN_RAY_MAGNITUDE
  ) {
    return null;
  }

  const cosRaw =
    (ux / magU) * (vx / magV) +
    (uy / magU) * (vy / magV) +
    (uz / magU) * (vz / magV);

  if (!Number.isFinite(cosRaw)) {
    return null;
  }

  const cosClamped = Math.max(-1, Math.min(1, cosRaw));
  const deg = (Math.acos(cosClamped) * 180) / Math.PI;

  return Math.max(0, Math.min(180, deg));
}

export function computeJointAngles(
  landmarks: readonly Point3D[] | null,
  selectedJoints: readonly JointId[],
  minVisibility = DEFAULT_MIN_VISIBILITY,
): Record<JointId, number | null> {
  const result: Record<JointId, number | null> = {
    left_elbow: null,
    right_elbow: null,
    left_shoulder: null,
    right_shoulder: null,
    left_hip: null,
    right_hip: null,
    left_knee: null,
    right_knee: null,
  };

  if (!landmarks || landmarks.length === 0) {
    return result;
  }

  const threshold = Number.isFinite(minVisibility)
    ? minVisibility
    : DEFAULT_MIN_VISIBILITY;

  for (const jointId of selectedJoints) {
    const def = JOINT_DEFINITION_MAP[jointId];
    if (!def) {
      continue;
    }

    const [iA, iB, iC] = def.indices;
    const a = landmarks[iA];
    const b = landmarks[iB];
    const c = landmarks[iC];

    if (!a || !b || !c) {
      continue;
    }

    const visA = Number.isFinite(a.visibility) ? a.visibility! : 0;
    const visB = Number.isFinite(b.visibility) ? b.visibility! : 0;
    const visC = Number.isFinite(c.visibility) ? c.visibility! : 0;

    if (visA < threshold || visB < threshold || visC < threshold) {
      continue;
    }

    result[jointId] = calculateAngle3D(a, b, c);
  }

  return result;
}

/**
 * Computes detailed finger articulation angles, extended finger count,
 * thumb-index pinch distance, and overall hand openness from a 21-point hand.
 */
export function computeHandMetrics(hand: TrackedHand): HandTelemetry {
  const lms = hand.landmarks;
  const fingerAngles: FingerAngles = {
    thumb: lms.length >= 21 ? calculateAngle3D(lms[1]!, lms[2]!, lms[4]!) : null,
    index: lms.length >= 21 ? calculateAngle3D(lms[5]!, lms[6]!, lms[8]!) : null,
    middle: lms.length >= 21 ? calculateAngle3D(lms[9]!, lms[10]!, lms[12]!) : null,
    ring: lms.length >= 21 ? calculateAngle3D(lms[13]!, lms[14]!, lms[16]!) : null,
    pinky: lms.length >= 21 ? calculateAngle3D(lms[17]!, lms[18]!, lms[20]!) : null,
  };

  let extendedFingers = 0;
  const anglesList: (number | null)[] = [
    fingerAngles.thumb,
    fingerAngles.index,
    fingerAngles.middle,
    fingerAngles.ring,
    fingerAngles.pinky,
  ];

  let angleSum = 0;
  let validAngleCount = 0;
  for (let i = 0; i < anglesList.length; i++) {
    const deg = anglesList[i];
    if (deg !== null && Number.isFinite(deg)) {
      // Thumb threshold ~145°, other 4 fingers ~150° when extended
      const threshold = i === 0 ? 145 : 150;
      if (deg >= threshold) {
        extendedFingers += 1;
      }
      angleSum += deg;
      validAngleCount += 1;
    }
  }

  let pinchDistance = 1;
  if (
    lms.length >= 21 &&
    isFinitePoint(lms[4]) &&
    isFinitePoint(lms[8])
  ) {
    const rawDist = distance3D(lms[4]!, lms[8]!);
    pinchDistance = Number.isFinite(rawDist)
      ? Math.round(rawDist * 1000) / 1000
      : 1;
  }

  const avgAngle = validAngleCount > 0 ? angleSum / validAngleCount : 90;
  // Map average flexion [65° (closed fist) .. 175° (open palm)] to [0..100%]
  const opennessPercent = Math.round(
    Math.max(0, Math.min(100, ((avgAngle - 65) / 110) * 100)),
  );

  return {
    handedness: hand.handedness,
    score: Number.isFinite(hand.score)
      ? Math.round(Math.max(0, Math.min(1, hand.score)) * 100) / 100
      : 1,
    fingerAngles,
    extendedFingers,
    pinchDistance,
    isPinching: pinchDistance < 0.055,
    opennessPercent,
  };
}

/**
 * Computes 3D head orientation (Roll, Yaw, Pitch) and eye/mouth openness
 * percentages from MediaPipe FaceLandmarker's 478 landmarks.
 */
export function computeFaceHeadMetrics(
  faceLandmarks: readonly Point3D[] | null,
): FaceHeadMetrics | null {
  if (!faceLandmarks || faceLandmarks.length < 468) {
    return null;
  }

  const noseTip = faceLandmarks[1];
  const forehead = faceLandmarks[10];
  const chin = faceLandmarks[152];
  const leftCheek = faceLandmarks[234];
  const rightCheek = faceLandmarks[454];
  const rightEyeOuter = faceLandmarks[33];
  const rightEyeInner = faceLandmarks[133];
  const rightEyeTop = faceLandmarks[159];
  const rightEyeBottom = faceLandmarks[145];
  const leftEyeOuter = faceLandmarks[263];
  const leftEyeInner = faceLandmarks[362];
  const leftEyeTop = faceLandmarks[386];
  const leftEyeBottom = faceLandmarks[374];
  const upperLip = faceLandmarks[13];
  const lowerLip = faceLandmarks[14];
  const mouthLeft = faceLandmarks[61];
  const mouthRight = faceLandmarks[291];

  if (
    !isFinitePoint(noseTip) ||
    !isFinitePoint(forehead) ||
    !isFinitePoint(chin) ||
    !isFinitePoint(leftCheek) ||
    !isFinitePoint(rightCheek) ||
    !isFinitePoint(rightEyeOuter) ||
    !isFinitePoint(leftEyeOuter)
  ) {
    return null;
  }

  // Head Roll (tilt): angle of the inter-ocular line (33 -> 263)
  const eyeDx = leftEyeOuter.x - rightEyeOuter.x;
  const eyeDy = leftEyeOuter.y - rightEyeOuter.y;
  const rawRoll = (Math.atan2(eyeDy, Math.max(1e-6, Math.abs(eyeDx))) * 180) / Math.PI;
  const headRollDeg = Math.round(Math.max(-90, Math.min(90, rawRoll)) * 10) / 10;

  // Head Yaw (left/right turn): horizontal asymmetry of nose tip between cheeks
  const faceWidth = Math.max(1e-5, distance3D(leftCheek, rightCheek));
  const midCheekX = (leftCheek.x + rightCheek.x) * 0.5;
  const rawYaw = ((noseTip.x - midCheekX) / (faceWidth * 0.5)) * 45;
  const headYawDeg = Math.round(Math.max(-75, Math.min(75, rawYaw)) * 10) / 10;

  // Head Pitch (nod up/down): vertical position of nose tip relative to forehead-chin midpoint
  const faceHeight = Math.max(1e-5, distance3D(forehead, chin));
  const midFaceY = (forehead.y + chin.y) * 0.5;
  const rawPitch = ((noseTip.y - midFaceY) / (faceHeight * 0.5) - 0.12) * 60;
  const headPitchDeg =
    Math.round(Math.max(-60, Math.min(60, rawPitch)) * 10) / 10;

  // Eye Aspect Ratio (EAR) -> Openness [0..100%]
  let rightEyeOpenness = 100;
  if (
    isFinitePoint(rightEyeInner) &&
    isFinitePoint(rightEyeTop) &&
    isFinitePoint(rightEyeBottom)
  ) {
    const horiz = Math.max(1e-5, distance3D(rightEyeOuter, rightEyeInner));
    const vert = distance3D(rightEyeTop, rightEyeBottom);
    const ear = vert / horiz;
    rightEyeOpenness = Math.round(
      Math.max(0, Math.min(100, ((ear - 0.08) / 0.24) * 100)),
    );
  }

  let leftEyeOpenness = 100;
  if (
    isFinitePoint(leftEyeInner) &&
    isFinitePoint(leftEyeTop) &&
    isFinitePoint(leftEyeBottom)
  ) {
    const horiz = Math.max(1e-5, distance3D(leftEyeOuter, leftEyeInner));
    const vert = distance3D(leftEyeTop, leftEyeBottom);
    const ear = vert / horiz;
    leftEyeOpenness = Math.round(
      Math.max(0, Math.min(100, ((ear - 0.08) / 0.24) * 100)),
    );
  }

  let mouthOpenness = 0;
  if (
    isFinitePoint(upperLip) &&
    isFinitePoint(lowerLip) &&
    isFinitePoint(mouthLeft) &&
    isFinitePoint(mouthRight)
  ) {
    const mouthW = Math.max(1e-5, distance3D(mouthLeft, mouthRight));
    const mouthH = distance3D(upperLip, lowerLip);
    const mar = mouthH / mouthW;
    mouthOpenness = Math.round(
      Math.max(0, Math.min(100, ((mar - 0.02) / 0.55) * 100)),
    );
  }

  return {
    headRollDeg: headRollDeg === 0 ? 0 : headRollDeg,
    headYawDeg: headYawDeg === 0 ? 0 : headYawDeg,
    headPitchDeg: headPitchDeg === 0 ? 0 : headPitchDeg,
    leftEyeOpenness,
    rightEyeOpenness,
    mouthOpenness,
  };
}

interface LandmarkFilterState {
  x: number;
  y: number;
  z: number;
  dx: number;
  dy: number;
  dz: number;
}

function alphaFor(cutoffHz: number, dtSec: number): number {
  const tau = 1 / (2 * Math.PI * cutoffHz);
  return 1 / (1 + tau / dtSec);
}

function stepAxis(
  rawVal: number,
  prevVal: number,
  prevDeriv: number,
  dtSec: number,
  alphaD: number,
  fMin: number,
  beta: number,
): { value: number; deriv: number } {
  if (!Number.isFinite(rawVal)) {
    return { value: prevVal, deriv: prevDeriv };
  }
  const rawDeriv = (rawVal - prevVal) / dtSec;
  const dHat = alphaD * rawDeriv + (1 - alphaD) * prevDeriv;
  const cutoff = fMin + beta * Math.abs(dHat);
  const alpha = alphaFor(cutoff, dtSec);
  const xHat = alpha * rawVal + (1 - alpha) * prevVal;
  return { value: xHat, deriv: dHat };
}

export class PoseSmoother {
  private lastTimestampMs: number | null = null;
  private states: Map<number, LandmarkFilterState> = new Map();

  reset(): void {
    this.states.clear();
    this.lastTimestampMs = null;
  }

  filter(
    landmarks: readonly Point3D[],
    timestampMs: number,
    config: SmoothingConfig,
  ): Point3D[] {
    if (!config.enabled) {
      this.reset();
      return landmarks.map(clonePoint);
    }

    const s = Number.isFinite(config.strength)
      ? Math.max(0, Math.min(1, config.strength))
      : 0.5;
    const fMin = 4.0 * Math.pow(0.08, s);
    const beta = 10.0 * (1 - 0.7 * s);
    const dCutoff = 0.8;

    const dtSec =
      this.lastTimestampMs === null
        ? Number.NaN
        : (timestampMs - this.lastTimestampMs) / 1000;

    if (
      this.lastTimestampMs === null ||
      !Number.isFinite(dtSec) ||
      dtSec <= 0 ||
      dtSec > 1.0
    ) {
      this.states.clear();
      this.lastTimestampMs = Number.isFinite(timestampMs) ? timestampMs : null;

      return landmarks.map((lm, i) => {
        if (isFinitePoint(lm)) {
          this.states.set(i, {
            x: lm.x,
            y: lm.y,
            z: lm.z,
            dx: 0,
            dy: 0,
            dz: 0,
          });
        }
        return clonePoint(lm);
      });
    }

    this.lastTimestampMs = timestampMs;
    const alphaD = alphaFor(dCutoff, dtSec);

    return landmarks.map((lm, i) => {
      const prev = this.states.get(i);
      if (!prev) {
        if (isFinitePoint(lm)) {
          this.states.set(i, {
            x: lm.x,
            y: lm.y,
            z: lm.z,
            dx: 0,
            dy: 0,
            dz: 0,
          });
        }
        return clonePoint(lm);
      }

      const nextX = stepAxis(lm.x, prev.x, prev.dx, dtSec, alphaD, fMin, beta);
      const nextY = stepAxis(lm.y, prev.y, prev.dy, dtSec, alphaD, fMin, beta);
      const nextZ = stepAxis(lm.z, prev.z, prev.dz, dtSec, alphaD, fMin, beta);

      this.states.set(i, {
        x: nextX.value,
        y: nextY.value,
        z: nextZ.value,
        dx: nextX.deriv,
        dy: nextY.deriv,
        dz: nextZ.deriv,
      });

      return {
        x: nextX.value,
        y: nextY.value,
        z: nextZ.value,
        ...(lm.visibility !== undefined ? { visibility: lm.visibility } : {}),
        ...(lm.presence !== undefined ? { presence: lm.presence } : {}),
      };
    });
  }
}

function toFiniteCoord(v: number | undefined): number {
  return v !== undefined && Number.isFinite(v) ? (v === 0 ? 0 : v) : 0;
}

export function buildPoseSnapshot(params: BuildSnapshotParams): PoseSnapshot {
  const rawTs = params.timestampMs;
  const safeMs =
    rawTs !== undefined &&
    Number.isFinite(rawTs) &&
    Number.isFinite(new Date(rawTs).getTime())
      ? rawTs === 0
        ? 0
        : rawTs
      : Date.now();
  const timestamp = new Date(safeMs).toISOString();

  const poseDetected = Boolean(
    params.landmarks && params.landmarks.length >= LANDMARK_COUNT,
  );

  const landmarks: SnapshotLandmark[] = [];
  for (let i = 0; i < LANDMARK_COUNT; i++) {
    const lm = params.landmarks?.[i];
    const visibility =
      lm && lm.visibility !== undefined && Number.isFinite(lm.visibility)
        ? Math.max(0, Math.min(1, lm.visibility === 0 ? 0 : lm.visibility))
        : 0;

    landmarks.push({
      index: i,
      name: POSE_LANDMARK_NAMES[i] ?? `landmark_${i}`,
      x: toFiniteCoord(lm?.x),
      y: toFiniteCoord(lm?.y),
      z: toFiniteCoord(lm?.z),
      visibility,
    });
  }

  const resolvedAngles: Partial<Record<JointId, number | null>> =
    params.angles ?? computeJointAngles(params.landmarks, params.selectedJoints);

  const activeJointAngles: SnapshotJointAngle[] = [];
  for (const jointId of params.selectedJoints) {
    const def = JOINT_DEFINITION_MAP[jointId];
    if (!def) {
      continue;
    }
    const rawDeg = resolvedAngles[jointId];
    const degrees =
      rawDeg !== undefined && rawDeg !== null && Number.isFinite(rawDeg)
        ? rawDeg === 0
          ? 0
          : rawDeg
        : null;
    activeJointAngles.push({
      joint: jointId,
      label: def.label,
      degrees,
    });
  }

  const safeStrength = Number.isFinite(params.smoothing.strength)
    ? Math.max(0, Math.min(1, params.smoothing.strength))
    : 0.5;

  const snapshot: PoseSnapshot = {
    schemaVersion: "1.0.0",
    timestamp,
    timestampMs: safeMs,
    poseDetected,
    model: params.model,
    mirrored: Boolean(params.mirrored),
    smoothing: {
      enabled: Boolean(params.smoothing.enabled),
      strength: safeStrength === 0 ? 0 : safeStrength,
    },
    landmarks,
    activeJointAngles,
  };

  if (params.hands !== undefined) {
    snapshot.hands = params.hands.map((h) => ({
      handedness: h.handedness,
      score: h.score,
      landmarks: h.landmarks.map(clonePoint),
    }));
    snapshot.handMetrics = params.hands.map((h) => computeHandMetrics(h));
  }

  if (params.faceLandmarks !== undefined) {
    snapshot.faceLandmarksCount = params.faceLandmarks?.length ?? 0;
    snapshot.faceHeadMetrics = computeFaceHeadMetrics(params.faceLandmarks);
  }

  if (params.persons !== undefined) {
    snapshot.personsCount = params.persons.length;
    snapshot.persons = params.persons.map((person) => {
      const personLandmarks: SnapshotLandmark[] = [];
      for (let i = 0; i < LANDMARK_COUNT; i++) {
        const lm = person.landmarks[i];
        const vis =
          lm && lm.visibility !== undefined && Number.isFinite(lm.visibility)
            ? Math.max(0, Math.min(1, lm.visibility === 0 ? 0 : lm.visibility))
            : 0;
        personLandmarks.push({
          index: i,
          name: POSE_LANDMARK_NAMES[i] ?? `landmark_${i}`,
          x: toFiniteCoord(lm?.x),
          y: toFiniteCoord(lm?.y),
          z: toFiniteCoord(lm?.z),
          visibility: vis,
        });
      }
      const pAngles: SnapshotJointAngle[] = [];
      for (const jointId of params.selectedJoints) {
        const def = JOINT_DEFINITION_MAP[jointId];
        if (!def) continue;
        const rawDeg = person.angles[jointId];
        const degrees =
          rawDeg !== undefined && rawDeg !== null && Number.isFinite(rawDeg)
            ? rawDeg === 0
              ? 0
              : rawDeg
            : null;
        pAngles.push({
          joint: jointId,
          label: def.label,
          degrees,
        });
      }
      return {
        personId: person.personId,
        label: person.label,
        landmarks: personLandmarks,
        activeJointAngles: pAngles,
        handsCount: person.hands.length,
        faceDetected: Boolean(
          person.faceLandmarks && person.faceLandmarks.length >= 468,
        ),
        faceHeadMetrics: person.faceHeadMetrics,
      };
    });
  }

  return snapshot;
}

export function serializePoseSnapshot(params: BuildSnapshotParams): string {
  return JSON.stringify(buildPoseSnapshot(params), null, 2);
}

const TORSO_ANCHOR_INDICES = [11, 12, 23, 24, 0] as const;

export function computePoseCentroid(
  landmarks: readonly Point3D[],
): { x: number; y: number } {
  let sumX = 0;
  let sumY = 0;
  let count = 0;

  for (const idx of TORSO_ANCHOR_INDICES) {
    const pt = landmarks[idx];
    if (
      pt &&
      Number.isFinite(pt.x) &&
      Number.isFinite(pt.y) &&
      (pt.visibility === undefined || pt.visibility >= 0.2)
    ) {
      sumX += pt.x;
      sumY += pt.y;
      count += 1;
    }
  }

  if (count > 0) {
    return { x: sumX / count, y: sumY / count };
  }

  for (let i = 0; i < landmarks.length; i++) {
    const pt = landmarks[i];
    if (pt && Number.isFinite(pt.x) && Number.isFinite(pt.y)) {
      sumX += pt.x;
      sumY += pt.y;
      count += 1;
    }
  }

  return count > 0 ? { x: sumX / count, y: sumY / count } : { x: 0.5, y: 0.5 };
}

function distSq2D(
  a: { x: number; y: number },
  b: { x: number; y: number },
): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function computeHeadAnchor(landmarks: readonly Point3D[]): {
  x: number;
  y: number;
} {
  const nose = landmarks[0];
  if (nose && Number.isFinite(nose.x) && Number.isFinite(nose.y)) {
    return { x: nose.x, y: nose.y };
  }
  const lSh = landmarks[11];
  const rSh = landmarks[12];
  if (
    lSh &&
    rSh &&
    Number.isFinite(lSh.x) &&
    Number.isFinite(lSh.y) &&
    Number.isFinite(rSh.x) &&
    Number.isFinite(rSh.y)
  ) {
    return {
      x: (lSh.x + rSh.x) * 0.5,
      y: (lSh.y + rSh.y) * 0.5 - 0.12,
    };
  }
  return computePoseCentroid(landmarks);
}

function computeHandDistanceToPerson(
  handWrist: Point3D,
  personLandmarks: readonly Point3D[],
  personCentroid: { x: number; y: number },
): number {
  let best = distSq2D(handWrist, personCentroid);
  for (const wristIdx of [15, 16, 17, 18, 19, 20] as const) {
    const pw = personLandmarks[wristIdx];
    if (
      pw &&
      Number.isFinite(pw.x) &&
      Number.isFinite(pw.y) &&
      (pw.visibility === undefined || pw.visibility >= 0.15)
    ) {
      const d = distSq2D(handWrist, pw);
      if (d < best) {
        best = d;
      }
    }
  }
  return best;
}

/**
 * Multi-person tracker & spatial associator for up to 2 simultaneous people.
 * Guarantees temporal ID stability (P1 vs P2), per-person One-Euro smoothing,
 * and spatial binding of up to 4 hands and 2 478-point face meshes.
 */
export class MultiPersonTracker {
  private readonly poseSmoothers: [PoseSmoother, PoseSmoother] = [
    new PoseSmoother(),
    new PoseSmoother(),
  ];
  private readonly faceSmoothers: [PoseSmoother, PoseSmoother] = [
    new PoseSmoother(),
    new PoseSmoother(),
  ];
  private readonly handSmoothers: Map<string, PoseSmoother> = new Map();
  private prevCentroids: Map<PersonId, { x: number; y: number }> = new Map();

  reset(): void {
    this.poseSmoothers[0].reset();
    this.poseSmoothers[1].reset();
    this.faceSmoothers[0].reset();
    this.faceSmoothers[1].reset();
    for (const s of this.handSmoothers.values()) {
      s.reset();
    }
    this.handSmoothers.clear();
    this.prevCentroids.clear();
  }

  private getHandSmoother(key: string): PoseSmoother {
    let smoother = this.handSmoothers.get(key);
    if (!smoother) {
      smoother = new PoseSmoother();
      this.handSmoothers.set(key, smoother);
    }
    return smoother;
  }

  update(params: {
    poses: readonly (readonly Point3D[])[];
    hands: readonly TrackedHand[];
    faces: readonly (readonly Point3D[])[];
    selectedJoints: readonly JointId[];
    timestampMs: number;
    smoothing: SmoothingConfig;
  }): TrackedPerson[] {
    const validPoses = params.poses
      .filter((p) => p && p.length >= LANDMARK_COUNT)
      .slice(0, 2);

    if (validPoses.length === 0) {
      this.prevCentroids.clear();
      return [];
    }

    const rawCentroids = validPoses.map((p) => computePoseCentroid(p));
    let assignedIds: PersonId[];

    if (validPoses.length === 1) {
      const c0 = rawCentroids[0]!;
      const prev0 = this.prevCentroids.get(0);
      const prev1 = this.prevCentroids.get(1);
      if (prev0 && prev1) {
        const d0 = distSq2D(c0, prev0);
        const d1 = distSq2D(c0, prev1);
        assignedIds = [d1 < d0 * 0.7 ? 1 : 0];
      } else {
        assignedIds = [0];
      }
    } else {
      const cA = rawCentroids[0]!;
      const cB = rawCentroids[1]!;
      const prev0 = this.prevCentroids.get(0);
      const prev1 = this.prevCentroids.get(1);

      if (prev0 && prev1) {
        const costKeep = distSq2D(cA, prev0) + distSq2D(cB, prev1);
        const costSwap = distSq2D(cA, prev1) + distSq2D(cB, prev0);
        assignedIds = costSwap < costKeep ? [1, 0] : [0, 1];
      } else if (prev0) {
        const dA = distSq2D(cA, prev0);
        const dB = distSq2D(cB, prev0);
        assignedIds = dB < dA ? [1, 0] : [0, 1];
      } else {
        // Initial 2-person acquisition: sort left-to-right in camera space
        assignedIds = cA.x <= cB.x ? [0, 1] : [1, 0];
      }
    }

    const nextCentroids = new Map<PersonId, { x: number; y: number }>();
    const personEntries: Array<{
      personId: PersonId;
      landmarks: Point3D[];
      centroid: { x: number; y: number };
      headAnchor: { x: number; y: number };
    }> = [];

    for (let i = 0; i < validPoses.length; i++) {
      const pid = assignedIds[i]!;
      const smoothedPose = this.poseSmoothers[pid].filter(
        validPoses[i]!,
        params.timestampMs,
        params.smoothing,
      );
      const centroid = computePoseCentroid(smoothedPose);
      const headAnchor = computeHeadAnchor(smoothedPose);
      nextCentroids.set(pid, centroid);
      personEntries.push({
        personId: pid,
        landmarks: smoothedPose,
        centroid,
        headAnchor,
      });
    }

    // Sort output so P1 (0) is always before P2 (1) when both are present
    personEntries.sort((a, b) => a.personId - b.personId);
    this.prevCentroids = nextCentroids;

    // Associate detected 478-point Faces (up to 2) to closest person headAnchor
    const assignedFaces = new Map<PersonId, Point3D[]>();
    const validFaces = params.faces
      .filter((f) => f && f.length >= 468)
      .slice(0, 2);
    const usedPersonForFace = new Set<PersonId>();

    for (const rawFace of validFaces) {
      const nosePt = rawFace[1] ?? rawFace[0]!;
      let bestPid: PersonId | null = null;
      let bestDist = Number.POSITIVE_INFINITY;

      for (const pe of personEntries) {
        if (usedPersonForFace.has(pe.personId)) continue;
        const d = distSq2D(nosePt, pe.headAnchor);
        if (d < bestDist) {
          bestDist = d;
          bestPid = pe.personId;
        }
      }

      if (bestPid !== null) {
        usedPersonForFace.add(bestPid);
        const smoothedFace = this.faceSmoothers[bestPid].filter(
          rawFace,
          params.timestampMs,
          params.smoothing,
        );
        assignedFaces.set(bestPid, smoothedFace);
      }
    }

    // Associate detected 21-point Hands (up to 4) to closest person wrist/torso
    const assignedHands = new Map<PersonId, TrackedHand[]>();
    for (const pe of personEntries) {
      assignedHands.set(pe.personId, []);
    }

    const slotCounters = new Map<string, number>();
    for (const rawHand of params.hands) {
      if (!rawHand.landmarks || rawHand.landmarks.length < 21) continue;
      const wristPt = rawHand.landmarks[0]!;

      let bestPid: PersonId = personEntries[0]!.personId;
      let bestDist = Number.POSITIVE_INFINITY;
      for (const pe of personEntries) {
        const d = computeHandDistanceToPerson(
          wristPt,
          pe.landmarks,
          pe.centroid,
        );
        if (d < bestDist) {
          bestDist = d;
          bestPid = pe.personId;
        }
      }

      const baseKey = `${bestPid}-${rawHand.handedness}`;
      const occ = slotCounters.get(baseKey) ?? 0;
      slotCounters.set(baseKey, occ + 1);
      const smootherKey = `${baseKey}-${occ}`;
      const smoothedHandLms = this.getHandSmoother(smootherKey).filter(
        rawHand.landmarks,
        params.timestampMs,
        params.smoothing,
      );

      assignedHands.get(bestPid)?.push({
        handedness: rawHand.handedness,
        score: rawHand.score,
        landmarks: smoothedHandLms,
        personId: bestPid,
      });
    }

    return personEntries.map((pe) => {
      const pid = pe.personId;
      const personHands = assignedHands.get(pid) ?? [];
      const personFace = assignedFaces.get(pid) ?? null;
      return {
        personId: pid,
        label: pid === 0 ? "P1" : "P2",
        landmarks: pe.landmarks,
        angles: computeJointAngles(pe.landmarks, params.selectedJoints),
        hands: personHands,
        handMetrics: personHands.map((h) => computeHandMetrics(h)),
        faceLandmarks: personFace,
        faceHeadMetrics: computeFaceHeadMetrics(personFace),
        centroid: pe.centroid,
      };
    });
  }
}
