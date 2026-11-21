import { describe, expect, it } from "vitest";
import {
  JOINT_DEFINITIONS,
  MultiPersonTracker,
  POSE_LANDMARK_NAMES,
  PoseSmoother,
  buildPoseSnapshot,
  calculateAngle3D,
  computeFaceHeadMetrics,
  computeHandMetrics,
  computeJointAngles,
  serializePoseSnapshot,
  type JointId,
  type Point3D,
  type PoseSnapshot,
  type SmoothingConfig,
  type TrackedHand,
} from "./geometry";

const ALL_JOINTS: readonly JointId[] = [
  "left_elbow",
  "right_elbow",
  "left_shoulder",
  "right_shoulder",
  "left_hip",
  "right_hip",
  "left_knee",
  "right_knee",
];

function createNeutralLandmarks(visibility = 0.95): Point3D[] {
  return Array.from({ length: 33 }, (_, i) => ({
    x: 0.5 + (i % 5) * 0.02,
    y: 0.1 + Math.floor(i / 5) * 0.08,
    z: 0,
    visibility,
  }));
}

function mean(values: readonly number[]): number {
  return values.reduce((acc, v) => acc + v, 0) / values.length;
}

function variance(values: readonly number[]): number {
  const m = mean(values);
  return mean(values.map((v) => (v - m) ** 2));
}

function rmse(actual: readonly number[], truth: readonly number[]): number {
  const mse = mean(actual.map((v, i) => (v - truth[i]) ** 2));
  return Math.sqrt(mse);
}

function diffs(values: readonly number[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < values.length; i++) {
    out.push(values[i] - values[i - 1]);
  }
  return out;
}

describe("calculateAngle3D", () => {
  it("computes 0° for collinear same-direction vectors (axis-aligned and non-axis)", () => {
    const axis = calculateAngle3D(
      { x: 1, y: 0, z: 0 },
      { x: 0, y: 0, z: 0 },
      { x: 2, y: 0, z: 0 },
    );
    expect(axis).not.toBeNull();
    expect(Math.abs(axis! - 0)).toBeLessThan(0.5);
    expect(axis!).toBeCloseTo(0, 6);

    const nonAxis = calculateAngle3D(
      { x: 0.3, y: 0.7, z: 0.9 },
      { x: 0, y: 0, z: 0 },
      { x: 0.6, y: 1.4, z: 1.8 },
    );
    expect(nonAxis).not.toBeNull();
    expect(nonAxis!).toBeCloseTo(0, 6);
  });

  it("computes 45° accurately in XY and YZ planes within ±0.5°", () => {
    const xy = calculateAngle3D(
      { x: 1, y: 0, z: 0 },
      { x: 0, y: 0, z: 0 },
      { x: 1, y: 1, z: 0 },
    );
    expect(xy).not.toBeNull();
    expect(Math.abs(xy! - 45)).toBeLessThan(0.5);
    expect(xy!).toBeCloseTo(45, 6);

    const yz = calculateAngle3D(
      { x: 0, y: 0, z: 1 },
      { x: 0, y: 0, z: 0 },
      { x: 0, y: 1, z: 1 },
    );
    expect(yz).not.toBeNull();
    expect(Math.abs(yz! - 45)).toBeLessThan(0.5);
    expect(yz!).toBeCloseTo(45, 6);
  });

  it("computes 90° accurately for planar and 3D orthogonal vectors", () => {
    const planar = calculateAngle3D(
      { x: 1, y: 0, z: 0 },
      { x: 0, y: 0, z: 0 },
      { x: 0, y: 1, z: 0 },
    );
    expect(planar).not.toBeNull();
    expect(Math.abs(planar! - 90)).toBeLessThan(0.5);
    expect(planar!).toBeCloseTo(90, 6);

    const orthogonal3D = calculateAngle3D(
      { x: 1, y: 1, z: 0 },
      { x: 0, y: 0, z: 0 },
      { x: 1, y: -1, z: 2 },
    );
    expect(orthogonal3D).not.toBeNull();
    expect(Math.abs(orthogonal3D! - 90)).toBeLessThan(0.5);
    expect(orthogonal3D!).toBeCloseTo(90, 6);
  });

  it("computes 180° for collinear opposite vectors (axis-aligned and non-axis)", () => {
    const axis = calculateAngle3D(
      { x: -1, y: 0, z: 0 },
      { x: 0, y: 0, z: 0 },
      { x: 2, y: 0, z: 0 },
    );
    expect(axis).not.toBeNull();
    expect(Math.abs(axis! - 180)).toBeLessThan(0.5);
    expect(axis!).toBeCloseTo(180, 6);

    const nonAxis = calculateAngle3D(
      { x: 0.3, y: 0.7, z: 0.9 },
      { x: 0, y: 0, z: 0 },
      { x: -0.6, y: -1.4, z: -1.8 },
    );
    expect(nonAxis).not.toBeNull();
    expect(Math.abs(nonAxis! - 180)).toBeLessThan(0.5);
    expect(nonAxis!).toBeCloseTo(180, 6);
  });

  it("computes true 3D non-planar angles and shifted vertices accurately", () => {
    const diagExpected = Math.acos(1 / Math.sqrt(3)) * (180 / Math.PI);
    const diag = calculateAngle3D(
      { x: 1, y: 1, z: 1 },
      { x: 0, y: 0, z: 0 },
      { x: 1, y: 0, z: 0 },
    );
    expect(diag).not.toBeNull();
    expect(Math.abs(diag! - diagExpected)).toBeLessThan(0.5);
    expect(diag!).toBeCloseTo(diagExpected, 6);

    const tetraExpected = Math.acos(1 / 3) * (180 / Math.PI);
    const tetra = calculateAngle3D(
      { x: 1, y: 1, z: 1 },
      { x: 0, y: 0, z: 0 },
      { x: 1, y: 1, z: -1 },
    );
    expect(tetra).not.toBeNull();
    expect(tetra!).toBeCloseTo(tetraExpected, 6);

    const shiftedDiag = calculateAngle3D(
      { x: 13.5, y: -6.25, z: 4.125 },
      { x: 12.5, y: -7.25, z: 3.125 },
      { x: 13.5, y: -7.25, z: 3.125 },
    );
    expect(shiftedDiag).not.toBeNull();
    expect(shiftedDiag!).toBeCloseTo(diagExpected, 6);

    const shiftedRight = calculateAngle3D(
      { x: 0.8, y: 0.4, z: -0.2 },
      { x: 0.5, y: 0.4, z: -0.2 },
      { x: 0.5, y: 0.7, z: -0.2 },
    );
    expect(shiftedRight).not.toBeNull();
    expect(shiftedRight!).toBeCloseTo(90, 6);
  });

  it("is symmetric with respect to ray order (a,b,c) vs (c,b,a)", () => {
    const a: Point3D = { x: 1, y: 2, z: 3 };
    const b: Point3D = { x: 0, y: 0, z: 0 };
    const c: Point3D = { x: -2, y: 4, z: -1 };

    const forward = calculateAngle3D(a, b, c);
    const reverse = calculateAngle3D(c, b, a);
    expect(forward).not.toBeNull();
    expect(reverse).not.toBeNull();
    expect(forward!).toBeCloseTo(reverse!, 10);
  });

  it("returns null for degenerate zero-length, sub-epsilon, and non-finite inputs", () => {
    expect(
      calculateAngle3D(
        { x: 0.5, y: 0.5, z: 0 },
        { x: 0.5, y: 0.5, z: 0 },
        { x: 1, y: 0.5, z: 0 },
      ),
    ).toBeNull();

    expect(
      calculateAngle3D(
        { x: 1, y: 0, z: 0 },
        { x: 0, y: 0, z: 0 },
        { x: 0, y: 0, z: 0 },
      ),
    ).toBeNull();

    expect(
      calculateAngle3D(
        { x: 0.2, y: 0.3, z: 0.4 },
        { x: 0.2, y: 0.3, z: 0.4 },
        { x: 0.2, y: 0.3, z: 0.4 },
      ),
    ).toBeNull();

    expect(
      calculateAngle3D(
        { x: 1e-9, y: 0, z: 0 },
        { x: 0, y: 0, z: 0 },
        { x: 1, y: 0, z: 0 },
      ),
    ).toBeNull();

    expect(
      calculateAngle3D(
        { x: Number.NaN, y: 0, z: 0 },
        { x: 0, y: 0, z: 0 },
        { x: 1, y: 0, z: 0 },
      ),
    ).toBeNull();

    expect(
      calculateAngle3D(
        { x: 1, y: 0, z: 0 },
        { x: 0, y: Number.NaN, z: 0 },
        { x: 0, y: 1, z: 0 },
      ),
    ).toBeNull();

    expect(
      calculateAngle3D(
        { x: 1, y: 0, z: Number.POSITIVE_INFINITY },
        { x: 0, y: 0, z: 0 },
        { x: 0, y: 1, z: 0 },
      ),
    ).toBeNull();

    expect(
      calculateAngle3D(
        { x: 1, y: 0, z: 0 },
        { x: 0, y: 0, z: 0 },
        { x: Number.NEGATIVE_INFINITY, y: 1, z: Number.NaN },
      ),
    ).toBeNull();
  });
});

describe("POSE_LANDMARK_NAMES, JOINT_DEFINITIONS & computeJointAngles", () => {
  it("exposes canonical 33 landmark names and 8 anatomical joint triplets", () => {
    expect(POSE_LANDMARK_NAMES).toHaveLength(33);
    expect(new Set(POSE_LANDMARK_NAMES).size).toBe(33);
    for (const name of POSE_LANDMARK_NAMES) {
      expect(name.length).toBeGreaterThan(0);
    }

    expect(POSE_LANDMARK_NAMES[0]).toBe("nose");
    expect(POSE_LANDMARK_NAMES[11]).toBe("left_shoulder");
    expect(POSE_LANDMARK_NAMES[12]).toBe("right_shoulder");
    expect(POSE_LANDMARK_NAMES[15]).toBe("left_wrist");
    expect(POSE_LANDMARK_NAMES[23]).toBe("left_hip");
    expect(POSE_LANDMARK_NAMES[28]).toBe("right_ankle");
    expect(POSE_LANDMARK_NAMES[32]).toBe("right_foot_index");

    expect(JOINT_DEFINITIONS).toHaveLength(8);
    const byId = Object.fromEntries(
      JOINT_DEFINITIONS.map((j) => [j.id, j.indices]),
    );
    expect(byId.left_elbow).toEqual([11, 13, 15]);
    expect(byId.right_elbow).toEqual([12, 14, 16]);
    expect(byId.left_shoulder).toEqual([23, 11, 13]);
    expect(byId.right_shoulder).toEqual([24, 12, 14]);
    expect(byId.left_hip).toEqual([11, 23, 25]);
    expect(byId.right_hip).toEqual([12, 24, 26]);
    expect(byId.left_knee).toEqual([23, 25, 27]);
    expect(byId.right_knee).toEqual([24, 26, 28]);
  });

  function createDistinctJointLandmarks(): Point3D[] {
    const lms = createNeutralLandmarks(0.9);
    // Left chain: 90, 180, 135, 45
    lms[11] = { x: 0, y: 0, z: 0, visibility: 0.9 };
    lms[13] = { x: 1, y: 0, z: 0, visibility: 0.9 };
    lms[15] = { x: 2, y: 0, z: 0, visibility: 0.9 };
    lms[23] = { x: 0, y: 1, z: 0, visibility: 0.9 };
    lms[25] = { x: 1, y: 2, z: 0, visibility: 0.9 };
    lms[27] = { x: 0, y: 2, z: 0, visibility: 0.9 };

    // Right chain: 45, 90, 180, 135
    lms[12] = { x: 0, y: 0, z: 0, visibility: 0.9 };
    lms[14] = { x: 1, y: 1, z: 0, visibility: 0.9 };
    lms[16] = { x: 0, y: 2, z: 0, visibility: 0.9 };
    lms[24] = { x: 0, y: 1, z: 0, visibility: 0.9 };
    lms[26] = { x: 0, y: 2, z: 0, visibility: 0.9 };
    lms[28] = { x: 1, y: 3, z: 0, visibility: 0.9 };
    return lms;
  }

  it("computes all 8 joints simultaneously with distinct expected angles and supports subsets", () => {
    const lms = createDistinctJointLandmarks();
    const all = computeJointAngles(lms, ALL_JOINTS);

    expect(all.left_shoulder).toBeCloseTo(90, 6);
    expect(all.left_elbow).toBeCloseTo(180, 6);
    expect(all.left_hip).toBeCloseTo(135, 6);
    expect(all.left_knee).toBeCloseTo(45, 6);
    expect(all.right_shoulder).toBeCloseTo(45, 6);
    expect(all.right_elbow).toBeCloseTo(90, 6);
    expect(all.right_hip).toBeCloseTo(180, 6);
    expect(all.right_knee).toBeCloseTo(135, 6);

    const subset = computeJointAngles(lms, ["left_elbow", "right_knee"]);
    expect(subset.left_elbow).toBeCloseTo(180, 6);
    expect(subset.right_knee).toBeCloseTo(135, 6);
    expect(subset.right_elbow).toBeNull();
    expect(subset.left_shoulder).toBeNull();
    expect(subset.right_shoulder).toBeNull();
    expect(subset.left_hip).toBeNull();
    expect(subset.right_hip).toBeNull();
    expect(subset.left_knee).toBeNull();
  });

  it("gates joints by minVisibility (default 0.4) across proximal, vertex, and distal landmarks", () => {
    const lms1 = createDistinctJointLandmarks();
    lms1[11].visibility = 0.39;
    const res1 = computeJointAngles(lms1, ALL_JOINTS);
    expect(res1.left_elbow).toBeNull();
    expect(res1.left_shoulder).toBeNull();
    expect(res1.left_hip).toBeNull();
    expect(res1.left_knee).toBeCloseTo(45, 6);
    expect(res1.right_elbow).toBeCloseTo(90, 6);

    const lms2 = createDistinctJointLandmarks();
    lms2[13].visibility = 0.39;
    const res2 = computeJointAngles(lms2, ALL_JOINTS);
    expect(res2.left_elbow).toBeNull();
    expect(res2.left_shoulder).toBeNull();
    expect(res2.left_hip).toBeCloseTo(135, 6);

    const lms3 = createDistinctJointLandmarks();
    lms3[15].visibility = 0.39;
    const res3 = computeJointAngles(lms3, ALL_JOINTS);
    expect(res3.left_elbow).toBeNull();
    expect(res3.left_shoulder).toBeCloseTo(90, 6);

    const lmsBoundary = createDistinctJointLandmarks();
    lmsBoundary[15].visibility = 0.4;
    lmsBoundary[16].visibility = undefined;
    lmsBoundary[28].visibility = Number.NaN;
    const resBoundary = computeJointAngles(lmsBoundary, ALL_JOINTS);
    expect(resBoundary.left_elbow).toBeCloseTo(180, 6);
    expect(resBoundary.right_elbow).toBeNull();
    expect(resBoundary.right_knee).toBeNull();

    const lmsCustom = createDistinctJointLandmarks();
    lmsCustom[15].visibility = 0.6;
    expect(
      computeJointAngles(lmsCustom, ["left_elbow"], 0.7).left_elbow,
    ).toBeNull();
  });

  it("handles null and short landmark arrays safely", () => {
    const nullResult = computeJointAngles(null, ALL_JOINTS);
    const emptyResult = computeJointAngles([], ALL_JOINTS);
    for (const joint of ALL_JOINTS) {
      expect(nullResult[joint]).toBeNull();
      expect(emptyResult[joint]).toBeNull();
    }
  });
});

describe("PoseSmoother", () => {
  it("preserves raw coordinates bit-for-bit when disabled even with prior filter state", () => {
    const smoother = new PoseSmoother();
    const enabledCfg: SmoothingConfig = { enabled: true, strength: 0.9 };
    for (let i = 0; i < 5; i++) {
      const frame = createNeutralLandmarks(0.8);
      frame[0] = { x: 0.2 + i * 0.01, y: 0.3, z: -0.1, visibility: 0.8 };
      smoother.filter(frame, i * 33, enabledCfg);
    }

    const raw = createNeutralLandmarks(0.85);
    raw[0] = {
      x: 0.1234567890123456,
      y: -0.9876543210987654,
      z: 0.000123456789,
      visibility: 0.73,
    };

    const out = smoother.filter(raw, 200, { enabled: false, strength: 0.9 });
    expect(out).toEqual(raw);
    expect(out[0]).not.toBe(raw[0]);
    expect(out[0].x).toBe(raw[0].x);
    expect(out[0].y).toBe(raw[0].y);
    expect(out[0].z).toBe(raw[0].z);
    expect(out[0].visibility).toBe(raw[0].visibility);
  });

  it("measurably reduces variance and RMSE on constant + deterministic jitter streams", () => {
    const smoother = new PoseSmoother();
    const cfg: SmoothingConfig = { enabled: true, strength: 0.6 };
    const dt = 1000 / 30;
    const rawX: number[] = [];
    const rawY: number[] = [];
    const rawZ: number[] = [];
    const smoothX: number[] = [];
    const smoothY: number[] = [];
    const smoothZ: number[] = [];

    for (let i = 0; i < 60; i++) {
      const jx = 0.04 * (i % 2 === 0 ? 1 : -1) + 0.015 * Math.sin(i * 2.3);
      const jy = 0.03 * (i % 2 === 0 ? -1 : 1) + 0.012 * Math.cos(i * 1.7);
      const jz = 0.025 * Math.sin(i * 2.7);
      const frame = createNeutralLandmarks(0.95);
      frame[0] = {
        x: 0.5 + jx,
        y: 0.4 + jy,
        z: -0.1 + jz,
        visibility: 0.95,
      };
      const out = smoother.filter(frame, i * dt, cfg);

      if (i >= 5) {
        rawX.push(frame[0].x);
        rawY.push(frame[0].y);
        rawZ.push(frame[0].z);
        smoothX.push(out[0].x);
        smoothY.push(out[0].y);
        smoothZ.push(out[0].z);
      }
    }

    const truthX = rawX.map(() => 0.5);
    expect(variance(smoothX)).toBeLessThan(variance(rawX) * 0.5);
    expect(variance(smoothY)).toBeLessThan(variance(rawY) * 0.5);
    expect(variance(smoothZ)).toBeLessThan(variance(rawZ) * 0.5);
    expect(rmse(smoothX, truthX)).toBeLessThan(rmse(rawX, truthX));
  });

  it("measurably reduces RMSE and frame-to-frame jitter on a moving sine wave", () => {
    const smoother = new PoseSmoother();
    const cfg: SmoothingConfig = { enabled: true, strength: 0.5 };
    const dt = 1000 / 30;
    const truthX: number[] = [];
    const rawX: number[] = [];
    const smoothX: number[] = [];

    for (let i = 0; i < 90; i++) {
      const tMs = i * dt;
      const tSec = tMs / 1000;
      const tx = 0.5 + 0.2 * Math.sin(2 * Math.PI * 0.5 * tSec);
      const rx =
        tx + 0.03 * ((i % 2 === 0 ? 1 : -1) * 0.7 + Math.cos(i * 1.9) * 0.3);
      const frame = createNeutralLandmarks(0.9);
      frame[0] = { x: rx, y: 0.5, z: 0, visibility: 0.9 };
      const out = smoother.filter(frame, tMs, cfg);

      if (i >= 10) {
        truthX.push(tx);
        rawX.push(rx);
        smoothX.push(out[0].x);
      }
    }

    expect(rmse(smoothX, truthX)).toBeLessThan(rmse(rawX, truthX));
    expect(variance(diffs(smoothX))).toBeLessThan(variance(diffs(rawX)) * 0.5);
  });

  it("applies stronger smoothing at higher strength and passes visibility through raw", () => {
    const lowSmoother = new PoseSmoother();
    const highSmoother = new PoseSmoother();
    const dt = 1000 / 30;
    const rawX: number[] = [];
    const lowX: number[] = [];
    const highX: number[] = [];

    for (let i = 0; i < 60; i++) {
      const jx = 0.04 * (i % 2 === 0 ? 1 : -1) + 0.015 * Math.sin(i * 2.3);
      const frame = createNeutralLandmarks(0.67);
      frame[0] = { x: 0.5 + jx, y: 0.5, z: 0, visibility: 0.67 };
      const outLow = lowSmoother.filter(frame, i * dt, {
        enabled: true,
        strength: 0.1,
      });
      const outHigh = highSmoother.filter(frame, i * dt, {
        enabled: true,
        strength: 0.9,
      });

      expect(outLow[0].visibility).toBe(0.67);
      expect(outHigh[0].visibility).toBe(0.67);

      if (i >= 5) {
        rawX.push(frame[0].x);
        lowX.push(outLow[0].x);
        highX.push(outHigh[0].x);
      }
    }

    expect(variance(lowX)).toBeLessThan(variance(rawX));
    expect(variance(highX)).toBeLessThan(variance(lowX));
  });

  it("clears prior filter history on reset() and handles non-positive or large dt safely", () => {
    const smoother = new PoseSmoother();
    const cfg: SmoothingConfig = { enabled: true, strength: 0.8 };

    for (let i = 0; i < 10; i++) {
      const frame = createNeutralLandmarks(0.9);
      frame[0] = { x: 0.1, y: 0.1, z: 0, visibility: 0.9 };
      smoother.filter(frame, i * 33.3, cfg);
    }

    const stepFrame = createNeutralLandmarks(0.9);
    stepFrame[0] = { x: 0.9, y: 0.9, z: 0, visibility: 0.9 };
    const damped = smoother.filter(stepFrame, 333, cfg);
    expect(damped[0].x).toBeLessThan(0.85);

    smoother.reset();
    const afterReset = smoother.filter(stepFrame, 366, cfg);
    expect(afterReset[0].x).toBeCloseTo(0.9, 10);

    const dup = smoother.filter(stepFrame, 366, cfg);
    expect(Number.isFinite(dup[0].x)).toBe(true);

    const largeGap = smoother.filter(stepFrame, 5000, cfg);
    expect(Number.isFinite(largeGap[0].x)).toBe(true);
    expect(largeGap[0].x).toBeCloseTo(0.9, 10);
  });
});

describe("buildPoseSnapshot & serializePoseSnapshot", () => {
  it("produces a valid 33-landmark snapshot and round-trips through JSON.parse", () => {
    const lms = createNeutralLandmarks(0.92);
    lms[3].visibility = undefined;

    const params = {
      landmarks: lms,
      selectedJoints: ["left_elbow", "right_knee"] as JointId[],
      angles: { left_elbow: 142.5, right_knee: null },
      model: "full" as const,
      mirrored: true,
      smoothing: { enabled: true, strength: 0.6 },
      timestampMs: 1700000000000,
    };

    const snapshot = buildPoseSnapshot(params);
    expect(snapshot.schemaVersion).toBe("1.0.0");
    expect(snapshot.timestampMs).toBe(1700000000000);
    expect(snapshot.timestamp).toBe("2023-11-14T22:13:20.000Z");
    expect(snapshot.poseDetected).toBe(true);
    expect(snapshot.model).toBe("full");
    expect(snapshot.mirrored).toBe(true);
    expect(snapshot.smoothing).toEqual({ enabled: true, strength: 0.6 });

    expect(snapshot.landmarks).toHaveLength(33);
    snapshot.landmarks.forEach((lm, idx) => {
      expect(lm.index).toBe(idx);
      expect(lm.name).toBe(POSE_LANDMARK_NAMES[idx]);
      expect(typeof lm.x).toBe("number");
      expect(typeof lm.y).toBe("number");
      expect(typeof lm.z).toBe("number");
      expect(typeof lm.visibility).toBe("number");
      expect(Number.isFinite(lm.x)).toBe(true);
      expect(Number.isFinite(lm.y)).toBe(true);
      expect(Number.isFinite(lm.z)).toBe(true);
      expect(Number.isFinite(lm.visibility)).toBe(true);
      expect(lm.visibility).toBeGreaterThanOrEqual(0);
      expect(lm.visibility).toBeLessThanOrEqual(1);
    });
    expect(snapshot.landmarks[3].visibility).toBe(0);

    expect(snapshot.activeJointAngles).toHaveLength(2);
    expect(snapshot.activeJointAngles[0]).toEqual({
      joint: "left_elbow",
      label: "Left Elbow",
      degrees: 142.5,
    });
    expect(snapshot.activeJointAngles[1]).toEqual({
      joint: "right_knee",
      label: "Right Knee",
      degrees: null,
    });

    const json = serializePoseSnapshot(params);
    const parsed = JSON.parse(json) as PoseSnapshot;
    expect(parsed).toEqual(snapshot);
    expect(parsed.landmarks).toHaveLength(33);
  });

  it("emits 33 zeroed landmarks with poseDetected=false when landmarks is null", () => {
    const params = {
      landmarks: null,
      selectedJoints: ["left_elbow", "right_knee"] as JointId[],
      angles: { left_elbow: null, right_knee: null },
      model: "lite" as const,
      mirrored: false,
      smoothing: { enabled: false, strength: 0.5 },
      timestampMs: 1700000000000,
    };

    const snapshot = buildPoseSnapshot(params);
    expect(snapshot.poseDetected).toBe(false);
    expect(snapshot.landmarks).toHaveLength(33);
    snapshot.landmarks.forEach((lm, idx) => {
      expect(lm).toEqual({
        index: idx,
        name: POSE_LANDMARK_NAMES[idx],
        x: 0,
        y: 0,
        z: 0,
        visibility: 0,
      });
    });
    expect(snapshot.activeJointAngles).toEqual([
      { joint: "left_elbow", label: "Left Elbow", degrees: null },
      { joint: "right_knee", label: "Right Knee", degrees: null },
    ]);

    const parsed = JSON.parse(serializePoseSnapshot(params)) as PoseSnapshot;
    expect(parsed.poseDetected).toBe(false);
    expect(parsed.landmarks).toHaveLength(33);
  });

  it("sanitizes non-finite coordinates and incomplete landmark arrays into valid JSON", () => {
    const partial: Point3D[] = [
      {
        x: Number.NaN,
        y: Number.POSITIVE_INFINITY,
        z: Number.NEGATIVE_INFINITY,
        visibility: Number.NaN,
      },
    ];

    const json = serializePoseSnapshot({
      landmarks: partial,
      selectedJoints: ["left_elbow"],
      model: "heavy",
      mirrored: false,
      smoothing: { enabled: true, strength: Number.NaN },
      timestampMs: Number.NaN,
    });

    const parsed = JSON.parse(json) as PoseSnapshot;
    expect(parsed.poseDetected).toBe(false);
    expect(parsed.landmarks).toHaveLength(33);
    expect(parsed.smoothing.strength).toBe(0.5);
    for (const lm of parsed.landmarks) {
      expect(Number.isFinite(lm.x)).toBe(true);
      expect(Number.isFinite(lm.y)).toBe(true);
      expect(Number.isFinite(lm.z)).toBe(true);
      expect(Number.isFinite(lm.visibility)).toBe(true);
    }
  });
});

describe("V2 Hand, Finger & Face/Head Biometrics", () => {
  it("computes 5-finger articulation angles, extended finger count, and pinch state", () => {
    const landmarks: Point3D[] = Array.from({ length: 21 }, (_, idx) => ({
      x: 0.5,
      y: 0.8 - idx * 0.02,
      z: 0,
    }));
    // Make all 5 finger chains straight lines (180 degrees -> extended)
    const chains: [number, number, number, number][] = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
      [17, 18, 19, 20],
    ];
    chains.forEach(([a, b, c, d], fIdx) => {
      const baseX = 0.3 + fIdx * 0.08;
      landmarks[a] = { x: baseX, y: 0.7, z: 0 };
      landmarks[b] = { x: baseX, y: 0.6, z: 0 };
      landmarks[c] = { x: baseX, y: 0.5, z: 0 };
      landmarks[d] = { x: baseX, y: 0.4, z: 0 };
    });

    const hand: TrackedHand = {
      handedness: "Left",
      score: 0.98,
      landmarks,
    };

    const metrics = computeHandMetrics(hand);
    expect(metrics.handedness).toBe("Left");
    expect(metrics.extendedFingers).toBe(5);
    expect(metrics.fingerAngles.thumb).toBeCloseTo(180, 1);
    expect(metrics.fingerAngles.index).toBeCloseTo(180, 1);
    expect(metrics.opennessPercent).toBeGreaterThanOrEqual(90);
    expect(metrics.isPinching).toBe(false);

    // Move thumb tip (4) right onto index tip (8) to trigger pinch
    landmarks[4] = { ...landmarks[8]! };
    const pinched = computeHandMetrics(hand);
    expect(pinched.isPinching).toBe(true);
    expect(pinched.pinchDistance).toBe(0);
  });

  it("computes 3D head orientation (Yaw, Pitch, Roll) and eye/mouth openness from 478 face landmarks", () => {
    expect(computeFaceHeadMetrics(null)).toBeNull();
    expect(computeFaceHeadMetrics([{ x: 0, y: 0, z: 0 }])).toBeNull();

    const face: Point3D[] = Array.from({ length: 478 }, () => ({
      x: 0.5,
      y: 0.5,
      z: 0,
    }));
    face[1] = { x: 0.5, y: 0.52, z: -0.05 }; // noseTip
    face[10] = { x: 0.5, y: 0.3, z: 0 }; // forehead
    face[152] = { x: 0.5, y: 0.7, z: 0 }; // chin
    face[234] = { x: 0.35, y: 0.5, z: 0 }; // leftCheek
    face[454] = { x: 0.65, y: 0.5, z: 0 }; // rightCheek
    face[33] = { x: 0.4, y: 0.45, z: 0 }; // rightEyeOuter
    face[133] = { x: 0.46, y: 0.45, z: 0 }; // rightEyeInner
    face[159] = { x: 0.43, y: 0.435, z: 0 }; // rightEyeTop
    face[145] = { x: 0.43, y: 0.465, z: 0 }; // rightEyeBottom
    face[263] = { x: 0.6, y: 0.45, z: 0 }; // leftEyeOuter
    face[362] = { x: 0.54, y: 0.45, z: 0 }; // leftEyeInner
    face[386] = { x: 0.57, y: 0.435, z: 0 }; // leftEyeTop
    face[374] = { x: 0.57, y: 0.465, z: 0 }; // leftEyeBottom
    face[13] = { x: 0.5, y: 0.6, z: 0 }; // upperLip
    face[14] = { x: 0.5, y: 0.62, z: 0 }; // lowerLip
    face[61] = { x: 0.45, y: 0.61, z: 0 }; // mouthLeft
    face[291] = { x: 0.55, y: 0.61, z: 0 }; // mouthRight

    const head = computeFaceHeadMetrics(face);
    expect(head).not.toBeNull();
    expect(head?.headRollDeg).toBe(0);
    expect(head?.headYawDeg).toBe(0);
    expect(head?.leftEyeOpenness).toBeGreaterThan(50);
    expect(head?.rightEyeOpenness).toBeGreaterThan(50);
  });

  it("tracks 2 simultaneous people (P1 & P2) with temporal ID stability and binds 4 hands + 2 faces spatially", () => {
    const tracker = new MultiPersonTracker();
    const makePoseAt = (centerX: number): Point3D[] =>
      Array.from({ length: 33 }, (_, idx) => ({
        x: centerX + (idx % 2 === 0 ? -0.02 : 0.02),
        y: idx === 0 ? 0.25 : 0.5,
        z: 0,
        visibility: 0.95,
      }));
    const makeFaceAt = (centerX: number): Point3D[] =>
      Array.from({ length: 478 }, () => ({
        x: centerX,
        y: 0.25,
        z: 0,
      }));
    const makeHandAt = (
      centerX: number,
      handedness: "Left" | "Right",
    ): TrackedHand => ({
      handedness,
      score: 0.95,
      landmarks: Array.from({ length: 21 }, () => ({
        x: centerX,
        y: 0.5,
        z: 0,
      })),
    });

    const poseLeft = makePoseAt(0.25);
    const poseRight = makePoseAt(0.75);
    const faceLeft = makeFaceAt(0.25);
    const faceRight = makeFaceAt(0.75);
    const hands: TrackedHand[] = [
      makeHandAt(0.23, "Left"),
      makeHandAt(0.27, "Right"),
      makeHandAt(0.73, "Left"),
      makeHandAt(0.77, "Right"),
    ];

    // Frame 1: [poseRight, poseLeft] -> sorted left-to-right on initial acquisition
    const frame1 = tracker.update({
      poses: [poseRight, poseLeft],
      hands,
      faces: [faceRight, faceLeft],
      selectedJoints: ["left_elbow", "right_elbow"],
      timestampMs: 1000,
      smoothing: { enabled: true, strength: 0.5 },
    });

    expect(frame1).toHaveLength(2);
    expect(frame1[0]?.personId).toBe(0);
    expect(frame1[0]?.label).toBe("P1");
    expect(frame1[0]?.centroid.x).toBeLessThan(0.4);
    expect(frame1[0]?.hands).toHaveLength(2);
    expect(frame1[0]?.faceLandmarks).toHaveLength(478);

    expect(frame1[1]?.personId).toBe(1);
    expect(frame1[1]?.label).toBe("P2");
    expect(frame1[1]?.centroid.x).toBeGreaterThan(0.6);
    expect(frame1[1]?.hands).toHaveLength(2);
    expect(frame1[1]?.faceLandmarks).toHaveLength(478);

    // Frame 2: swap input order [poseLeft, poseRight] -> temporal matching preserves P1 (left) and P2 (right)
    const frame2 = tracker.update({
      poses: [poseLeft, poseRight],
      hands,
      faces: [faceLeft, faceRight],
      selectedJoints: ["left_elbow", "right_elbow"],
      timestampMs: 1033,
      smoothing: { enabled: true, strength: 0.5 },
    });

    expect(frame2[0]?.personId).toBe(0);
    expect(frame2[0]?.centroid.x).toBeLessThan(0.4);
    expect(frame2[1]?.personId).toBe(1);
    expect(frame2[1]?.centroid.x).toBeGreaterThan(0.6);
  });
});
