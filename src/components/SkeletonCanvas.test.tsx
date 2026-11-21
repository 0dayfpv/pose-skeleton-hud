import "@testing-library/jest-dom/vitest";
import { createRef } from "react";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AppStatus, JointId, Point3D, ViewMode } from "../engine/geometry";
import SkeletonCanvas, {
  drawSkeletonFrame,
  type DrawSkeletonFrameParams,
  type SkeletonCanvasProps,
} from "./SkeletonCanvas";

interface CanvasStateSnapshot {
  globalAlpha: number;
  fillStyle: string;
  strokeStyle: string;
  lineWidth: number;
  shadowBlur: number;
  shadowColor: string;
  font: string;
  textAlign: CanvasTextAlign;
  textBaseline: CanvasTextBaseline;
  scaleX: number;
  scaleY: number;
}

interface ClearRectCall {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface FillRectCall {
  x: number;
  y: number;
  w: number;
  h: number;
  fillStyle: string;
  globalAlpha: number;
}

interface StrokeCall {
  points: Array<{ x: number; y: number }>;
  strokeStyle: string;
  lineWidth: number;
  globalAlpha: number;
  shadowBlur: number;
}

interface ArcCall {
  x: number;
  y: number;
  radius: number;
  fillStyle: string;
  strokeStyle: string;
  globalAlpha: number;
  shadowBlur: number;
}

interface FillTextCall {
  text: string;
  x: number;
  y: number;
  fillStyle: string;
  globalAlpha: number;
  scaleX: number;
  scaleY: number;
}

interface MockCanvasRecords {
  clearRectCalls: ClearRectCall[];
  fillRectCalls: FillRectCall[];
  strokeCalls: StrokeCall[];
  arcCalls: ArcCall[];
  fillTextCalls: FillTextCall[];
  setTransformCalls: Array<[number, number, number, number, number, number]>;
  saveCount: number;
  restoreCount: number;
}

function createMockCanvasContext(): {
  ctx: CanvasRenderingContext2D;
  records: MockCanvasRecords;
} {
  let state: CanvasStateSnapshot = {
    globalAlpha: 1,
    fillStyle: "#000000",
    strokeStyle: "#000000",
    lineWidth: 1,
    shadowBlur: 0,
    shadowColor: "rgba(0, 0, 0, 0)",
    font: "10px sans-serif",
    textAlign: "start",
    textBaseline: "alphabetic",
    scaleX: 1,
    scaleY: 1,
  };
  const stack: CanvasStateSnapshot[] = [];
  let currentPath: Array<{ x: number; y: number }> = [];
  let pendingArc: { x: number; y: number; radius: number } | null = null;

  const records: MockCanvasRecords = {
    clearRectCalls: [],
    fillRectCalls: [],
    strokeCalls: [],
    arcCalls: [],
    fillTextCalls: [],
    setTransformCalls: [],
    saveCount: 0,
    restoreCount: 0,
  };

  const mock = {
    get globalAlpha() {
      return state.globalAlpha;
    },
    set globalAlpha(v: number) {
      state.globalAlpha = v;
    },
    get fillStyle() {
      return state.fillStyle;
    },
    set fillStyle(v: string) {
      state.fillStyle = v;
    },
    get strokeStyle() {
      return state.strokeStyle;
    },
    set strokeStyle(v: string) {
      state.strokeStyle = v;
    },
    get lineWidth() {
      return state.lineWidth;
    },
    set lineWidth(v: number) {
      state.lineWidth = v;
    },
    get shadowBlur() {
      return state.shadowBlur;
    },
    set shadowBlur(v: number) {
      state.shadowBlur = v;
    },
    get shadowColor() {
      return state.shadowColor;
    },
    set shadowColor(v: string) {
      state.shadowColor = v;
    },
    get font() {
      return state.font;
    },
    set font(v: string) {
      state.font = v;
    },
    get textAlign() {
      return state.textAlign;
    },
    set textAlign(v: CanvasTextAlign) {
      state.textAlign = v;
    },
    get textBaseline() {
      return state.textBaseline;
    },
    set textBaseline(v: CanvasTextBaseline) {
      state.textBaseline = v;
    },
    lineCap: "butt" as CanvasLineCap,
    lineJoin: "miter" as CanvasLineJoin,
    save() {
      records.saveCount += 1;
      stack.push({ ...state });
    },
    restore() {
      records.restoreCount += 1;
      const prev = stack.pop();
      if (prev) {
        state = prev;
      }
    },
    scale(sx: number, sy: number) {
      state.scaleX *= sx;
      state.scaleY *= sy;
    },
    translate() {},
    setTransform(
      a: number,
      b: number,
      c: number,
      d: number,
      e: number,
      f: number,
    ) {
      records.setTransformCalls.push([a, b, c, d, e, f]);
      state.scaleX = a;
      state.scaleY = d;
    },
    clearRect(x: number, y: number, w: number, h: number) {
      records.clearRectCalls.push({ x, y, w, h });
    },
    fillRect(x: number, y: number, w: number, h: number) {
      records.fillRectCalls.push({
        x,
        y,
        w,
        h,
        fillStyle: state.fillStyle,
        globalAlpha: state.globalAlpha,
      });
    },
    strokeRect() {},
    beginPath() {
      currentPath = [];
      pendingArc = null;
    },
    closePath() {},
    moveTo(x: number, y: number) {
      currentPath.push({ x, y });
    },
    lineTo(x: number, y: number) {
      currentPath.push({ x, y });
    },
    quadraticCurveTo(_cpx: number, _cpy: number, x: number, y: number) {
      currentPath.push({ x, y });
    },
    arc(x: number, y: number, radius: number) {
      pendingArc = { x, y, radius };
    },
    roundRect(x: number, y: number, w: number, h: number) {
      currentPath.push({ x, y }, { x: x + w, y: y + h });
    },
    fill() {
      if (pendingArc) {
        records.arcCalls.push({
          x: pendingArc.x,
          y: pendingArc.y,
          radius: pendingArc.radius,
          fillStyle: state.fillStyle,
          strokeStyle: state.strokeStyle,
          globalAlpha: state.globalAlpha,
          shadowBlur: state.shadowBlur,
        });
        pendingArc = null;
      } else if (currentPath.length >= 2) {
        const p0 = currentPath[0];
        const p1 = currentPath[1];
        records.fillRectCalls.push({
          x: p0.x,
          y: p0.y,
          w: p1.x - p0.x,
          h: p1.y - p0.y,
          fillStyle: state.fillStyle,
          globalAlpha: state.globalAlpha,
        });
      }
    },
    stroke() {
      if (currentPath.length > 0) {
        records.strokeCalls.push({
          points: [...currentPath],
          strokeStyle: state.strokeStyle,
          lineWidth: state.lineWidth,
          globalAlpha: state.globalAlpha,
          shadowBlur: state.shadowBlur,
        });
      }
    },
    fillText(text: string, x: number, y: number) {
      records.fillTextCalls.push({
        text,
        x,
        y,
        fillStyle: state.fillStyle,
        globalAlpha: state.globalAlpha,
        scaleX: state.scaleX,
        scaleY: state.scaleY,
      });
    },
    measureText(text: string) {
      return { width: text.length * 7 };
    },
    setLineDash() {},
  };

  return { ctx: mock as unknown as CanvasRenderingContext2D, records };
}

function createMockLandmarks(
  overrides: Record<number, Partial<Point3D>> = {},
): Point3D[] {
  const result: Point3D[] = Array.from({ length: 33 }, (_, i) => {
    const col = i % 6;
    const row = Math.floor(i / 6);
    const base: Point3D = {
      x: 0.25 + col * 0.08,
      y: 0.15 + row * 0.12,
      z: 0,
      visibility: 0.95,
    };
    const patch = overrides[i];
    return patch ? { ...base, ...patch } : base;
  });
  return result;
}

function parseHexToRgb(color: string): { r: number; g: number; b: number } | null {
  const normalized = color.trim().toLowerCase();
  const hexMatch = /^#([0-9a-f]{6})$/.exec(normalized);
  if (!hexMatch) return null;
  const hex = hexMatch[1];
  return {
    r: parseInt(hex.slice(0, 2), 16),
    g: parseInt(hex.slice(2, 4), 16),
    b: parseInt(hex.slice(4, 6), 16),
  };
}

function isCyanColor(color: string): boolean {
  const rgb = parseHexToRgb(color);
  if (!rgb) return false;
  return rgb.b > rgb.r + 50 && rgb.g > rgb.r + 50;
}

function isRightWarmOrMagentaColor(color: string): boolean {
  const rgb = parseHexToRgb(color);
  if (!rgb) return false;
  return rgb.r > rgb.g + 40;
}

function isWhiteOrGrayColor(color: string): boolean {
  const rgb = parseHexToRgb(color);
  if (!rgb) return false;
  return (
    Math.abs(rgb.r - rgb.g) < 35 &&
    Math.abs(rgb.g - rgb.b) < 45 &&
    rgb.r > 120
  );
}

function findJointArcNear(
  records: MockCanvasRecords,
  targetX: number,
  targetY: number,
  tolerance = 3,
): ArcCall | undefined {
  const matches = records.arcCalls.filter(
    (a) =>
      a.radius < 20 &&
      a.fillStyle.toLowerCase() !== "#f8fafc" &&
      Math.hypot(a.x - targetX, a.y - targetY) <= tolerance,
  );
  return matches[0];
}

function findBoneStroke(
  records: MockCanvasRecords,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  tolerance = 3,
): StrokeCall | undefined {
  return records.strokeCalls.find((s) => {
    if (s.points.length !== 2) return false;
    const [p0, p1] = s.points;
    const forward =
      Math.hypot(p0.x - x1, p0.y - y1) <= tolerance &&
      Math.hypot(p1.x - x2, p1.y - y2) <= tolerance;
    const reverse =
      Math.hypot(p0.x - x2, p0.y - y2) <= tolerance &&
      Math.hypot(p1.x - x1, p1.y - y1) <= tolerance;
    return forward || reverse;
  });
}

class MockResizeObserver {
  static instances: MockResizeObserver[] = [];
  private readonly callback: ResizeObserverCallback;
  observedTargets: Element[] = [];
  observeSpy = vi.fn((target: Element) => {
    this.observedTargets.push(target);
  });
  unobserveSpy = vi.fn();
  disconnectSpy = vi.fn();

  constructor(callback: ResizeObserverCallback) {
    this.callback = callback;
    MockResizeObserver.instances.push(this);
  }

  observe(target: Element): void {
    this.observeSpy(target);
  }

  unobserve(target: Element): void {
    this.unobserveSpy(target);
  }

  disconnect(): void {
    this.disconnectSpy();
  }

  triggerResize(width: number, height: number): void {
    const entries = this.observedTargets.map(
      (target) =>
        ({
          target,
          contentRect: {
            width,
            height,
            top: 0,
            left: 0,
            right: width,
            bottom: height,
            x: 0,
            y: 0,
            toJSON: () => ({}),
          } as DOMRectReadOnly,
          borderBoxSize: [],
          contentBoxSize: [],
          devicePixelContentBoxSize: [],
        }) as ResizeObserverEntry,
    );
    this.callback(entries, this as unknown as ResizeObserver);
  }
}

function buildDefaultProps(
  overrides: Partial<SkeletonCanvasProps> = {},
): SkeletonCanvasProps {
  return {
    videoRef: createRef<HTMLVideoElement | null>(),
    canvasRef: createRef<HTMLCanvasElement | null>(),
    landmarks: null,
    angles: {},
    selectedJoints: [],
    viewMode: "overlay" as ViewMode,
    mirrored: false,
    status: "ready" as AppStatus,
    ...overrides,
  };
}

const originalDprDescriptor = Object.getOwnPropertyDescriptor(
  window,
  "devicePixelRatio",
);

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  MockResizeObserver.instances = [];
  if (originalDprDescriptor) {
    Object.defineProperty(window, "devicePixelRatio", originalDprDescriptor);
  }
});

describe("drawSkeletonFrame", () => {
  it("handles null landmarks cleanly in overlay and blackout modes", () => {
    const overlayMock = createMockCanvasContext();
    drawSkeletonFrame(overlayMock.ctx, 800, 600, {
      landmarks: null,
      angles: {},
      selectedJoints: [],
      viewMode: "overlay",
      mirrored: false,
    });

    expect(overlayMock.records.clearRectCalls).toEqual([
      { x: 0, y: 0, w: 800, h: 600 },
    ]);
    expect(overlayMock.records.fillRectCalls).toHaveLength(0);
    expect(overlayMock.records.fillTextCalls).toHaveLength(0);
    expect(overlayMock.records.saveCount).toBe(
      overlayMock.records.restoreCount,
    );

    const blackoutMock = createMockCanvasContext();
    drawSkeletonFrame(blackoutMock.ctx, 800, 600, {
      landmarks: null,
      angles: {},
      selectedJoints: [],
      viewMode: "blackout",
      mirrored: false,
    });

    const backdrop = blackoutMock.records.fillRectCalls.find(
      (c) =>
        c.x === 0 &&
        c.y === 0 &&
        c.w === 800 &&
        c.h === 600 &&
        c.fillStyle.toLowerCase() === "#05080f",
    );
    expect(backdrop).toBeDefined();
    expect(blackoutMock.records.strokeCalls.length).toBeGreaterThanOrEqual(3);
    expect(blackoutMock.records.fillTextCalls).toHaveLength(0);
    expect(blackoutMock.records.saveCount).toBe(
      blackoutMock.records.restoreCount,
    );
  });

  it("clears canvas in overlay mode without drawing an opaque backdrop", () => {
    const { ctx, records } = createMockCanvasContext();
    const landmarks = createMockLandmarks();

    drawSkeletonFrame(ctx, 800, 600, {
      landmarks,
      angles: {},
      selectedJoints: [],
      viewMode: "overlay",
      mirrored: false,
    });

    expect(records.clearRectCalls).toEqual([{ x: 0, y: 0, w: 800, h: 600 }]);
    const opaqueBg = records.fillRectCalls.find(
      (c) => c.w === 800 && c.h === 600 && c.fillStyle.toLowerCase() === "#05080f",
    );
    expect(opaqueBg).toBeUndefined();
    expect(records.strokeCalls.length).toBeGreaterThanOrEqual(35);
    expect(records.arcCalls.length).toBeGreaterThanOrEqual(33);
  });

  it("draws blackout grid and scales joint/bone depth from z-coordinate", () => {
    const { ctx, records } = createMockCanvasContext();
    const landmarks = createMockLandmarks({
      11: { x: 0.25, y: 0.3, z: -0.5, visibility: 0.95 },
      13: { x: 0.2, y: 0.55, z: -0.5, visibility: 0.95 },
      12: { x: 0.75, y: 0.3, z: 0.5, visibility: 0.95 },
      14: { x: 0.8, y: 0.55, z: 0.5, visibility: 0.95 },
    });

    drawSkeletonFrame(ctx, 800, 600, {
      landmarks,
      angles: {},
      selectedJoints: [],
      viewMode: "blackout",
      mirrored: false,
    });

    const backdrop = records.fillRectCalls.find(
      (c) =>
        c.x === 0 &&
        c.y === 0 &&
        c.w === 800 &&
        c.h === 600 &&
        c.fillStyle.toLowerCase() === "#05080f",
    );
    expect(backdrop).toBeDefined();
    expect(records.strokeCalls.length).toBeGreaterThanOrEqual(38);

    const closeJoint = findJointArcNear(records, 0.25 * 800, 0.3 * 600);
    const farJoint = findJointArcNear(records, 0.75 * 800, 0.3 * 600);
    expect(closeJoint).toBeDefined();
    expect(farJoint).toBeDefined();
    expect(closeJoint!.radius).toBeGreaterThan(farJoint!.radius);
    expect(closeJoint!.shadowBlur).toBeGreaterThan(farJoint!.shadowBlur);
    expect(closeJoint!.globalAlpha).toBeGreaterThan(farJoint!.globalAlpha);

    const closeBone = findBoneStroke(
      records,
      0.25 * 800,
      0.3 * 600,
      0.2 * 800,
      0.55 * 600,
    );
    const farBone = findBoneStroke(
      records,
      0.75 * 800,
      0.3 * 600,
      0.8 * 800,
      0.55 * 600,
    );
    expect(closeBone).toBeDefined();
    expect(farBone).toBeDefined();
    expect(closeBone!.lineWidth).toBeGreaterThan(farBone!.lineWidth);
    expect(closeBone!.shadowBlur).toBeGreaterThan(farBone!.shadowBlur);
  });

  it("applies smart anatomical coloring for left, right, face parity, and torso", () => {
    const { ctx, records } = createMockCanvasContext();
    const landmarks = createMockLandmarks({
      0: { x: 0.5, y: 0.12, z: 0, visibility: 0.95 },
      2: { x: 0.54, y: 0.1, z: 0, visibility: 0.95 },
      5: { x: 0.46, y: 0.1, z: 0, visibility: 0.95 },
      11: { x: 0.62, y: 0.28, z: 0, visibility: 0.95 },
      12: { x: 0.38, y: 0.28, z: 0, visibility: 0.95 },
      13: { x: 0.68, y: 0.45, z: 0, visibility: 0.95 },
      14: { x: 0.32, y: 0.45, z: 0, visibility: 0.95 },
      23: { x: 0.58, y: 0.58, z: 0, visibility: 0.95 },
      24: { x: 0.42, y: 0.58, z: 0, visibility: 0.95 },
    });

    drawSkeletonFrame(ctx, 1000, 1000, {
      landmarks,
      angles: {},
      selectedJoints: [],
      viewMode: "overlay",
      mirrored: false,
    });

    const noseArc = findJointArcNear(records, 500, 120);
    const leftEyeArc = findJointArcNear(records, 540, 100);
    const rightEyeArc = findJointArcNear(records, 460, 100);
    const leftElbowArc = findJointArcNear(records, 680, 450);
    const rightElbowArc = findJointArcNear(records, 320, 450);

    expect(leftEyeArc).toBeDefined();
    expect(leftElbowArc).toBeDefined();
    expect(rightEyeArc).toBeDefined();
    expect(rightElbowArc).toBeDefined();
    expect(noseArc).toBeDefined();

    expect(isCyanColor(leftEyeArc!.fillStyle)).toBe(true);
    expect(isCyanColor(leftElbowArc!.fillStyle)).toBe(true);
    expect(isRightWarmOrMagentaColor(rightEyeArc!.fillStyle)).toBe(true);
    expect(isRightWarmOrMagentaColor(rightElbowArc!.fillStyle)).toBe(true);
    expect(isWhiteOrGrayColor(noseArc!.fillStyle)).toBe(true);

    const torsoTop = findBoneStroke(records, 620, 280, 380, 280);
    const torsoBottom = findBoneStroke(records, 580, 580, 420, 580);
    const torsoLeft = findBoneStroke(records, 620, 280, 580, 580);
    const torsoRight = findBoneStroke(records, 380, 280, 420, 580);

    for (const edge of [torsoTop, torsoBottom, torsoLeft, torsoRight]) {
      expect(edge).toBeDefined();
      expect(isWhiteOrGrayColor(edge!.strokeStyle)).toBe(true);
    }

    expect(leftElbowArc!.fillStyle).not.toBe(rightElbowArc!.fillStyle);
    expect(leftElbowArc!.fillStyle).not.toBe(torsoTop!.strokeStyle);
    expect(rightElbowArc!.fillStyle).not.toBe(torsoTop!.strokeStyle);
  });

  it("dims low-visibility joints and attached bones", () => {
    const { ctx, records } = createMockCanvasContext();
    const landmarks = createMockLandmarks({
      11: { x: 0.6, y: 0.3, z: 0, visibility: 0.95 },
      13: { x: 0.72, y: 0.45, z: 0, visibility: 0.2 },
      12: { x: 0.4, y: 0.3, z: 0, visibility: 0.95 },
      14: { x: 0.28, y: 0.45, z: 0, visibility: 0.95 },
    });

    drawSkeletonFrame(ctx, 1000, 1000, {
      landmarks,
      angles: {},
      selectedJoints: [],
      viewMode: "overlay",
      mirrored: false,
    });

    const lowVisArc = findJointArcNear(records, 720, 450);
    const highVisArc = findJointArcNear(records, 280, 450);
    expect(lowVisArc).toBeDefined();
    expect(highVisArc).toBeDefined();
    expect(lowVisArc!.globalAlpha).toBeLessThan(highVisArc!.globalAlpha * 0.6);

    const lowVisBone = findBoneStroke(records, 600, 300, 720, 450);
    const highVisBone = findBoneStroke(records, 400, 300, 280, 450);
    expect(lowVisBone).toBeDefined();
    expect(highVisBone).toBeDefined();
    expect(lowVisBone!.globalAlpha).toBeLessThan(
      highVisBone!.globalAlpha * 0.6,
    );
  });

  it("mirrors landmark x via (1 - x) while keeping fillText upright", () => {
    const unmirrored = createMockCanvasContext();
    const landmarks = createMockLandmarks({
      13: { x: 0.2, y: 0.5, z: 0, visibility: 0.95 },
    });

    const baseParams: DrawSkeletonFrameParams = {
      landmarks,
      angles: { left_elbow: 90 },
      selectedJoints: ["left_elbow"],
      viewMode: "overlay",
      mirrored: false,
    };

    drawSkeletonFrame(unmirrored.ctx, 1000, 500, baseParams);
    expect(findJointArcNear(unmirrored.records, 200, 250)).toBeDefined();
    expect(unmirrored.records.fillTextCalls).toHaveLength(1);
    expect(unmirrored.records.fillTextCalls[0].text).toBe("90°");
    expect(unmirrored.records.fillTextCalls[0].x).toBeLessThan(350);
    expect(unmirrored.records.fillTextCalls[0].scaleX).toBeGreaterThan(0);

    const mirroredMock = createMockCanvasContext();
    drawSkeletonFrame(mirroredMock.ctx, 1000, 500, {
      ...baseParams,
      mirrored: true,
    });

    expect(findJointArcNear(mirroredMock.records, 800, 250)).toBeDefined();
    expect(findJointArcNear(mirroredMock.records, 200, 250)).toBeUndefined();
    expect(mirroredMock.records.fillTextCalls).toHaveLength(1);
    expect(mirroredMock.records.fillTextCalls[0].text).toBe("90°");
    expect(mirroredMock.records.fillTextCalls[0].x).toBeGreaterThan(650);
    for (const call of mirroredMock.records.fillTextCalls) {
      expect(call.scaleX).toBe(1);
      expect(call.scaleY).toBe(1);
    }
  });

  it("renders only selected non-null joint angles (including 0°) and clamps pills inside canvas bounds", () => {
    const mockA = createMockCanvasContext();
    const landmarksA = createMockLandmarks({
      13: { x: 0.3, y: 0.5, z: 0, visibility: 0.95 },
      14: { x: 0.7, y: 0.5, z: 0, visibility: 0.95 },
      26: { x: 0.65, y: 0.75, z: 0, visibility: 0.95 },
      25: { x: 0.35, y: 0.75, z: 0, visibility: 0.95 },
    });

    drawSkeletonFrame(mockA.ctx, 800, 600, {
      landmarks: landmarksA,
      angles: {
        left_elbow: 0,
        right_elbow: 120,
        right_knee: null,
        left_knee: 90.4,
      },
      selectedJoints: ["left_elbow", "right_knee", "left_knee"] as JointId[],
      viewMode: "overlay",
      mirrored: false,
    });

    const textsA = mockA.records.fillTextCalls.map((c) => c.text);
    expect(textsA).toContain("0°");
    expect(textsA).toContain("90°");
    expect(textsA).not.toContain("120°");
    expect(textsA.some((t) => t.includes("null") || t.includes("--"))).toBe(
      false,
    );

    // Boundary clamping at top-left (0,0) and bottom-right (1,1)
    const mockB = createMockCanvasContext();
    const landmarksB = createMockLandmarks({
      13: { x: 0, y: 0, z: 0, visibility: 0.95 },
      14: { x: 1, y: 1, z: 0, visibility: 0.95 },
    });

    drawSkeletonFrame(mockB.ctx, 800, 600, {
      landmarks: landmarksB,
      angles: { left_elbow: 142, right_elbow: 45 },
      selectedJoints: ["left_elbow", "right_elbow"],
      viewMode: "overlay",
      mirrored: false,
    });

    expect(mockB.records.fillTextCalls).toHaveLength(2);
    for (const textCall of mockB.records.fillTextCalls) {
      expect(textCall.x).toBeGreaterThanOrEqual(0);
      expect(textCall.x).toBeLessThanOrEqual(800);
      expect(textCall.y).toBeGreaterThanOrEqual(0);
      expect(textCall.y).toBeLessThanOrEqual(600);
    }
    for (const pill of mockB.records.fillRectCalls) {
      expect(pill.x).toBeGreaterThanOrEqual(0);
      expect(pill.y).toBeGreaterThanOrEqual(0);
      expect(pill.x + pill.w).toBeLessThanOrEqual(800);
      expect(pill.y + pill.h).toBeLessThanOrEqual(600);
    }
  });
});

describe("<SkeletonCanvas />", () => {
  beforeEach(() => {
    vi.stubGlobal("ResizeObserver", MockResizeObserver);
    const { ctx } = createMockCanvasContext();
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(
      (contextId: string) =>
        contextId === "2d" ? ctx : null,
    );
  });

  it("renders pose-video, pose-canvas, aspect-locked stage, and omits status-overlay when ready", () => {
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({
      width: 1600,
      height: 1000,
      top: 0,
      left: 0,
      right: 1600,
      bottom: 1000,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    } as DOMRect);

    const props = buildDefaultProps({
      status: "ready",
      videoAspectRatio: 16 / 9,
    });
    render(<SkeletonCanvas {...props} />);

    act(() => {
      for (const observer of MockResizeObserver.instances) {
        observer.triggerResize(1600, 1000);
      }
    });

    const video = screen.getByTestId("pose-video");
    const canvas = screen.getByTestId("pose-canvas");
    const stage = screen.getByTestId("pose-stage");

    expect(video.tagName).toBe("VIDEO");
    expect(canvas.tagName).toBe("CANVAS");
    expect(props.videoRef.current).toBe(video);
    expect(props.canvasRef.current).toBe(canvas);
    expect(stage.style.width).toBe("1600px");
    expect(stage.style.height).toBe("900px");
    expect(screen.queryByTestId("status-overlay")).toBeNull();
  });

  it("flips pose-video via [transform:scaleX(-1)] in mirror mode without flipping pose-canvas", () => {
    const props = buildDefaultProps({ mirrored: false });
    const { rerender } = render(<SkeletonCanvas {...props} />);

    const video = screen.getByTestId("pose-video");
    const canvas = screen.getByTestId("pose-canvas");

    expect(video.className).not.toContain("[transform:scaleX(-1)]");
    expect(video.className).not.toContain("scale-x-[-1]");
    expect(canvas.className).not.toContain("[transform:scaleX(-1)]");
    expect(canvas.className).not.toContain("scale-x-[-1]");

    rerender(<SkeletonCanvas {...props} mirrored={true} />);

    expect(video.className).toContain("[transform:scaleX(-1)]");
    expect(canvas.className).not.toContain("[transform:scaleX(-1)]");
    expect(canvas.className).not.toContain("scale-x-[-1]");
    expect(canvas.style.transform).toBe("");
  });

  it("hides pose-video in blackout mode while keeping pose-canvas visible", () => {
    const props = buildDefaultProps({ viewMode: "overlay" });
    const { rerender } = render(<SkeletonCanvas {...props} />);

    const video = screen.getByTestId("pose-video");
    const canvas = screen.getByTestId("pose-canvas");

    expect(video.className).not.toContain("opacity-0");

    rerender(<SkeletonCanvas {...props} viewMode="blackout" />);

    expect(video.className).toContain("opacity-0");
    expect(video.className).toContain("pointer-events-none");
    expect(canvas.className).not.toContain("opacity-0");
  });

  it("shows status-overlay without retry-button for requesting-camera and loading-model", () => {
    const props = buildDefaultProps({ status: "requesting-camera" });
    const { rerender } = render(<SkeletonCanvas {...props} />);

    const overlay = screen.getByTestId("status-overlay");
    expect(overlay).toHaveTextContent(/requesting camera/i);
    expect(screen.queryByTestId("retry-button")).toBeNull();

    rerender(
      <SkeletonCanvas
        {...props}
        status="loading-model"
        statusMessage="Downloading Heavy weights..."
      />,
    );

    expect(screen.getByTestId("status-overlay")).toHaveTextContent(
      "Downloading Heavy weights...",
    );
    expect(screen.queryByTestId("retry-button")).toBeNull();
  });

  it("shows status-overlay and retry-button for permission-denied, no-camera, and error", () => {
    const errorCases: Array<{
      status: AppStatus;
      expectedText: RegExp;
      statusMessage?: string;
    }> = [
      {
        status: "permission-denied",
        expectedText: /permission denied/i,
      },
      {
        status: "no-camera",
        expectedText: /no camera/i,
      },
      {
        status: "error",
        expectedText: /WebGL context lost/i,
        statusMessage: "WebGL context lost",
      },
    ];

    for (const tc of errorCases) {
      const onRetry = vi.fn();
      const props = buildDefaultProps({
        status: tc.status,
        statusMessage: tc.statusMessage,
        onRetry,
      });
      const { unmount } = render(<SkeletonCanvas {...props} />);

      const overlay = screen.getByTestId("status-overlay");
      expect(overlay).toHaveTextContent(tc.expectedText);

      const retryBtn = screen.getByTestId("retry-button");
      expect(retryBtn).toBeEnabled();
      unmount();
    }
  });

  it("invokes onRetry when retry-button is clicked and handles missing onRetry safely", () => {
    const errorStatuses: AppStatus[] = [
      "permission-denied",
      "no-camera",
      "error",
    ];

    for (const status of errorStatuses) {
      const onRetry = vi.fn();
      const props = buildDefaultProps({ status, onRetry });
      const { unmount } = render(<SkeletonCanvas {...props} />);

      fireEvent.click(screen.getByTestId("retry-button"));
      expect(onRetry).toHaveBeenCalledTimes(1);
      unmount();
    }

    const noHandlerProps = buildDefaultProps({
      status: "error",
      onRetry: undefined,
    });
    render(<SkeletonCanvas {...noHandlerProps} />);
    expect(() =>
      fireEvent.click(screen.getByTestId("retry-button")),
    ).not.toThrow();
  });

  it("synchronizes HiDPI backing store via ResizeObserver and disconnects on unmount", () => {
    Object.defineProperty(window, "devicePixelRatio", {
      configurable: true,
      value: 2,
    });

    const { ctx, records } = createMockCanvasContext();
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(ctx);
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({
      width: 640,
      height: 480,
      top: 0,
      left: 0,
      right: 640,
      bottom: 480,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    } as DOMRect);

    const props = buildDefaultProps({
      landmarks: createMockLandmarks(),
      viewMode: "blackout",
    });
    const { unmount } = render(<SkeletonCanvas {...props} />);

    act(() => {
      for (const observer of MockResizeObserver.instances) {
        observer.triggerResize(640, 480);
      }
    });

    const canvas = screen.getByTestId("pose-canvas") as HTMLCanvasElement;
    expect(canvas.width).toBe(1280);
    expect(canvas.height).toBe(960);
    expect(records.setTransformCalls).toContainEqual([2, 0, 0, 2, 0, 0]);

    unmount();
    expect(MockResizeObserver.instances.length).toBeGreaterThan(0);
    for (const observer of MockResizeObserver.instances) {
      expect(observer.disconnectSpy).toHaveBeenCalled();
    }
  });

  it("renders V2 21-point hands/fingers and 478-point face/eyes/iris/cranial dome layers", () => {
    const { ctx, records } = createMockCanvasContext();
    const poseLandmarks = createMockLandmarks();
    const handLandmarks: Point3D[] = Array.from({ length: 21 }, (_, idx) => ({
      x: 0.3 + idx * 0.01,
      y: 0.6 - idx * 0.01,
      z: -0.05,
    }));
    const faceLandmarks: Point3D[] = Array.from({ length: 478 }, (_, idx) => ({
      x: 0.45 + (idx % 10) * 0.01,
      y: 0.2 + Math.floor(idx / 50) * 0.02,
      z: -0.02,
    }));

    drawSkeletonFrame(ctx, 800, 600, {
      landmarks: poseLandmarks,
      hands: [
        {
          handedness: "Left",
          score: 0.99,
          landmarks: handLandmarks,
        },
      ],
      faceLandmarks,
      showFaceMesh: true,
      viewMode: "blackout",
      mirrored: true,
    });

    expect(records.strokeCalls.length).toBeGreaterThan(40);
    expect(records.arcCalls.length).toBeGreaterThan(40);
    expect(
      records.fillTextCalls.some((c) => c.text.includes("L.HAND")),
    ).toBe(true);
  });
});
