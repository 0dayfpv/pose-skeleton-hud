import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  JOINT_DEFINITIONS,
  type JointId,
  type ModelVariant,
  type SmoothingConfig,
  type ViewMode,
} from "../engine/geometry";
import ControlPanel, { type ControlPanelProps } from "./ControlPanel";

function createDefaultProps(
  overrides: Partial<ControlPanelProps> = {},
): ControlPanelProps {
  return {
    isOpen: true,
    onToggleOpen: vi.fn(),
    telemetry: {
      fps: 60,
      inferenceMs: 4.2,
      visibleLandmarks: 33,
      totalLandmarks: 33,
      delegate: "GPU",
    },
    viewMode: "overlay" as ViewMode,
    onViewModeChange: vi.fn(),
    mirrored: false,
    onMirroredChange: vi.fn(),
    model: "lite" as ModelVariant,
    onModelChange: vi.fn(),
    minPoseDetectionConfidence: 0.5,
    onMinPoseDetectionConfidenceChange: vi.fn(),
    minTrackingConfidence: 0.5,
    onMinTrackingConfidenceChange: vi.fn(),
    smoothing: { enabled: true, strength: 0.5 } as SmoothingConfig,
    onSmoothingChange: vi.fn(),
    selectedJoints: JOINT_DEFINITIONS.map((d) => d.id) as readonly JointId[],
    onToggleJoint: vi.fn(),
    angles: {
      left_elbow: 92.4,
      right_elbow: 145.0,
      left_shoulder: null,
      right_shoulder: 45.2,
      left_hip: 175.8,
      right_hip: 176.1,
      left_knee: 168.0,
      right_knee: null,
    },
    onExportSnapshot: vi.fn(),
    ...overrides,
  };
}

describe("ControlPanel component", () => {
  afterEach(() => {
    cleanup();
  });

  it("formats angle readouts safely with zero NaN or undefined strings", () => {
    const props = createDefaultProps({
      angles: {
        left_elbow: 90,
        right_elbow: 135.46,
        left_shoulder: null,
        right_shoulder: Number.NaN,
        left_hip: 180,
        right_hip: 0,
        left_knee: 120,
        right_knee: 45,
      },
      selectedJoints: [
        "left_elbow",
        "right_elbow",
        "left_shoulder",
        "right_shoulder",
      ],
    });
    render(<ControlPanel {...props} />);
    expect(screen.getByTestId("joint-angle-left_elbow")).toHaveTextContent(
      "90.0°",
    );
    expect(screen.getByTestId("joint-angle-right_elbow")).toHaveTextContent(
      "135.5°",
    );
    expect(screen.getByTestId("joint-angle-left_shoulder")).toHaveTextContent(
      "--°",
    );
    expect(screen.getByTestId("joint-angle-right_shoulder")).toHaveTextContent(
      "--°",
    );
    expect(screen.getByTestId("joint-angle-left_hip")).toHaveTextContent("--°");
  });

  it("renders all required controls and joint readouts when open", () => {
    const props = createDefaultProps();
    render(<ControlPanel {...props} />);

    const panel = screen.getByTestId("control-panel");
    expect(panel).toBeInTheDocument();
    expect(panel).not.toHaveClass("hidden");

    expect(screen.getByTestId("mode-overlay")).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByTestId("mode-blackout")).toHaveAttribute(
      "aria-pressed",
      "false",
    );
    expect(screen.getByTestId("mirror-toggle")).toHaveAttribute(
      "aria-pressed",
      "false",
    );

    expect(screen.getByTestId("model-lite")).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByTestId("model-full")).toHaveAttribute(
      "aria-pressed",
      "false",
    );
    expect(screen.getByTestId("model-heavy")).toHaveAttribute(
      "aria-pressed",
      "false",
    );

    expect(screen.getByTestId("joint-angle-left_elbow")).toHaveTextContent(
      "92.4°",
    );
    expect(screen.getByTestId("joint-angle-left_shoulder")).toHaveTextContent(
      "--°",
    );
  });

  it("hides control-panel when isOpen is false", () => {
    const props = createDefaultProps({ isOpen: false });
    render(<ControlPanel {...props} />);

    expect(screen.getByTestId("control-panel")).toHaveClass("hidden");
  });

  it("triggers view mode, mirror, model, and snapshot callbacks", () => {
    const props = createDefaultProps();
    render(<ControlPanel {...props} />);

    fireEvent.click(screen.getByTestId("mode-blackout"));
    expect(props.onViewModeChange).toHaveBeenCalledWith("blackout");

    fireEvent.click(screen.getByTestId("mirror-toggle"));
    expect(props.onMirroredChange).toHaveBeenCalledWith(true);

    fireEvent.click(screen.getByTestId("model-full"));
    expect(props.onModelChange).toHaveBeenCalledWith("full");

    fireEvent.click(screen.getByTestId("model-heavy"));
    expect(props.onModelChange).toHaveBeenCalledWith("heavy");

    fireEvent.click(screen.getByTestId("snapshot-json-button"));
    expect(props.onExportSnapshot).toHaveBeenCalledTimes(1);
  });

  it("updates confidence sliders and smoothing controls with proper clamping", () => {
    const props = createDefaultProps();
    render(<ControlPanel {...props} />);

    fireEvent.change(screen.getByTestId("slider-detection-confidence"), {
      target: { value: "0.75" },
    });
    expect(props.onMinPoseDetectionConfidenceChange).toHaveBeenCalledWith(0.75);

    fireEvent.change(screen.getByTestId("slider-tracking-confidence"), {
      target: { value: "0.65" },
    });
    expect(props.onMinTrackingConfidenceChange).toHaveBeenCalledWith(0.65);

    fireEvent.click(screen.getByTestId("smoothing-toggle"));
    expect(props.onSmoothingChange).toHaveBeenCalledWith({
      enabled: false,
      strength: 0.5,
    });

    fireEvent.change(screen.getByTestId("slider-smoothing-strength"), {
      target: { value: "0.8" },
    });
    expect(props.onSmoothingChange).toHaveBeenCalledWith({
      enabled: true,
      strength: 0.8,
    });
  });

  it("toggles each of the 8 joints independently", () => {
    const props = createDefaultProps();
    render(<ControlPanel {...props} />);

    for (const def of JOINT_DEFINITIONS) {
      fireEvent.click(screen.getByTestId(`joint-toggle-${def.id}`));
      expect(props.onToggleJoint).toHaveBeenCalledWith(def.id);
    }
  });

  it("renders and toggles V2 detail modules (hands, face, 3D face mesh) and biometrics", () => {
    const onDetailModulesChange = vi.fn();
    const props = createDefaultProps({
      detailModules: {
        enableHands: true,
        enableFace: true,
        showFaceMesh: true,
      },
      onDetailModulesChange,
      faceHeadMetrics: {
        headYawDeg: -12.5,
        headPitchDeg: 5.0,
        headRollDeg: 2.3,
        leftEyeOpenness: 92,
        rightEyeOpenness: 94,
        mouthOpenness: 18,
      },
      handMetrics: [
        {
          handedness: "Left",
          score: 0.99,
          fingerAngles: {
            thumb: 170,
            index: 175,
            middle: 172,
            ring: 168,
            pinky: 165,
          },
          extendedFingers: 5,
          pinchDistance: 0.12,
          isPinching: false,
          opennessPercent: 95,
        },
      ],
    });

    render(<ControlPanel {...props} />);

    expect(screen.getByTestId("head-yaw-deg")).toHaveTextContent("-12.5°");
    expect(screen.getByTestId("head-pitch-deg")).toHaveTextContent("5°");
    expect(screen.getByTestId("head-roll-deg")).toHaveTextContent("2.3°");
    expect(screen.getByTestId("control-panel")).toHaveTextContent(
      "LEFT HAND (5/5 EXT)",
    );

    fireEvent.click(screen.getByTestId("toggle-hands"));
    expect(onDetailModulesChange).toHaveBeenCalledWith({
      enableHands: false,
      enableFace: true,
      showFaceMesh: true,
    });

    fireEvent.click(screen.getByTestId("toggle-face"));
    expect(onDetailModulesChange).toHaveBeenCalledWith({
      enableHands: true,
      enableFace: false,
      showFaceMesh: true,
    });

    fireEvent.click(screen.getByTestId("toggle-face-mesh"));
    expect(onDetailModulesChange).toHaveBeenCalledWith({
      enableHands: true,
      enableFace: true,
      showFaceMesh: false,
    });
  });
});
