import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const mockInit = vi.fn();
const mockSwitchModel = vi.fn();
const mockUpdateThresholds = vi.fn();
const mockDetectFrame = vi.fn();
const mockDispose = vi.fn();
const mockIsReady = vi.fn();
const mockGetActiveDelegate = vi.fn();
const mockGetCurrentModel = vi.fn();

vi.mock("./engine/poseEngine", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./engine/poseEngine")>();
  return {
    ...actual,
    PoseEngine: class MockPoseEngine {
      init = mockInit;
      switchModel = mockSwitchModel;
      updateThresholds = mockUpdateThresholds;
      detectFrame = mockDetectFrame;
      dispose = mockDispose;
      isReady = mockIsReady;
      getActiveDelegate = mockGetActiveDelegate;
      getCurrentModel = mockGetCurrentModel;
    },
  };
});

function createFakeMediaStream(): MediaStream {
  const track = {
    stop: vi.fn(),
    kind: "video",
  } as unknown as MediaStreamTrack;
  return {
    getTracks: () => [track],
    getVideoTracks: () => [track],
  } as unknown as MediaStream;
}

describe("App integration", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockInit.mockResolvedValue("GPU");
    mockSwitchModel.mockResolvedValue("GPU");
    mockIsReady.mockReturnValue(true);
    mockGetActiveDelegate.mockReturnValue("GPU");
    mockGetCurrentModel.mockReturnValue("lite");
    mockDetectFrame.mockReturnValue({
      landmarks: null,
      inferenceMs: 2.5,
      delegate: "GPU",
    });

    Object.defineProperty(HTMLMediaElement.prototype, "play", {
      configurable: true,
      writable: true,
      value: vi.fn().mockResolvedValue(undefined),
    });

    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      writable: true,
      value: {
        getUserMedia: vi.fn().mockResolvedValue(createFakeMediaStream()),
      },
    });
  });

  afterEach(() => {
    cleanup();
  });

  it("initializes camera and PoseEngine and transitions to ready state", async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.queryByTestId("status-overlay")).not.toBeInTheDocument();
    });

    expect(mockInit).toHaveBeenCalledWith(
      expect.objectContaining({
        model: "lite",
        minPoseDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      }),
    );
    expect(screen.getByTestId("hud-fps")).toBeInTheDocument();
    expect(screen.getByTestId("hud-inference-ms")).toBeInTheDocument();
    expect(screen.getByTestId("hud-visible-joints")).toHaveTextContent(
      /\d+\s*\/\s*33/,
    );
  });

  it("shows permission-denied status overlay and recovers on Retry", async () => {
    const getUserMediaMock = vi
      .fn()
      .mockRejectedValueOnce(
        new DOMException("Permission denied", "NotAllowedError"),
      )
      .mockResolvedValueOnce(createFakeMediaStream());

    Object.defineProperty(HTMLMediaElement.prototype, "play", {
      configurable: true,
      writable: true,
      value: vi.fn().mockResolvedValue(undefined),
    });

    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      writable: true,
      value: { getUserMedia: getUserMediaMock },
    });

    render(<App />);

    const retryBtn = await screen.findByTestId("retry-button");
    expect(screen.getByTestId("status-overlay")).toBeInTheDocument();

    fireEvent.click(retryBtn);

    await waitFor(() => {
      expect(screen.queryByTestId("status-overlay")).not.toBeInTheDocument();
    });
    expect(getUserMediaMock).toHaveBeenCalledTimes(2);
  });

  it("collapses and expands control-panel while keeping Performance HUD visible", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.queryByTestId("status-overlay")).not.toBeInTheDocument();
    });

    const toggleBtn = screen.getByTestId("panel-toggle");
    expect(toggleBtn).toHaveAttribute("aria-expanded", "true");

    fireEvent.click(toggleBtn);
    expect(toggleBtn).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByTestId("control-panel")).toHaveClass("hidden");
    expect(screen.getByTestId("hud-fps")).toBeVisible();

    fireEvent.click(toggleBtn);
    expect(toggleBtn).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByTestId("control-panel")).not.toHaveClass("hidden");
  });

  it("switches models and updates confidence thresholds live", async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.queryByTestId("status-overlay")).not.toBeInTheDocument();
    });

    fireEvent.change(screen.getByTestId("slider-detection-confidence"), {
      target: { value: "0.75" },
    });
    expect(mockUpdateThresholds).toHaveBeenCalledWith(0.75, 0.5);

    fireEvent.click(screen.getByTestId("model-full"));
    await waitFor(() => {
      expect(mockSwitchModel).toHaveBeenCalledWith("full");
    });
  });
});
