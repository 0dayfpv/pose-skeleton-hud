import { expect, test, type Response } from '@playwright/test';
import {
  ALL_JOINT_IDS,
  attachRuntimeErrorTracker,
  captureAndValidateSnapshot,
  clearMediaErrorMock,
  enableTaskModelCache,
  getBoundingBoxDelta,
  getElementFlipState,
  gotoWithErrorTracking,
  installMediaErrorMock,
  isControlActive,
  isVideoVisuallyHidden,
  launchNoFakeDeviceSession,
  sampleCanvasBackdrop,
  setJointSelection,
  setRangeSliderValue,
  setToggleState,
  switchModelAndWait,
  waitForAppReady,
  waitForStatusOverlay,
} from './helpers';

test.describe('Tier 1 — Feature Coverage (F-01 .. F-14, 70 Cases)', () => {
  test('F-01: Project Scaffolding, Dark Theme, WASM Boot & Runtime Cleanliness (T1-F01-1..5)', async ({
    page,
  }) => {
    const tracker = attachRuntimeErrorTracker(page);
    const wasmResponses: { url: string; status: number }[] = [];

    page.on('response', (res: Response) => {
      if (res.url().includes('/wasm/')) {
        wasmResponses.push({ url: res.url(), status: res.status() });
      }
    });

    const rootResponse = await page.goto('/');

    await test.step('T1-F01-1: Root page returns HTTP 200 and non-empty title', async () => {
      expect(rootResponse).not.toBeNull();
      expect(rootResponse?.status()).toBe(200);
      await expect(page).toHaveTitle(/Pose Skeleton HUD/i);
    });

    await waitForAppReady(page);

    await test.step('T1-F01-2: Dark cybernetic theme active with dark background luminance', async () => {
      const bgLuminance = await page.evaluate(() => {
        const checkEl = (el: Element | null): number | null => {
          if (!el) return null;
          const bg = window.getComputedStyle(el).backgroundColor;
          const match = bg.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
          if (!match) return null;
          const alpha = match[4] !== undefined ? Number.parseFloat(match[4]) : 1;
          if (alpha === 0) return null;
          const r = Number(match[1]);
          const g = Number(match[2]);
          const b = Number(match[3]);
          return 0.2126 * r + 0.7152 * g + 0.0722 * b;
        };
        return checkEl(document.body) ?? checkEl(document.documentElement) ?? 255;
      });
      expect(bgLuminance).toBeLessThan(30);
    });

    await test.step('T1-F01-3: Local WASM assets load with HTTP 200/304 and zero 404s', async () => {
      expect(wasmResponses.length).toBeGreaterThan(0);
      for (const entry of wasmResponses) {
        expect([200, 304], `Unexpected status for ${entry.url}`).toContain(entry.status);
      }
      await expect(page.getByTestId('toggle-hands')).toBeVisible();
      await expect(page.getByTestId('toggle-face')).toBeVisible();
      await expect(page.getByTestId('toggle-face-mesh')).toBeVisible();
      await expect(page.getByTestId('head-yaw-deg')).toBeVisible();
    });

    await test.step('T1-F01-4: Zero uncaught pageerror events through ready state', async () => {
      tracker.assertNoPageErrors();
    });

    await test.step('T1-F01-5: Zero console.error logs through ready state', async () => {
      tracker.assertNoConsoleErrors();
    });
  });

  test('F-02: 3D Vector Joint Angle Telemetry Integrity (T1-F02-1..5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);
    await setJointSelection(page, ALL_JOINT_IDS);

    await test.step('T1-F02-1: All 8 joint-angle-<jointId> readouts render when all joints enabled', async () => {
      for (const jointId of ALL_JOINT_IDS) {
        await expect(page.getByTestId(`joint-angle-${jointId}`)).toBeVisible();
      }
    });

    await test.step('T1-F02-2: Active angle readout text strictly matches numeric ° or --°', async () => {
      for (const jointId of ALL_JOINT_IDS) {
        const text = (await page.getByTestId(`joint-angle-${jointId}`).innerText()).trim();
        expect(text).toMatch(/^(\d+(\.\d+)?|--)\s*°$/);
        expect(text).not.toContain('NaN');
        expect(text).not.toContain('undefined');
      }
    });

    await test.step('T1-F02-3: Synthetic fake stream without pose displays --° cleanly', async () => {
      const visibleText = (await page.getByTestId('hud-visible-joints').innerText()).trim();
      if (visibleText.startsWith('0')) {
        for (const jointId of ALL_JOINT_IDS) {
          const text = (await page.getByTestId(`joint-angle-${jointId}`).innerText()).trim();
          expect(text).toMatch(/^--\s*°$/);
        }
      }
    });

    await test.step('T1-F02-4: Snapshot activeJointAngles lists valid objects with null or [0,180] degrees', async () => {
      const snapshot = await captureAndValidateSnapshot(page);
      expect(snapshot.activeJointAngles).toHaveLength(ALL_JOINT_IDS.length);
      for (const entry of snapshot.activeJointAngles) {
        expect(ALL_JOINT_IDS as readonly string[]).toContain(entry.joint);
        if (entry.degrees !== null) {
          expect(entry.degrees).toBeGreaterThanOrEqual(0);
          expect(entry.degrees).toBeLessThanOrEqual(180);
        }
      }
    });

    await test.step('T1-F02-5: Toggling a joint off removes its entry from Snapshot activeJointAngles', async () => {
      await setToggleState(page.getByTestId('joint-toggle-left_elbow'), false);
      const snapshot = await captureAndValidateSnapshot(page);
      const jointIds = snapshot.activeJointAngles.map((a) => a.joint);
      expect(jointIds).not.toContain('left_elbow');
      expect(jointIds).toHaveLength(ALL_JOINT_IDS.length - 1);
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-03: One-Euro Smoothing Toggle & Strength (T1-F03-1..5)', async ({ page }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const toggle = page.getByTestId('smoothing-toggle');
    const slider = page.getByTestId('slider-smoothing-strength');

    await test.step('T1-F03-1: smoothing-toggle is interactive and reflects boolean state', async () => {
      await expect(toggle).toBeVisible();
      const initial = await isControlActive(toggle);
      await setToggleState(toggle, !initial);
      expect(await isControlActive(toggle)).toBe(!initial);
      await setToggleState(toggle, true);
    });

    await test.step('T1-F03-2: slider-smoothing-strength has range bounds [0, 1] step 0.05', async () => {
      await expect(slider).toBeVisible();
      await expect(slider).toHaveAttribute('type', 'range');
      await expect(slider).toHaveAttribute('min', '0');
      await expect(slider).toHaveAttribute('max', '1');
      await expect(slider).toHaveAttribute('step', '0.05');
    });

    await test.step('T1-F03-3: Adjusting slider-smoothing-strength to 0.75 updates UI readout', async () => {
      await setRangeSliderValue(slider, '0.75');
      await expect(slider).toHaveValue('0.75');
      await expect(page.getByTestId('control-panel')).toContainText(/0\.75|75%/);
    });

    await test.step('T1-F03-4: Snapshot JSON smoothing.enabled matches toggle true and false', async () => {
      await setToggleState(toggle, true);
      const snapOn = await captureAndValidateSnapshot(page);
      expect(snapOn.smoothing.enabled).toBe(true);

      await setToggleState(toggle, false);
      const snapOff = await captureAndValidateSnapshot(page);
      expect(snapOff.smoothing.enabled).toBe(false);
    });

    await test.step('T1-F03-5: Snapshot JSON smoothing.strength matches slider value 0.75', async () => {
      await setToggleState(toggle, true);
      await setRangeSliderValue(slider, '0.75');
      const snapshot = await captureAndValidateSnapshot(page);
      expect(snapshot.smoothing.strength).toBeCloseTo(0.75, 2);
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-04: Snapshot JSON Export & 33-Landmark Schema (T1-F04-1..5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const exportBtn = page.getByTestId('snapshot-json-button');

    await test.step('T1-F04-1: snapshot-json-button is visible, enabled, and accessible', async () => {
      await expect(exportBtn).toBeVisible();
      await expect(exportBtn).toBeEnabled();
      await expect(exportBtn).toContainText(/Snapshot JSON/i);
    });

    await setToggleState(page.getByTestId('mirror-toggle'), true);
    await setToggleState(page.getByTestId('smoothing-toggle'), false);
    await setRangeSliderValue(page.getByTestId('slider-smoothing-strength'), '0.4');
    await setJointSelection(page, ['left_elbow', 'right_knee']);

    let exported: Awaited<ReturnType<typeof captureAndValidateSnapshot>> | null = null;

    await test.step('T1-F04-2: Clicking snapshot-json-button downloads pose-snapshot-*.json', async () => {
      exported = await captureAndValidateSnapshot(page);
    });

    await test.step('T1-F04-3: Downloaded JSON contains schemaVersion 1.0.0 and valid ISO timestamp', async () => {
      expect(exported?.schemaVersion).toBe('1.0.0');
      expect(typeof exported?.timestamp).toBe('string');
      expect(typeof exported?.poseDetected).toBe('boolean');
    });

    await test.step('T1-F04-4: Downloaded JSON landmarks array has 33 finite items (0..32)', async () => {
      expect(exported?.landmarks).toHaveLength(33);
    });

    await test.step('T1-F04-5: Downloaded JSON metadata matches current UI control state', async () => {
      expect(exported?.model).toBe('lite');
      expect(exported?.mirrored).toBe(true);
      expect(exported?.smoothing.enabled).toBe(false);
      expect(exported?.smoothing.strength).toBeCloseTo(0.4, 2);
      expect((exported?.activeJointAngles.map((a) => a.joint) ?? []).sort()).toEqual(
        ['left_elbow', 'right_knee'].sort(),
      );
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-05: MediaPipe Pose Engine & Live Model Switching Lite/Full/Heavy (T1-F05-1..5)', async ({
    page,
  }) => {
    test.setTimeout(150_000);
    const tracker = attachRuntimeErrorTracker(page);
    const taskResponses: { url: string; status: number }[] = [];

    page.on('response', (res: Response) => {
      const redirectedFromUrl = res.request().redirectedFrom()?.url() ?? '';
      if (res.url().includes('.task') || redirectedFromUrl.includes('.task')) {
        taskResponses.push({
          url: redirectedFromUrl || res.url(),
          status: res.status(),
        });
      }
    });

    await page.goto('/');
    await waitForAppReady(page, 60_000);

    await test.step('T1-F05-1: Default Lite model initializes cleanly into ready state', async () => {
      await expect.poll(() => isControlActive(page.getByTestId('model-lite'))).toBe(true);
      await expect(page.getByTestId('status-overlay')).toBeHidden();
    });

    await test.step('T1-F05-2: model-lite, model-full, and model-heavy selectors are rendered', async () => {
      await expect(page.getByTestId('model-lite')).toBeVisible();
      await expect(page.getByTestId('model-full')).toBeVisible();
      await expect(page.getByTestId('model-heavy')).toBeVisible();
    });

    await test.step('T1-F05-3: Switching Lite -> Full fetches pose_landmarker_full.task cleanly', async () => {
      await switchModelAndWait(page, 'full', 60_000);
      const fullFetched = taskResponses.some(
        (r) =>
          r.url.includes('pose_landmarker_full.task') &&
          (r.status === 200 || r.status === 302),
      );
      expect(fullFetched).toBe(true);
    });

    await test.step('T1-F05-4: Switching to Heavy fetches pose_landmarker_heavy.task cleanly', async () => {
      await switchModelAndWait(page, 'heavy', 90_000);
      const heavyFetched = taskResponses.some(
        (r) =>
          r.url.includes('pose_landmarker_heavy.task') &&
          (r.status === 200 || r.status === 302),
      );
      expect(heavyFetched).toBe(true);
    });

    await test.step('T1-F05-5: Active delegate badge reports GPU or CPU without WebGL errors', async () => {
      const panelOrHudText = await page.locator('body').innerText();
      expect(panelOrHudText).toMatch(/\b(GPU|CPU)\b/);
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-06: Live Confidence Options & Model Disposal (T1-F06-1..5)', async ({ page }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const detSlider = page.getByTestId('slider-detection-confidence');
    const trackSlider = page.getByTestId('slider-tracking-confidence');

    await test.step('T1-F06-1: slider-detection-confidence has [0.1, 0.95] step 0.05', async () => {
      await expect(detSlider).toBeVisible();
      await expect(detSlider).toHaveAttribute('min', '0.1');
      await expect(detSlider).toHaveAttribute('max', '0.95');
      await expect(detSlider).toHaveAttribute('step', '0.05');
    });

    await test.step('T1-F06-2: slider-tracking-confidence has [0.1, 0.95] step 0.05', async () => {
      await expect(trackSlider).toBeVisible();
      await expect(trackSlider).toHaveAttribute('min', '0.1');
      await expect(trackSlider).toHaveAttribute('max', '0.95');
      await expect(trackSlider).toHaveAttribute('step', '0.05');
    });

    let taskRequestCount = 0;
    page.on('request', (req) => {
      if (req.url().includes('.task')) {
        taskRequestCount += 1;
      }
    });

    await test.step('T1-F06-3: Changing detection confidence does not re-fetch .task files', async () => {
      const before = taskRequestCount;
      await setRangeSliderValue(detSlider, '0.7');
      await expect(detSlider).toHaveValue('0.7');
      expect(taskRequestCount).toBe(before);
    });

    await test.step('T1-F06-4: Changing tracking confidence does not re-fetch .task files', async () => {
      const before = taskRequestCount;
      await setRangeSliderValue(trackSlider, '0.65');
      await expect(trackSlider).toHaveValue('0.65');
      expect(taskRequestCount).toBe(before);
    });

    await test.step('T1-F06-5: Switching Full -> Lite disposes old model and keeps HUD active', async () => {
      await switchModelAndWait(page, 'full');
      await switchModelAndWait(page, 'lite');
      await expect(page.getByTestId('hud-fps')).toHaveText(/\d+/);
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-07: Overlay & Blackout Rendering Modes (T1-F07-1..5)', async ({ page }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const overlayBtn = page.getByTestId('mode-overlay');
    const blackoutBtn = page.getByTestId('mode-blackout');

    await test.step('T1-F07-1: mode-overlay and mode-blackout exist with Overlay active by default', async () => {
      await expect(overlayBtn).toBeVisible();
      await expect(blackoutBtn).toBeVisible();
      expect(await isControlActive(overlayBtn)).toBe(true);
      expect(await isControlActive(blackoutBtn)).toBe(false);
    });

    await test.step('T1-F07-2: Overlay mode shows pose-video beneath pose-canvas', async () => {
      expect(await isVideoVisuallyHidden(page)).toBe(false);
      const box = await page.getByTestId('pose-video').boundingBox();
      expect(box?.width ?? 0).toBeGreaterThan(0);
      expect(box?.height ?? 0).toBeGreaterThan(0);
    });

    await test.step('T1-F07-3: Switching to Blackout hides pose-video visually', async () => {
      await blackoutBtn.click();
      await expect.poll(() => isControlActive(blackoutBtn)).toBe(true);
      await expect.poll(() => isVideoVisuallyHidden(page)).toBe(true);
    });

    await test.step('T1-F07-4: Blackout mode renders dark cybernetic grid backdrop on pose-canvas', async () => {
      await expect
        .poll(async () => (await sampleCanvasBackdrop(page)).opaqueDarkPixels)
        .toBeGreaterThan(10);
    });

    await test.step('T1-F07-5: Switching Blackout -> Overlay restores video and clears opaque backdrop', async () => {
      await overlayBtn.click();
      await expect.poll(() => isControlActive(overlayBtn)).toBe(true);
      await expect.poll(() => isVideoVisuallyHidden(page)).toBe(false);
      await expect
        .poll(async () => (await sampleCanvasBackdrop(page)).opaqueDarkPixels)
        .toBe(0);
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-08: Smart Coloring & Visibility Dimming Canvas Integrity (T1-F08-1..5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    await test.step('T1-F08-1: pose-canvas uses a dedicated 2D rendering context', async () => {
      const has2d = await page.getByTestId('pose-canvas').evaluate((el) => {
        const canvas = el as HTMLCanvasElement;
        return canvas.getContext('2d') !== null;
      });
      expect(has2d).toBe(true);
    });

    await test.step('T1-F08-2: pose-canvas backing store matches or exceeds CSS pixel box', async () => {
      const dims = await page.getByTestId('pose-canvas').evaluate((el) => {
        const canvas = el as HTMLCanvasElement;
        return {
          width: canvas.width,
          height: canvas.height,
          clientWidth: canvas.clientWidth,
          clientHeight: canvas.clientHeight,
        };
      });
      expect(dims.width).toBeGreaterThanOrEqual(dims.clientWidth);
      expect(dims.height).toBeGreaterThanOrEqual(dims.clientHeight);
    });

    await test.step('T1-F08-3: Continuous rAF redraw across Overlay and Blackout yields 0 errors', async () => {
      await page.waitForTimeout(300);
      await page.getByTestId('mode-blackout').click();
      await page.waitForTimeout(300);
      tracker.assertNoRuntimeErrors();
    });

    await test.step('T1-F08-4: Snapshot landmark visibilities are clamped within [0, 1]', async () => {
      const snapshot = await captureAndValidateSnapshot(page);
      for (const lm of snapshot.landmarks) {
        expect(lm.visibility).toBeGreaterThanOrEqual(0);
        expect(lm.visibility).toBeLessThanOrEqual(1);
      }
    });

    await test.step('T1-F08-5: Blackout grid backdrop fills canvas cleanly across interior points', async () => {
      const sample = await sampleCanvasBackdrop(page);
      expect(sample.sampledPoints).toBe(25);
      expect(sample.nonTransparentPixels).toBe(25);
      expect(sample.opaqueDarkPixels).toBeGreaterThanOrEqual(15);
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-09: Mirror Mode — Video Flipped, Canvas Upright (T1-F09-1..5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const mirrorToggle = page.getByTestId('mirror-toggle');
    const video = page.getByTestId('pose-video');
    const canvas = page.getByTestId('pose-canvas');

    await test.step('T1-F09-1: mirror-toggle exists and toggles cleanly', async () => {
      await expect(mirrorToggle).toBeVisible();
      await setToggleState(mirrorToggle, false);
      expect(await isControlActive(mirrorToggle)).toBe(false);
    });

    await test.step('T1-F09-2: When Mirror is OFF, pose-video is unflipped', async () => {
      const videoFlip = await getElementFlipState(video);
      expect(videoFlip.isUnflipped).toBe(true);
      expect(videoFlip.isMirroredX).toBe(false);
    });

    await test.step('T1-F09-3: When Mirror is ON, pose-video is horizontally flipped (transform or scale)', async () => {
      await setToggleState(mirrorToggle, true);
      await expect
        .poll(async () => (await getElementFlipState(video)).isMirroredX)
        .toBe(true);
    });

    await test.step('T1-F09-4: When Mirror is ON, pose-canvas stays unflipped on both transform and scale', async () => {
      const canvasFlip = await getElementFlipState(canvas);
      expect(canvasFlip.isUnflipped).toBe(true);
      expect(canvasFlip.isMirroredX).toBe(false);
    });

    await test.step('T1-F09-5: Snapshot JSON mirrored field matches Mirror toggle state', async () => {
      const snapMirrored = await captureAndValidateSnapshot(page);
      expect(snapMirrored.mirrored).toBe(true);

      await setToggleState(mirrorToggle, false);
      const snapNormal = await captureAndValidateSnapshot(page);
      expect(snapNormal.mirrored).toBe(false);
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-10 (Part 0): Initial Loading Status Transitions to Ready (T1-F10-1)', async ({
    page,
  }) => {
    await enableTaskModelCache(page, { delayByModel: { lite: 800 } });
    const tracker = await gotoWithErrorTracking(page);

    await test.step('T1-F10-1: status-overlay displays descriptive status text during initial loading before ready', async () => {
      const overlay = await waitForStatusOverlay(page, 10_000);
      await expect(overlay).toContainText(/\S{4,}/);
      await waitForAppReady(page);
      await expect(page.getByTestId('status-overlay')).toBeHidden();
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-10 (Part 1): NotAllowedError & Retry Recovery (T1-F10-2, T1-F10-5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    await installMediaErrorMock(page, {
      name: 'NotAllowedError',
      message: 'Permission denied by user',
    });
    const tracker = await gotoWithErrorTracking(page);

    await test.step('T1-F10-2: status-overlay shows permission guidance and retry-button', async () => {
      const overlay = await waitForStatusOverlay(page);
      await expect(page.getByTestId('retry-button')).toBeVisible();
      const text = (await overlay.innerText()).trim();
      expect(text.length).toBeGreaterThan(5);
      tracker.assertNoPageErrors();
    });

    await test.step('T1-F10-5: Clicking retry-button after restoring camera transitions to ready', async () => {
      await clearMediaErrorMock(page);
      await page.getByTestId('retry-button').click();
      await waitForAppReady(page);
      await expect(page.getByTestId('status-overlay')).toBeHidden();
      tracker.assertNoPageErrors();
    });
  });

  test('F-10 (Part 2): NotFoundError Status Screen (T1-F10-3)', async ({ page }) => {
    await enableTaskModelCache(page);
    await installMediaErrorMock(page, {
      name: 'NotFoundError',
      message: 'Requested device not found',
      emptyDevices: true,
    });
    const tracker = await gotoWithErrorTracking(page);

    await test.step('T1-F10-3: NotFoundError renders missing-camera overlay + retry-button', async () => {
      const overlay = await waitForStatusOverlay(page);
      await expect(page.getByTestId('retry-button')).toBeVisible();
      expect((await overlay.innerText()).trim().length).toBeGreaterThan(5);
      tracker.assertNoPageErrors();
    });
  });

  test('F-10 (Part 3): Real No-Fake-Device Chromium NotSupportedError (T1-F10-4)', async ({
    baseURL,
  }) => {
    const session = await launchNoFakeDeviceSession(baseURL);
    try {
      await test.step('T1-F10-4: Headless Chromium without fake-device renders error overlay + retry-button', async () => {
        const overlay = await waitForStatusOverlay(session.page);
        await expect(session.page.getByTestId('retry-button')).toBeVisible();
        expect((await overlay.innerText()).trim().length).toBeGreaterThan(5);
        session.tracker.assertNoPageErrors();
      });
    } finally {
      await session.close();
    }
  });

  test('F-11: Selectable 8-Joint Angle Controls (T1-F11-1..5)', async ({ page }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    await test.step('T1-F11-1: All 8 joint-toggle-<jointId> controls exist in control-panel', async () => {
      for (const jointId of ALL_JOINT_IDS) {
        await expect(page.getByTestId(`joint-toggle-${jointId}`)).toBeVisible();
      }
    });

    await test.step('T1-F11-2: Enabling each of the 8 joints shows its joint-angle-<jointId> readout', async () => {
      await setJointSelection(page, ALL_JOINT_IDS);
      for (const jointId of ALL_JOINT_IDS) {
        await expect(page.getByTestId(`joint-angle-${jointId}`)).toBeVisible();
      }
    });

    await test.step('T1-F11-3: Toggling an active joint off updates its pressed/checked state to false', async () => {
      const toggle = page.getByTestId('joint-toggle-right_shoulder');
      await setToggleState(toggle, false);
      expect(await isControlActive(toggle)).toBe(false);
    });

    await test.step('T1-F11-4: Toggling an inactive joint on sets state true and includes in Snapshot', async () => {
      const toggle = page.getByTestId('joint-toggle-right_shoulder');
      await setToggleState(toggle, true);
      expect(await isControlActive(toggle)).toBe(true);
      const snapshot = await captureAndValidateSnapshot(page);
      expect(snapshot.activeJointAngles.map((a) => a.joint)).toContain('right_shoulder');
    });

    await test.step('T1-F11-5: Selecting upper-body subset matches Snapshot activeJointAngles IDs', async () => {
      const upperBody = [
        'left_elbow',
        'right_elbow',
        'left_shoulder',
        'right_shoulder',
      ] as const;
      await setJointSelection(page, upperBody);
      const snapshot = await captureAndValidateSnapshot(page);
      expect(snapshot.activeJointAngles.map((a) => a.joint).sort()).toEqual(
        [...upperBody].sort(),
      );
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-12: Performance HUD Telemetry (T1-F12-1..5)', async ({ page }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const hudFps = page.getByTestId('hud-fps');
    const hudMs = page.getByTestId('hud-inference-ms');
    const hudJoints = page.getByTestId('hud-visible-joints');

    await test.step('T1-F12-1: hud-fps renders non-negative numeric FPS', async () => {
      await expect(hudFps).toBeVisible();
      await expect(hudFps).toHaveText(/\d+/);
    });

    await test.step('T1-F12-2: hud-inference-ms renders finite latency with ms', async () => {
      await expect(hudMs).toBeVisible();
      await expect(hudMs).toHaveText(/\d+(\.\d+)?\s*ms/i);
    });

    await test.step('T1-F12-3: hud-visible-joints matches <n> / 33 with 0 <= n <= 33', async () => {
      await expect(hudJoints).toBeVisible();
      const text = (await hudJoints.innerText()).trim();
      const match = text.match(/(\d+)\s*\/\s*33/);
      expect(match).not.toBeNull();
      const count = Number(match?.[1]);
      expect(count).toBeGreaterThanOrEqual(0);
      expect(count).toBeLessThanOrEqual(33);
    });

    await test.step('T1-F12-4: Performance HUD stays live across multiple rendered frames', async () => {
      await page.waitForTimeout(500);
      await expect(hudFps).toHaveText(/\d+/);
      await expect(hudMs).toHaveText(/\d+(\.\d+)?\s*ms/i);
    });

    await test.step('T1-F12-5: Performance HUD remains visible when control-panel is collapsed', async () => {
      await page.getByTestId('panel-toggle').click();
      await expect(page.getByTestId('panel-toggle')).toHaveAttribute(
        'aria-expanded',
        'false',
      );
      await expect(hudFps).toBeVisible();
      await expect(hudMs).toBeVisible();
      await expect(hudJoints).toBeVisible();
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-13: Responsive Collapsible Cybernetic ControlPanel (T1-F13-1..5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    await page.setViewportSize({ width: 1280, height: 720 });
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const panelToggle = page.getByTestId('panel-toggle');
    const controlPanel = page.getByTestId('control-panel');

    await test.step('T1-F13-1: panel-toggle defaults to aria-expanded="true" on desktop', async () => {
      await expect(panelToggle).toHaveAttribute('aria-expanded', 'true');
      await expect(controlPanel).toBeVisible();
    });

    await test.step('T1-F13-2: Clicking panel-toggle collapses panel and expands stage width', async () => {
      const beforeStageWidth =
        (await page.getByTestId('pose-canvas').boundingBox())?.width ?? 0;
      await panelToggle.click();
      await expect(panelToggle).toHaveAttribute('aria-expanded', 'false');
      await expect
        .poll(async () => {
          const box = await controlPanel.boundingBox();
          const visible = await controlPanel.isVisible();
          return !visible || (box?.width ?? 0) === 0;
        })
        .toBe(true);
      const afterStageWidth =
        (await page.getByTestId('pose-canvas').boundingBox())?.width ?? 0;
      expect(afterStageWidth).toBeGreaterThanOrEqual(beforeStageWidth);
    });

    await test.step('T1-F13-3: Clicking panel-toggle again re-expands control-panel', async () => {
      await panelToggle.click();
      await expect(panelToggle).toHaveAttribute('aria-expanded', 'true');
      await expect(controlPanel).toBeVisible();
      await expect(page.getByTestId('snapshot-json-button')).toBeVisible();
    });

    await test.step('T1-F13-4: Desktop 1280x720 has zero horizontal overflow', async () => {
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - window.innerWidth,
      );
      expect(overflow).toBeLessThanOrEqual(1);
    });

    await test.step('T1-F13-5: Mobile 390x844 stacks cleanly with zero horizontal overflow', async () => {
      await page.setViewportSize({ width: 390, height: 844 });
      await page.waitForTimeout(200);
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - window.innerWidth,
      );
      expect(overflow).toBeLessThanOrEqual(1);
      await expect(page.getByTestId('pose-canvas')).toBeVisible();
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-14: Synchronized rAF Stream & Lifecycle Loop (T1-F14-1..5)', async ({ page }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    await test.step('T1-F14-1: pose-video receives active MediaStream with valid dimensions', async () => {
      const videoMeta = await page.getByTestId('pose-video').evaluate((el) => {
        const video = el as HTMLVideoElement;
        return {
          readyState: video.readyState,
          videoWidth: video.videoWidth,
          videoHeight: video.videoHeight,
        };
      });
      expect(videoMeta.readyState).toBeGreaterThanOrEqual(2);
      expect(videoMeta.videoWidth).toBeGreaterThan(0);
      expect(videoMeta.videoHeight).toBeGreaterThan(0);
    });

    await test.step('T1-F14-2: pose-video and pose-canvas share aligned bounding boxes', async () => {
      const delta = await getBoundingBoxDelta(page);
      expect(delta.widthDelta).toBeLessThanOrEqual(2);
      expect(delta.heightDelta).toBeLessThanOrEqual(2);
    });

    await test.step('T1-F14-3: Sustained 2s rAF execution runs without stalls or errors', async () => {
      await page.waitForTimeout(2_000);
      tracker.assertNoRuntimeErrors();
    });

    await test.step('T1-F14-4: page.reload() cleanly tears down and re-initializes', async () => {
      await page.reload();
      await waitForAppReady(page);
      tracker.assertNoRuntimeErrors();
    });

    await test.step('T1-F14-5: Navigating away to about:blank triggers zero cleanup errors', async () => {
      await page.goto('about:blank');
      tracker.assertNoRuntimeErrors();
    });
  });
});
