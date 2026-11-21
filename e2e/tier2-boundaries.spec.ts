import { expect, test } from '@playwright/test';
import {
  ALL_JOINT_IDS,
  attachRuntimeErrorTracker,
  captureAndValidateSnapshot,
  clearMediaErrorMock,
  enableTaskModelCache,
  getBoundingBoxDelta,
  getElementFlipState,
  gotoWithErrorTracking,
  installDelayedMediaMock,
  installMediaErrorMock,
  installMissingMediaDevicesMock,
  isControlActive,
  isVideoVisuallyHidden,
  sampleCanvasBackdrop,
  setJointSelection,
  setMediaErrorMock,
  setRangeSliderValue,
  setToggleState,
  switchModelAndWait,
  waitForAppReady,
  waitForStatusOverlay,
} from './helpers';

test.describe('Tier 2 — Boundary & Corner Cases (F-01 .. F-14, 70 Cases)', () => {
  test('F-01 Boundaries: Deep Link, Reduced Motion, Blocked Favicon & Double Reload (T2-F01-1, T2-F01-3, T2-F01-4, T2-F01-5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.route('**/favicon.ico', (route) => route.abort());

    const tracker = attachRuntimeErrorTracker(page);
    const res = await page.goto('/?debug=1');

    await test.step('T2-F01-1: Deep-link query params boot cleanly with HTTP 200', async () => {
      expect(res?.status()).toBe(200);
      await waitForAppReady(page);
    });

    await test.step('T2-F01-3: Reduced-motion media preference loads without UI breakage', async () => {
      await expect(page.getByTestId('pose-canvas')).toBeVisible();
      await expect(page.getByTestId('control-panel')).toBeVisible();
    });

    await test.step('T2-F01-4: Blocked unrelated favicon request does not crash app mount', async () => {
      await expect(page.getByTestId('hud-fps')).toBeVisible();
      tracker.assertNoPageErrors();
    });

    await test.step('T2-F01-5: Double-reload stress leaves clean WASM & zero errors', async () => {
      tracker.clear();
      await page.reload();
      await waitForAppReady(page);
      await page.reload();
      await waitForAppReady(page);
      tracker.assertNoRuntimeErrors();
    });
  });

  test('F-01 & F-08 HiDPI Boundary: deviceScaleFactor 2 Context (T2-F01-2, T2-F08-4)', async ({
    browser,
  }) => {
    const context = await browser.newContext({
      deviceScaleFactor: 2,
      viewport: { width: 1280, height: 720 },
      permissions: ['camera'],
    });
    const page = await context.newPage();
    try {
      await enableTaskModelCache(page);
      const tracker = await gotoWithErrorTracking(page);
      await waitForAppReady(page);

      await test.step('T2-F01-2: High-DPI context renders without layout clipping or errors', async () => {
        const overflow = await page.evaluate(
          () => document.documentElement.scrollWidth - window.innerWidth,
        );
        expect(overflow).toBeLessThanOrEqual(1);
        await expect(page.getByTestId('pose-canvas')).toBeVisible();
      });

      await test.step('T2-F08-4: Canvas drawing buffer scales for deviceScaleFactor 2', async () => {
        const dims = await page.getByTestId('pose-canvas').evaluate((el) => {
          const canvas = el as HTMLCanvasElement;
          return {
            width: canvas.width,
            height: canvas.height,
            clientWidth: canvas.clientWidth,
            clientHeight: canvas.clientHeight,
          };
        });
        expect(dims.width).toBeGreaterThanOrEqual(dims.clientWidth * 1.5);
        expect(dims.height).toBeGreaterThanOrEqual(dims.clientHeight * 1.5);
      });

      tracker.assertNoRuntimeErrors();
    } finally {
      await context.close();
    }
  });

  test('F-02 Boundaries: Zero-Pose, 0 vs 8 Joints & Rapid Joint Toggling (T2-F02-1..5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    await test.step('T2-F02-1: Synthetic zero-pose stream yields zero NaN strings in HUD/panel', async () => {
      const panelText = await page.getByTestId('control-panel').innerText();
      const fpsText = await page.getByTestId('hud-fps').innerText();
      expect(panelText).not.toContain('NaN');
      expect(fpsText).not.toContain('NaN');
    });

    await test.step('T2-F02-2: Snapshot with poseDetected=false preserves valid null or numeric degrees', async () => {
      await setJointSelection(page, ['left_elbow', 'right_knee']);
      const snap = await captureAndValidateSnapshot(page);
      if (!snap.poseDetected) {
        for (const angle of snap.activeJointAngles) {
          expect(angle.degrees).toBeNull();
        }
      }
    });

    await test.step('T2-F02-3: 0 selected joints exports activeJointAngles: []', async () => {
      await setJointSelection(page, []);
      const snap = await captureAndValidateSnapshot(page);
      expect(snap.activeJointAngles).toEqual([]);
    });

    await test.step('T2-F02-4: All 8 selected joints export 8 unique joint entries', async () => {
      await setJointSelection(page, ALL_JOINT_IDS);
      const snap = await captureAndValidateSnapshot(page);
      expect(snap.activeJointAngles).toHaveLength(8);
      const unique = new Set(snap.activeJointAngles.map((a) => a.joint));
      expect(unique.size).toBe(8);
    });

    await test.step('T2-F02-5: Rapidly toggling a joint 10 times produces zero duplicates', async () => {
      const toggle = page.getByTestId('joint-toggle-left_elbow');
      for (let i = 0; i < 10; i += 1) {
        await toggle.click();
      }
      const snap = await captureAndValidateSnapshot(page);
      const ids = snap.activeJointAngles.map((a) => a.joint);
      expect(new Set(ids).size).toBe(ids.length);
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-03 Boundaries: Smoothing Clamping, Rapid Toggles & Preserved Strength (T2-F03-1..5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const toggle = page.getByTestId('smoothing-toggle');
    const slider = page.getByTestId('slider-smoothing-strength');

    await test.step('T2-F03-1: Smoothing strength minimum boundary 0 exports as 0', async () => {
      await setToggleState(toggle, true);
      await setRangeSliderValue(slider, '0');
      const snap = await captureAndValidateSnapshot(page);
      expect(snap.smoothing.strength).toBe(0);
    });

    await test.step('T2-F03-2: Smoothing strength maximum boundary 1 exports as 1', async () => {
      await setRangeSliderValue(slider, '1');
      const snap = await captureAndValidateSnapshot(page);
      expect(snap.smoothing.strength).toBe(1);
    });

    await test.step('T2-F03-3: Out-of-range DOM inputs (-0.5 and 1.5) clamp to [0, 1]', async () => {
      const clampedLow = await slider.evaluate((el) => {
        const input = el as HTMLInputElement;
        const setter = Object.getOwnPropertyDescriptor(
          HTMLInputElement.prototype,
          'value',
        )?.set;
        setter?.call(input, '-0.5');
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
        return Number(input.value);
      });
      expect(clampedLow).toBe(0);

      const clampedHigh = await slider.evaluate((el) => {
        const input = el as HTMLInputElement;
        const setter = Object.getOwnPropertyDescriptor(
          HTMLInputElement.prototype,
          'value',
        )?.set;
        setter?.call(input, '1.5');
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
        return Number(input.value);
      });
      expect(clampedHigh).toBe(1);
    });

    await test.step('T2-F03-4: 6 rapid smoothing-toggle clicks keep UI and Snapshot synchronized', async () => {
      for (let i = 0; i < 6; i += 1) {
        await toggle.click();
      }
      const active = await isControlActive(toggle);
      const snap = await captureAndValidateSnapshot(page);
      expect(snap.smoothing.enabled).toBe(active);
    });

    await test.step('T2-F03-5: Adjusting strength while smoothing is disabled preserves strength in JSON', async () => {
      await setToggleState(toggle, false);
      if (await slider.isEnabled()) {
        await setRangeSliderValue(slider, '0.35');
        const snap = await captureAndValidateSnapshot(page);
        expect(snap.smoothing.enabled).toBe(false);
        expect(snap.smoothing.strength).toBeCloseTo(0.35, 2);
      } else {
        await setToggleState(toggle, true);
        await setRangeSliderValue(slider, '0.35');
        await setToggleState(toggle, false);
        const snap = await captureAndValidateSnapshot(page);
        expect(snap.smoothing.enabled).toBe(false);
        expect(snap.smoothing.strength).toBeCloseTo(0.35, 2);
      }
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-04 Boundaries: Zero-Pose Fallback, Rapid Double Export & Filename Hygiene (T2-F04-1..5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    await test.step('T2-F04-1: Immediate zero-pose snapshot exports 33 finite landmarks', async () => {
      const snap = await captureAndValidateSnapshot(page);
      expect(snap.landmarks).toHaveLength(33);
      if (!snap.poseDetected) {
        for (const lm of snap.landmarks) {
          expect(lm.x).toBe(0);
          expect(lm.y).toBe(0);
          expect(lm.z).toBe(0);
          expect(lm.visibility).toBe(0);
        }
      }
    });

    await test.step('T2-F04-2: Two rapid consecutive downloads both succeed cleanly', async () => {
      const first = await captureAndValidateSnapshot(page);
      await page.waitForTimeout(100);
      const second = await captureAndValidateSnapshot(page);
      expect(second.timestampMs).toBeGreaterThanOrEqual(first.timestampMs);
    });

    await test.step('T2-F04-3: All 33 landmarks have sequential indices 0..32 and non-empty names', async () => {
      const snap = await captureAndValidateSnapshot(page);
      snap.landmarks.forEach((lm, idx) => {
        expect(lm.index).toBe(idx);
        expect(lm.name.trim().length).toBeGreaterThan(0);
      });
    });

    await test.step('T2-F04-4: Exporting with 0 selected joints still outputs 33 landmarks and empty activeJointAngles', async () => {
      await setJointSelection(page, []);
      const snap = await captureAndValidateSnapshot(page);
      expect(snap.landmarks).toHaveLength(33);
      expect(snap.activeJointAngles).toEqual([]);
    });

    await test.step('T2-F04-5: Snapshot filename contains no spaces or illegal path characters', async () => {
      const downloadPromise = page.waitForEvent('download');
      await page.getByTestId('snapshot-json-button').click();
      const download = await downloadPromise;
      const filename = download.suggestedFilename();
      expect(filename).toMatch(/^pose-snapshot-[^\s/\\:]+\.json$/);
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-05 Boundaries (Part 1): Rapid Model Race Guard & Active No-Op (T2-F05-1, T2-F05-2)', async ({
    page,
  }) => {
    await enableTaskModelCache(page, { delayByModel: { heavy: 400 } });
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    await test.step('T2-F05-2: Clicking already-active Lite button triggers no duplicate .task fetch', async () => {
      let taskFetches = 0;
      const onRequest = (req: { url: () => string }) => {
        if (req.url().includes('.task')) {
          taskFetches += 1;
        }
      };
      page.on('request', onRequest);
      try {
        await page.getByTestId('model-lite').click();
        await page.waitForTimeout(200);
        expect(taskFetches).toBe(0);
      } finally {
        page.off('request', onRequest);
      }
    });

    await test.step('T2-F05-1: Rapid Lite -> Heavy -> Lite settles cleanly on Lite', async () => {
      await page.getByTestId('model-heavy').click();
      await page.waitForTimeout(30);
      await page.getByTestId('model-lite').click();
      await expect
        .poll(() => isControlActive(page.getByTestId('model-lite')), { timeout: 30_000 })
        .toBe(true);
      await expect(page.getByTestId('status-overlay')).toBeHidden({ timeout: 30_000 });
      await page.waitForTimeout(550);
      expect(await isControlActive(page.getByTestId('model-lite'))).toBe(true);
      await expect(page.getByTestId('status-overlay')).toBeHidden();
      const snap = await captureAndValidateSnapshot(page);
      expect(snap.model).toBe('lite');
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-05 Boundaries (Part 2): Slow Model Loading Overlay (T2-F05-4)', async ({
    page,
  }) => {
    await enableTaskModelCache(page, { delayByModel: { lite: 600 } });
    const tracker = await gotoWithErrorTracking(page);

    await test.step('T2-F05-4: Slow .task fetch keeps status-overlay visible until model resolves', async () => {
      await waitForStatusOverlay(page, 10_000);
      await waitForAppReady(page, 45_000);
      await expect(page.getByTestId('status-overlay')).toBeHidden();
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-05 Boundaries (Part 3): Model HTTP 500 Error & Retry Recovery (T2-F05-3, T2-F05-5)', async ({
    page,
  }) => {
    const taskCache = await enableTaskModelCache(page, { failModel: 'lite' });
    const tracker = await gotoWithErrorTracking(page);

    await test.step('T2-F05-3: Model HTTP 500 transitions cleanly to status-overlay + retry-button', async () => {
      await waitForStatusOverlay(page, 25_000);
      await expect(page.getByTestId('retry-button')).toBeVisible({ timeout: 25_000 });
      await expect(page.getByTestId('status-overlay')).toContainText(/\S{4,}/);
      tracker.assertNoPageErrors();
    });

    await test.step('T2-F05-5: Clicking retry-button recovers cleanly when .task fetch succeeds', async () => {
      taskCache.setFailModel(null);
      await page.getByTestId('retry-button').click();
      await waitForAppReady(page, 45_000);
      await expect(page.getByTestId('status-overlay')).toBeHidden();
      tracker.assertNoPageErrors();
    });
  });

  test('F-06 Boundaries: Confidence Slider Bounds & Rapid Sweeping (T2-F06-1..5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const det = page.getByTestId('slider-detection-confidence');
    const track = page.getByTestId('slider-tracking-confidence');

    await test.step('T2-F06-1: Detection confidence min boundary 0.1', async () => {
      await setRangeSliderValue(det, '0.1');
      await expect(det).toHaveValue('0.1');
    });

    await test.step('T2-F06-2: Detection confidence max boundary 0.95', async () => {
      await setRangeSliderValue(det, '0.95');
      await expect(det).toHaveValue('0.95');
    });

    await test.step('T2-F06-3: Tracking confidence min boundary 0.1', async () => {
      await setRangeSliderValue(track, '0.1');
      await expect(track).toHaveValue('0.1');
    });

    await test.step('T2-F06-4: Tracking confidence max boundary 0.95', async () => {
      await setRangeSliderValue(track, '0.95');
      await expect(track).toHaveValue('0.95');
    });

    await test.step('T2-F06-5: Rapid step sweeping of both sliders triggers zero exceptions', async () => {
      for (const val of ['0.1', '0.5', '0.95', '0.55']) {
        await setRangeSliderValue(det, val);
        await setRangeSliderValue(track, val);
      }
      await expect(det).toHaveValue('0.55');
      await expect(track).toHaveValue('0.55');
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-07 Boundaries: Mode Idempotency, Rapid Toggling & DOM Retention (T2-F07-1..5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const overlayBtn = page.getByTestId('mode-overlay');
    const blackoutBtn = page.getByTestId('mode-blackout');

    await test.step('T2-F07-1: Clicking mode-overlay when already in Overlay is idempotent', async () => {
      await overlayBtn.click();
      expect(await isControlActive(overlayBtn)).toBe(true);
      expect(await isVideoVisuallyHidden(page)).toBe(false);
    });

    await test.step('T2-F07-2: Clicking mode-blackout twice is idempotent', async () => {
      await blackoutBtn.click();
      await blackoutBtn.click();
      await expect.poll(() => isControlActive(blackoutBtn)).toBe(true);
      await expect.poll(() => isVideoVisuallyHidden(page)).toBe(true);
    });

    await test.step('T2-F07-3: Rapidly toggling Overlay <-> Blackout 8 times ends consistent', async () => {
      for (let i = 0; i < 8; i += 1) {
        await (i % 2 === 0 ? overlayBtn : blackoutBtn).click();
      }
      const isBlackout = await isControlActive(blackoutBtn);
      await expect.poll(() => isVideoVisuallyHidden(page)).toBe(isBlackout);
    });

    await test.step('T2-F07-4: Blackout grid covers full resized canvas after viewport resize', async () => {
      await blackoutBtn.click();
      await page.setViewportSize({ width: 960, height: 640 });
      await expect
        .poll(async () => (await sampleCanvasBackdrop(page)).opaqueDarkPixels)
        .toBeGreaterThanOrEqual(15);
    });

    await test.step('T2-F07-5: pose-video remains mounted in DOM during Blackout mode', async () => {
      await expect(page.getByTestId('pose-video')).toHaveCount(1);
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-08 Boundaries: Dynamic Viewport Resize, Panel Collapse & Clean Frame (T2-F08-1, T2-F08-2, T2-F08-3, T2-F08-5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    await test.step('T2-F08-1: Shrinking viewport to 640x480 updates pose-canvas without collapse', async () => {
      await page.setViewportSize({ width: 640, height: 480 });
      await page.waitForTimeout(150);
      const box = await page.getByTestId('pose-canvas').boundingBox();
      expect(box?.width ?? 0).toBeGreaterThan(100);
      expect(box?.height ?? 0).toBeGreaterThan(100);
    });

    await test.step('T2-F08-2: Expanding viewport to 1920x1080 scales stage cleanly', async () => {
      await page.setViewportSize({ width: 1920, height: 1080 });
      await page.waitForTimeout(150);
      const box = await page.getByTestId('pose-canvas').boundingBox();
      expect(box?.width ?? 0).toBeGreaterThan(400);
      expect(box?.height ?? 0).toBeGreaterThan(300);
    });

    await test.step('T2-F08-3: Collapsing panel keeps Blackout grid rendering cleanly', async () => {
      await page.getByTestId('mode-blackout').click();
      await page.getByTestId('panel-toggle').click();
      await expect
        .poll(async () => (await sampleCanvasBackdrop(page)).opaqueDarkPixels)
        .toBeGreaterThanOrEqual(15);
    });

    await test.step('T2-F08-5: Zero-pose frames in Overlay mode keep transparent backdrop', async () => {
      await page.getByTestId('panel-toggle').click();
      await page.getByTestId('mode-overlay').click();
      const visibleText = await page.getByTestId('hud-visible-joints').innerText();
      if (visibleText.trim().startsWith('0')) {
        await expect
          .poll(async () => (await sampleCanvasBackdrop(page)).opaqueDarkPixels)
          .toBe(0);
      }
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-09 Boundaries: Rapid Mirror Toggles, Keyboard & Panel Independence (T2-F09-1..5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const mirrorToggle = page.getByTestId('mirror-toggle');
    const video = page.getByTestId('pose-video');
    const canvas = page.getByTestId('pose-canvas');

    await setToggleState(mirrorToggle, false);

    await test.step('T2-F09-1: 7 rapid clicks leave video mirrored and canvas unflipped', async () => {
      for (let i = 0; i < 7; i += 1) {
        await mirrorToggle.click();
      }
      await expect
        .poll(async () => (await getElementFlipState(video)).isMirroredX)
        .toBe(true);
      const canvasFlip = await getElementFlipState(canvas);
      expect(canvasFlip.isUnflipped).toBe(true);
    });

    await test.step('T2-F09-2: 8 more clicks restore unmirrored video and unflipped canvas', async () => {
      await setToggleState(mirrorToggle, false);
      for (let i = 0; i < 8; i += 1) {
        await mirrorToggle.click();
      }
      await expect
        .poll(async () => (await getElementFlipState(video)).isUnflipped)
        .toBe(true);
      const canvasFlip = await getElementFlipState(canvas);
      expect(canvasFlip.isUnflipped).toBe(true);
    });

    await test.step('T2-F09-3: Keyboard Space toggles focused mirror-toggle', async () => {
      const before = await isControlActive(mirrorToggle);
      await mirrorToggle.focus();
      await page.keyboard.press('Space');
      await expect.poll(() => isControlActive(mirrorToggle)).toBe(!before);
    });

    await test.step('T2-F09-4: Collapsing panel preserves mirrored video & upright canvas', async () => {
      await setToggleState(mirrorToggle, true);
      await page.getByTestId('panel-toggle').click();
      expect((await getElementFlipState(video)).isMirroredX).toBe(true);
      expect((await getElementFlipState(canvas)).isUnflipped).toBe(true);
      await page.getByTestId('panel-toggle').click();
    });

    await test.step('T2-F09-5: Snapshot JSON reflects exact final mirrored boolean', async () => {
      const active = await isControlActive(mirrorToggle);
      const snap = await captureAndValidateSnapshot(page);
      expect(snap.mirrored).toBe(active);
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-10 Boundaries (Part 1): NotReadableError (T2-F10-1)', async ({ page }) => {
    await enableTaskModelCache(page);
    await installMediaErrorMock(page, {
      name: 'NotReadableError',
      message: 'Could not start video source',
    });
    const tracker = await gotoWithErrorTracking(page);

    await test.step('T2-F10-1: NotReadableError displays descriptive error screen + retry-button', async () => {
      const overlay = await waitForStatusOverlay(page);
      await expect(page.getByTestId('retry-button')).toBeVisible();
      expect((await overlay.innerText()).trim().length).toBeGreaterThan(5);
      tracker.assertNoPageErrors();
    });
  });

  test('F-10 Boundaries (Part 2): OverconstrainedError (T2-F10-2)', async ({ page }) => {
    await enableTaskModelCache(page);
    await installMediaErrorMock(page, {
      name: 'OverconstrainedError',
      message: 'Constraints cannot be satisfied',
    });
    const tracker = await gotoWithErrorTracking(page);

    await test.step('T2-F10-2: OverconstrainedError displays descriptive error screen + retry-button', async () => {
      const overlay = await waitForStatusOverlay(page);
      await expect(page.getByTestId('retry-button')).toBeVisible();
      expect((await overlay.innerText()).trim().length).toBeGreaterThan(5);
      tracker.assertNoPageErrors();
    });
  });

  test('F-10 Boundaries (Part 3): Missing navigator.mediaDevices (T2-F10-3)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    await installMissingMediaDevicesMock(page);
    const tracker = await gotoWithErrorTracking(page);

    await test.step('T2-F10-3: Undefined navigator.mediaDevices renders graceful error overlay', async () => {
      const overlay = await waitForStatusOverlay(page);
      await expect(page.getByTestId('retry-button')).toBeVisible();
      expect((await overlay.innerText()).trim().length).toBeGreaterThan(5);
      tracker.assertNoPageErrors();
    });
  });

  test('F-10 Boundaries (Part 4): Repeated Failed Retries & NotFoundError Recovery (T2-F10-4, T2-F10-5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    await installMediaErrorMock(page, {
      name: 'NotAllowedError',
      message: 'Permission denied',
    });
    const tracker = await gotoWithErrorTracking(page);

    await test.step('T2-F10-4: Multiple consecutive failed retry clicks remain stable without crash', async () => {
      await waitForStatusOverlay(page);
      const retry = page.getByTestId('retry-button');
      await retry.click();
      await expect(retry).toBeVisible();
      await retry.click();
      await expect(page.getByTestId('status-overlay')).toBeVisible();
      tracker.assertNoPageErrors();
    });

    await test.step('T2-F10-5: Switching to NotFoundError then clearing on Retry transitions to ready', async () => {
      await setMediaErrorMock(page, {
        name: 'NotFoundError',
        message: 'Requested device not found',
        emptyDevices: true,
      });
      await page.getByTestId('retry-button').click();
      const overlay = await waitForStatusOverlay(page);
      await expect(page.getByTestId('retry-button')).toBeVisible();
      expect((await overlay.innerText()).trim().length).toBeGreaterThan(5);

      await clearMediaErrorMock(page);
      await page.getByTestId('retry-button').click();
      await waitForAppReady(page);
      await expect(page.getByTestId('status-overlay')).toBeHidden();
      await expect(page.getByTestId('retry-button')).toBeHidden();
      tracker.assertNoPageErrors();
    });
  });

  test('F-11 Boundaries: 0/8 Joints, Left-Only, Right-Only & Keyboard Toggle (T2-F11-1..5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    await test.step('T2-F11-1: Deselecting all 8 joints leaves UI stable with 0 active joints', async () => {
      await setJointSelection(page, []);
      const snap = await captureAndValidateSnapshot(page);
      expect(snap.activeJointAngles).toHaveLength(0);
    });

    await test.step('T2-F11-2: Selecting all 8 joints displays all 8 readouts cleanly', async () => {
      await setJointSelection(page, ALL_JOINT_IDS);
      for (const id of ALL_JOINT_IDS) {
        await expect(page.getByTestId(`joint-angle-${id}`)).toBeVisible();
      }
    });

    await test.step('T2-F11-3: Left-only joints subset exports exact 4 left IDs', async () => {
      const leftOnly = [
        'left_elbow',
        'left_shoulder',
        'left_hip',
        'left_knee',
      ] as const;
      await setJointSelection(page, leftOnly);
      const snap = await captureAndValidateSnapshot(page);
      expect(snap.activeJointAngles.map((a) => a.joint).sort()).toEqual(
        [...leftOnly].sort(),
      );
    });

    await test.step('T2-F11-4: Right-only joints subset exports exact 4 right IDs', async () => {
      const rightOnly = [
        'right_elbow',
        'right_shoulder',
        'right_hip',
        'right_knee',
      ] as const;
      await setJointSelection(page, rightOnly);
      const snap = await captureAndValidateSnapshot(page);
      expect(snap.activeJointAngles.map((a) => a.joint).sort()).toEqual(
        [...rightOnly].sort(),
      );
    });

    await test.step('T2-F11-5: Keyboard Space on joint-toggle-left_elbow toggles selection', async () => {
      const toggle = page.getByTestId('joint-toggle-left_elbow');
      const before = await isControlActive(toggle);
      await toggle.focus();
      await page.keyboard.press('Space');
      await expect.poll(() => isControlActive(toggle)).toBe(!before);
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-12 Boundaries: Non-NaN Telemetry, Range Bounds & Mobile/Blackout HUD Visibility (T2-F12-1..5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    await test.step('T2-F12-1: hud-fps never displays NaN or negative values', async () => {
      const text = (await page.getByTestId('hud-fps').innerText()).trim();
      expect(text).not.toContain('NaN');
      const fps = Number.parseFloat(text.replace(/[^\d.-]/g, ''));
      expect(fps).toBeGreaterThanOrEqual(0);
    });

    await test.step('T2-F12-2: hud-inference-ms never shows negative or Infinity', async () => {
      const text = (await page.getByTestId('hud-inference-ms').innerText()).trim();
      expect(text).not.toContain('Infinity');
      expect(text).not.toContain('NaN');
      const ms = Number.parseFloat(text.replace(/[^\d.-]/g, ''));
      expect(ms).toBeGreaterThanOrEqual(0);
    });

    await test.step('T2-F12-3: hud-visible-joints stays within 0..33 across model switch', async () => {
      await switchModelAndWait(page, 'full');
      const text = (await page.getByTestId('hud-visible-joints').innerText()).trim();
      const match = text.match(/(\d+)\s*\/\s*33/);
      const count = Number(match?.[1] ?? -1);
      expect(count).toBeGreaterThanOrEqual(0);
      expect(count).toBeLessThanOrEqual(33);
    });

    await test.step('T2-F12-4: At mobile 390x844, HUD metrics remain visible alongside panel-toggle', async () => {
      await page.setViewportSize({ width: 390, height: 844 });
      await expect(page.getByTestId('hud-fps')).toBeVisible();
      await expect(page.getByTestId('hud-inference-ms')).toBeVisible();
      await expect(page.getByTestId('hud-visible-joints')).toBeVisible();
      await expect(page.getByTestId('panel-toggle')).toBeVisible();
    });

    await test.step('T2-F12-5: During Blackout mode, Performance HUD remains visible', async () => {
      await page.getByTestId('mode-blackout').click();
      await expect(page.getByTestId('hud-fps')).toBeVisible();
      await expect(page.getByTestId('hud-inference-ms')).toBeVisible();
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-13 Boundaries: 320px Mobile, 768px Tablet, Rapid Collapse & Scrollability (T2-F13-1..5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    await test.step('T2-F13-1: Ultra-narrow 320x568 viewport has zero horizontal scroll', async () => {
      await page.setViewportSize({ width: 320, height: 568 });
      await page.waitForTimeout(150);
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - window.innerWidth,
      );
      expect(overflow).toBeLessThanOrEqual(1);
    });

    await test.step('T2-F13-2: Tablet portrait 768x1024 keeps stage and controls accessible', async () => {
      await page.setViewportSize({ width: 768, height: 1024 });
      await expect(page.getByTestId('pose-canvas')).toBeVisible();
      await expect(page.getByTestId('control-panel')).toBeVisible();
    });

    await test.step('T2-F13-3: 5 rapid panel-toggle clicks keep aria-expanded synced', async () => {
      const toggle = page.getByTestId('panel-toggle');
      const controlPanel = page.getByTestId('control-panel');
      if ((await toggle.getAttribute('aria-expanded')) !== 'true') {
        await toggle.click();
        await expect(toggle).toHaveAttribute('aria-expanded', 'true');
      }
      for (let i = 0; i < 5; i += 1) {
        await toggle.click();
      }
      await expect(toggle).toHaveAttribute('aria-expanded', 'false');
      await expect
        .poll(async () => {
          const visible = await controlPanel.isVisible();
          const box = await controlPanel.boundingBox();
          return !visible || (box?.width ?? 0) === 0 || (box?.height ?? 0) === 0;
        })
        .toBe(true);
    });

    await test.step('T2-F13-4: Resizing window preserves collapsed aria-expanded="false"', async () => {
      await page.setViewportSize({ width: 1280, height: 720 });
      await expect(page.getByTestId('panel-toggle')).toHaveAttribute(
        'aria-expanded',
        'false',
      );
    });

    await test.step('T2-F13-5: Constrained height 1024x600 keeps Snapshot button reachable', async () => {
      await page.getByTestId('panel-toggle').click();
      await page.setViewportSize({ width: 1024, height: 600 });
      const btn = page.getByTestId('snapshot-json-button');
      await btn.scrollIntoViewIfNeeded();
      await expect(btn).toBeVisible();
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-14 Boundaries (Part 1): Delayed getUserMedia Holds Loading State (T2-F14-1)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    await installDelayedMediaMock(page, 500);
    const tracker = await gotoWithErrorTracking(page);

    await test.step('T2-F14-1: 500ms delayed getUserMedia shows requesting-camera overlay then ready', async () => {
      await waitForStatusOverlay(page, 10_000);
      await waitForAppReady(page, 45_000);
    });

    tracker.assertNoRuntimeErrors();
  });

  test('F-14 Boundaries (Part 2): Fast Reload, Resize Alignment, Visibility Resume & 3s Stream (T2-F14-2..5)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = attachRuntimeErrorTracker(page);

    await test.step('T2-F14-2: Fast reload mid-initialization completes with zero uncaught rejections', async () => {
      await page.goto('/', { waitUntil: 'domcontentloaded' });
      await page.waitForTimeout(80);
      await page.reload();
      await waitForAppReady(page);
      tracker.assertNoRuntimeErrors();
    });

    await test.step('T2-F14-3: Viewport resize keeps pose-video and pose-canvas aligned <= 2px', async () => {
      await page.setViewportSize({ width: 960, height: 600 });
      await page.waitForTimeout(200);
      const delta = await getBoundingBoxDelta(page);
      expect(delta.widthDelta).toBeLessThanOrEqual(2);
      expect(delta.heightDelta).toBeLessThanOrEqual(2);
    });

    await test.step('T2-F14-4: Visibility change event resumes rAF loop with zero monotonic timestamp errors', async () => {
      await page.evaluate(async () => {
        try {
          Object.defineProperty(document, 'hidden', {
            configurable: true,
            get: () => true,
          });
          Object.defineProperty(document, 'visibilityState', {
            configurable: true,
            get: () => 'hidden',
          });
          document.dispatchEvent(new Event('visibilitychange'));
          await new Promise((resolve) => setTimeout(resolve, 150));
        } finally {
          Reflect.deleteProperty(document, 'hidden');
          Reflect.deleteProperty(document, 'visibilityState');
          document.dispatchEvent(new Event('visibilitychange'));
        }
      });
      await page.waitForTimeout(300);
      tracker.assertNoRuntimeErrors();
    });

    await test.step('T2-F14-5: Sustained 3-second idle stream keeps HUD healthy with zero errors', async () => {
      await page.waitForTimeout(3_000);
      await expect(page.getByTestId('hud-fps')).toHaveText(/\d+/);
      tracker.assertNoRuntimeErrors();
    });
  });
});
