import { expect, test } from '@playwright/test';
import {
  captureAndValidateSnapshot,
  clearMediaErrorMock,
  enableTaskModelCache,
  getBoundingBoxDelta,
  getElementFlipState,
  gotoWithErrorTracking,
  installMediaErrorMock,
  isControlActive,
  isVideoVisuallyHidden,
  sampleCanvasBackdrop,
  setJointSelection,
  setRangeSliderValue,
  setToggleState,
  switchModelAndWait,
  waitForAppReady,
  waitForStatusOverlay,
} from './helpers';

test.describe('Tier 3 — Cross-Feature Pairwise Combinations (T3-01 .. T3-14)', () => {
  test('T3-01: Blackout + Mirror ON keeps video hidden/flipped, canvas dark/unflipped, and exports mirrored: true', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    await page.getByTestId('mode-blackout').click();
    await setToggleState(page.getByTestId('mirror-toggle'), true);

    await expect.poll(() => isVideoVisuallyHidden(page)).toBe(true);
    const videoFlip = await getElementFlipState(page.getByTestId('pose-video'));
    expect(videoFlip.isMirroredX).toBe(true);

    const canvasFlip = await getElementFlipState(page.getByTestId('pose-canvas'));
    expect(canvasFlip.isUnflipped).toBe(true);
    expect(canvasFlip.isMirroredX).toBe(false);

    await expect
      .poll(async () => (await sampleCanvasBackdrop(page)).opaqueDarkPixels)
      .toBeGreaterThanOrEqual(15);

    const snapshot = await captureAndValidateSnapshot(page);
    expect(snapshot.mirrored).toBe(true);

    tracker.assertNoRuntimeErrors();
  });

  test('T3-02: Model Full + Smoothing OFF + Snapshot JSON alignment', async ({ page }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    await switchModelAndWait(page, 'full');
    await setToggleState(page.getByTestId('smoothing-toggle'), false);

    const snapshot = await captureAndValidateSnapshot(page);
    expect(snapshot.model).toBe('full');
    expect(snapshot.smoothing.enabled).toBe(false);
    expect(snapshot.landmarks).toHaveLength(33);

    tracker.assertNoRuntimeErrors();
  });

  test('T3-03: Panel Collapse + Blackout + Joint Subset preservation', async ({ page }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const subset = ['left_elbow', 'right_hip'] as const;
    await setJointSelection(page, subset);
    await page.getByTestId('mode-blackout').click();
    await page.getByTestId('panel-toggle').click();

    await expect(page.getByTestId('panel-toggle')).toHaveAttribute(
      'aria-expanded',
      'false',
    );
    await expect(page.getByTestId('hud-fps')).toBeVisible();
    await expect
      .poll(async () => (await sampleCanvasBackdrop(page)).opaqueDarkPixels)
      .toBeGreaterThanOrEqual(15);

    await page.getByTestId('panel-toggle').click();
    await expect(page.getByTestId('panel-toggle')).toHaveAttribute(
      'aria-expanded',
      'true',
    );
    for (const id of subset) {
      expect(await isControlActive(page.getByTestId(`joint-toggle-${id}`))).toBe(true);
    }

    const snapshot = await captureAndValidateSnapshot(page);
    expect(snapshot.activeJointAngles.map((a) => a.joint).sort()).toEqual(
      [...subset].sort(),
    );

    tracker.assertNoRuntimeErrors();
  });

  test('T3-04: Confidence Sliders persist across Model Switch Lite -> Full -> Lite', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const det = page.getByTestId('slider-detection-confidence');
    const track = page.getByTestId('slider-tracking-confidence');
    await setRangeSliderValue(det, '0.8');
    await setRangeSliderValue(track, '0.75');

    await switchModelAndWait(page, 'full');
    await expect(det).toHaveValue('0.8');
    await expect(track).toHaveValue('0.75');

    await switchModelAndWait(page, 'lite');
    await expect(det).toHaveValue('0.8');
    await expect(track).toHaveValue('0.75');

    tracker.assertNoRuntimeErrors();
  });

  test('T3-05: Error Recovery from NotAllowedError leaves Blackout, Mirror & HUD controls interactive', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    await installMediaErrorMock(page, {
      name: 'NotAllowedError',
      message: 'Permission denied',
    });
    const tracker = await gotoWithErrorTracking(page);

    await waitForStatusOverlay(page);
    await clearMediaErrorMock(page);
    await page.getByTestId('retry-button').click();
    await waitForAppReady(page);

    await page.getByTestId('mode-blackout').click();
    await setToggleState(page.getByTestId('mirror-toggle'), true);

    await expect.poll(() => isVideoVisuallyHidden(page)).toBe(true);
    expect((await getElementFlipState(page.getByTestId('pose-video'))).isMirroredX).toBe(
      true,
    );
    expect((await getElementFlipState(page.getByTestId('pose-canvas'))).isUnflipped).toBe(
      true,
    );
    await expect(page.getByTestId('hud-fps')).toHaveText(/\d+/);

    tracker.assertNoPageErrors();
  });

  test('T3-06: Max Smoothing (1.0) + Min Confidence (0.1) keeps Performance HUD responsive', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    await setRangeSliderValue(page.getByTestId('slider-detection-confidence'), '0.1');
    await setRangeSliderValue(page.getByTestId('slider-tracking-confidence'), '0.1');
    await setToggleState(page.getByTestId('smoothing-toggle'), true);
    await setRangeSliderValue(page.getByTestId('slider-smoothing-strength'), '1');

    await page.waitForTimeout(400);
    await expect(page.getByTestId('hud-fps')).toHaveText(/\d+/);
    await expect(page.getByTestId('hud-inference-ms')).toHaveText(/\d+(\.\d+)?\s*ms/i);

    tracker.assertNoRuntimeErrors();
  });

  test('T3-07: Lower-body 4-Joint Selection + Mirror ON + Snapshot JSON verification', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const lowerBody = ['left_hip', 'right_hip', 'left_knee', 'right_knee'] as const;
    await setJointSelection(page, lowerBody);
    await setToggleState(page.getByTestId('mirror-toggle'), true);

    const snapshot = await captureAndValidateSnapshot(page);
    expect(snapshot.mirrored).toBe(true);
    expect(snapshot.activeJointAngles.map((a) => a.joint).sort()).toEqual(
      [...lowerBody].sort(),
    );

    tracker.assertNoRuntimeErrors();
  });

  test('T3-08: Mobile Viewport (390x844) + Mirror ON + Video/Canvas Bounding Box Alignment', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    await page.setViewportSize({ width: 390, height: 844 });
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    await setToggleState(page.getByTestId('mirror-toggle'), true);

    const videoFlip = await getElementFlipState(page.getByTestId('pose-video'));
    const canvasFlip = await getElementFlipState(page.getByTestId('pose-canvas'));
    expect(videoFlip.isMirroredX).toBe(true);
    expect(canvasFlip.isUnflipped).toBe(true);

    const delta = await getBoundingBoxDelta(page);
    expect(delta.widthDelta).toBeLessThanOrEqual(2);
    expect(delta.heightDelta).toBeLessThanOrEqual(2);

    tracker.assertNoRuntimeErrors();
  });

  test('T3-09: Model Heavy + Snapshot JSON exports model: "heavy" and 33 landmarks', async ({
    page,
  }) => {
    test.setTimeout(150_000);
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    await switchModelAndWait(page, 'heavy', 90_000);
    const snapshot = await captureAndValidateSnapshot(page);
    expect(snapshot.model).toBe('heavy');
    expect(snapshot.landmarks).toHaveLength(33);

    tracker.assertNoRuntimeErrors();
  });

  test('T3-10: Overlay <-> Blackout across HiDPI Resize & Panel Toggle', async ({
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

      await page.getByTestId('mode-blackout').click();
      await page.getByTestId('panel-toggle').click();
      await page.setViewportSize({ width: 900, height: 600 });

      await expect
        .poll(async () => (await sampleCanvasBackdrop(page)).opaqueDarkPixels)
        .toBeGreaterThanOrEqual(15);

      await page.getByTestId('panel-toggle').click();
      await page.getByTestId('mode-overlay').click();
      await expect.poll(() => isVideoVisuallyHidden(page)).toBe(false);

      tracker.assertNoRuntimeErrors();
    } finally {
      await context.close();
    }
  });

  test('T3-11: Permission Denied on Mobile Viewport (390x844) keeps Retry reachable', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    await page.setViewportSize({ width: 390, height: 844 });
    await installMediaErrorMock(page, {
      name: 'NotAllowedError',
      message: 'Permission denied on mobile',
    });
    const tracker = await gotoWithErrorTracking(page);

    const overlay = await waitForStatusOverlay(page);
    const retry = page.getByTestId('retry-button');
    await expect(overlay).toBeVisible();
    await expect(retry).toBeVisible();

    const retryBox = await retry.boundingBox();
    expect(retryBox).not.toBeNull();
    expect(retryBox!.x).toBeGreaterThanOrEqual(0);
    expect(retryBox!.x + retryBox!.width).toBeLessThanOrEqual(390);
    expect(retryBox!.y + retryBox!.height).toBeLessThanOrEqual(844);

    tracker.assertNoPageErrors();
  });

  test('T3-12: Confidence 0.95 + Smoothing 0.0 Disabled + Snapshot JSON consistency', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    await setRangeSliderValue(page.getByTestId('slider-detection-confidence'), '0.95');
    await setRangeSliderValue(page.getByTestId('slider-tracking-confidence'), '0.95');
    await setToggleState(page.getByTestId('smoothing-toggle'), true);
    await setRangeSliderValue(page.getByTestId('slider-smoothing-strength'), '0');
    await setToggleState(page.getByTestId('smoothing-toggle'), false);

    const snapshot = await captureAndValidateSnapshot(page);
    expect(snapshot.smoothing.enabled).toBe(false);
    expect(snapshot.smoothing.strength).toBe(0);

    tracker.assertNoRuntimeErrors();
  });

  test('T3-13: Performance HUD formatting across Overlay/Blackout and Lite -> Full switch', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const assertHudFormatted = async () => {
      await expect(page.getByTestId('hud-fps')).toHaveText(/\d+/);
      await expect(page.getByTestId('hud-inference-ms')).toHaveText(/\d+(\.\d+)?\s*ms/i);
      await expect(page.getByTestId('hud-visible-joints')).toHaveText(/\d+\s*\/\s*33/);
    };

    await assertHudFormatted();
    await page.getByTestId('mode-blackout').click();
    await assertHudFormatted();
    await switchModelAndWait(page, 'full');
    await assertHudFormatted();

    tracker.assertNoRuntimeErrors();
  });

  test('T3-14: Collapse Panel + Resize + Re-expand preserves selected joints & angle formatting', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const selected = ['left_shoulder', 'right_elbow', 'left_knee'] as const;
    await setJointSelection(page, selected);

    await page.getByTestId('panel-toggle').click();
    await page.setViewportSize({ width: 820, height: 680 });
    await page.waitForTimeout(150);
    await page.getByTestId('panel-toggle').click();

    for (const id of selected) {
      expect(await isControlActive(page.getByTestId(`joint-toggle-${id}`))).toBe(true);
      const readout = (await page.getByTestId(`joint-angle-${id}`).innerText()).trim();
      expect(readout).toMatch(/^(\d+(\.\d+)?|--)\s*°$/);
    }

    tracker.assertNoRuntimeErrors();
  });
});
