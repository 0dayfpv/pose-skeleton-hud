import { expect, test } from '@playwright/test';
import {
  ALL_JOINT_IDS,
  captureAndValidateSnapshot,
  clearMediaErrorMock,
  enableTaskModelCache,
  getElementFlipState,
  gotoWithErrorTracking,
  installMediaErrorMock,
  isVideoVisuallyHidden,
  sampleCanvasBackdrop,
  setJointSelection,
  setRangeSliderValue,
  setToggleState,
  switchModelAndWait,
  waitForAppReady,
  waitForStatusOverlay,
} from './helpers';

test.describe('Tier 4 — Real-World Application Scenarios (T4-01 .. T4-07)', () => {
  test('T4-01: Full Biomechanics Lab Calibration & Snapshot Export', async ({ page }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    await expect(page.getByTestId('hud-fps')).toHaveText(/\d+/);

    await setRangeSliderValue(page.getByTestId('slider-detection-confidence'), '0.65');
    await setRangeSliderValue(page.getByTestId('slider-tracking-confidence'), '0.6');
    await setToggleState(page.getByTestId('smoothing-toggle'), true);
    await setRangeSliderValue(page.getByTestId('slider-smoothing-strength'), '0.7');

    const upperBody = [
      'left_elbow',
      'right_elbow',
      'left_shoulder',
      'right_shoulder',
    ] as const;
    await setJointSelection(page, upperBody);

    await setToggleState(page.getByTestId('mirror-toggle'), true);
    await page.getByTestId('mode-blackout').click();

    await expect.poll(() => isVideoVisuallyHidden(page)).toBe(true);
    expect((await getElementFlipState(page.getByTestId('pose-video'))).isMirroredX).toBe(
      true,
    );
    expect((await getElementFlipState(page.getByTestId('pose-canvas'))).isUnflipped).toBe(
      true,
    );

    const snapshot = await captureAndValidateSnapshot(page);
    expect(snapshot.mirrored).toBe(true);
    expect(snapshot.smoothing.enabled).toBe(true);
    expect(snapshot.smoothing.strength).toBeCloseTo(0.7, 2);
    expect(snapshot.activeJointAngles.map((a) => a.joint).sort()).toEqual(
      [...upperBody].sort(),
    );
    expect(snapshot.landmarks).toHaveLength(33);

    tracker.assertNoRuntimeErrors();
  });

  test('T4-02: Camera Permission Denied -> User Grants Permission -> Retry Recovery', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    await installMediaErrorMock(page, {
      name: 'NotAllowedError',
      message: 'User denied camera access',
    });
    const tracker = await gotoWithErrorTracking(page);

    const overlay = await waitForStatusOverlay(page);
    await expect(page.getByTestId('retry-button')).toBeVisible();
    expect((await overlay.innerText()).trim().length).toBeGreaterThan(5);

    await clearMediaErrorMock(page);
    await page.getByTestId('retry-button').click();
    await waitForAppReady(page);

    await expect(page.getByTestId('status-overlay')).toBeHidden();
    await expect(page.getByTestId('hud-fps')).toHaveText(/\d+/);

    const snapshot = await captureAndValidateSnapshot(page);
    expect(snapshot.landmarks).toHaveLength(33);

    tracker.assertNoPageErrors();
  });

  test('T4-03: Missing Webcam (NotFoundError) -> Plug-In -> Retry -> Full Pose Inspection', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    await installMediaErrorMock(page, {
      name: 'NotFoundError',
      message: 'No camera device detected',
      emptyDevices: true,
    });
    const tracker = await gotoWithErrorTracking(page);

    await waitForStatusOverlay(page);
    await expect(page.getByTestId('retry-button')).toBeVisible();

    await clearMediaErrorMock(page);
    await page.getByTestId('retry-button').click();
    await waitForAppReady(page);

    await page.getByTestId('mode-blackout').click();
    await setJointSelection(page, ALL_JOINT_IDS);
    await expect(page.getByTestId('hud-visible-joints')).toHaveText(/\d+\s*\/\s*33/);

    const snapshot = await captureAndValidateSnapshot(page);
    expect(snapshot.activeJointAngles).toHaveLength(8);

    tracker.assertNoPageErrors();
  });

  test('T4-04: Progressive Model Upgrade (Lite -> Full -> Heavy) & Cross-Model Snapshot Audit', async ({
    page,
  }) => {
    test.setTimeout(150_000);
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const liteSnap = await captureAndValidateSnapshot(page);
    expect(liteSnap.model).toBe('lite');

    await switchModelAndWait(page, 'full');
    const fullSnap = await captureAndValidateSnapshot(page);
    expect(fullSnap.model).toBe('full');

    await switchModelAndWait(page, 'heavy', 90_000);
    const heavySnap = await captureAndValidateSnapshot(page);
    expect(heavySnap.model).toBe('heavy');

    const bodyText = await page.locator('body').innerText();
    expect(bodyText).toMatch(/\b(GPU|CPU)\b/);

    tracker.assertNoRuntimeErrors();
  });

  test('T4-05: Presenter Mode (Collapsed Panel + Mirror + Blackout 3D Inspection)', async ({
    page,
  }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const lowerBody = ['left_hip', 'right_hip', 'left_knee', 'right_knee'] as const;
    await setJointSelection(page, lowerBody);
    await setToggleState(page.getByTestId('mirror-toggle'), true);
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

    expect((await getElementFlipState(page.getByTestId('pose-video'))).isMirroredX).toBe(
      true,
    );
    expect((await getElementFlipState(page.getByTestId('pose-canvas'))).isUnflipped).toBe(
      true,
    );

    await page.getByTestId('panel-toggle').click();
    const snapshot = await captureAndValidateSnapshot(page);
    expect(snapshot.mirrored).toBe(true);
    expect(snapshot.activeJointAngles.map((a) => a.joint).sort()).toEqual(
      [...lowerBody].sort(),
    );

    tracker.assertNoRuntimeErrors();
  });

  test('T4-06: Mobile Field Assessment (390x844) Workflow', async ({ page }) => {
    await enableTaskModelCache(page);
    await page.setViewportSize({ width: 390, height: 844 });
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - window.innerWidth,
    );
    expect(overflow).toBeLessThanOrEqual(1);

    await expect(page.getByTestId('hud-fps')).toBeVisible();
    await page.getByTestId('panel-toggle').click();
    await expect(page.getByTestId('panel-toggle')).toHaveAttribute(
      'aria-expanded',
      'false',
    );
    await page.getByTestId('panel-toggle').click();

    await setToggleState(page.getByTestId('smoothing-toggle'), false);
    await page.getByTestId('mode-blackout').click();
    await page.getByTestId('mode-overlay').click();

    const snapshot = await captureAndValidateSnapshot(page);
    expect(snapshot.smoothing.enabled).toBe(false);
    expect(snapshot.landmarks).toHaveLength(33);

    tracker.assertNoRuntimeErrors();
  });

  test('T4-07: Sustained Monitoring & Reload Resilience Session', async ({ page }) => {
    await enableTaskModelCache(page);
    const tracker = await gotoWithErrorTracking(page);
    await waitForAppReady(page);

    await setRangeSliderValue(page.getByTestId('slider-detection-confidence'), '0.55');
    await setRangeSliderValue(page.getByTestId('slider-smoothing-strength'), '0.6');
    await page.waitForTimeout(3_000);
    tracker.assertNoRuntimeErrors();

    await page.reload();
    await waitForAppReady(page);

    const snapshot = await captureAndValidateSnapshot(page);
    expect(snapshot.landmarks).toHaveLength(33);

    tracker.assertNoRuntimeErrors();
  });
});
