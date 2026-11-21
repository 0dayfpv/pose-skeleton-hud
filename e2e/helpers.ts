import fs from 'node:fs/promises';
import {
  chromium,
  expect,
  type Browser,
  type BrowserContext,
  type ConsoleMessage,
  type Download,
  type Locator,
  type Page,
  type Route,
} from '@playwright/test';

export const DEFAULT_BASE_URL = 'http://localhost:4173';

export const WEBGL_SWIFTSHADER_ARGS = [
  '--enable-webgl',
  '--use-gl=angle',
  '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader',
] as const;

export const FAKE_WEBCAM_ARGS = [
  '--use-fake-device-for-media-stream',
  '--use-fake-ui-for-media-stream',
  ...WEBGL_SWIFTSHADER_ARGS,
] as const;

export const ALL_JOINT_IDS = [
  'left_elbow',
  'right_elbow',
  'left_shoulder',
  'right_shoulder',
  'left_hip',
  'right_hip',
  'left_knee',
  'right_knee',
] as const;

export type JointId = (typeof ALL_JOINT_IDS)[number];
export type ModelVariant = 'lite' | 'full' | 'heavy';

export type MediaMockErrorName =
  | 'NotAllowedError'
  | 'NotFoundError'
  | 'NotSupportedError'
  | 'NotReadableError'
  | 'OverconstrainedError';

export interface MediaMockErrorSpec {
  name: MediaMockErrorName | string;
  message: string;
  emptyDevices?: boolean;
}

interface E2EMediaWindow extends Window {
  __e2eMediaMockError?: MediaMockErrorSpec | null;
}

export interface RuntimeErrorTracker {
  readonly pageErrors: readonly string[];
  readonly consoleErrors: readonly string[];
  assertNoPageErrors: () => void;
  assertNoConsoleErrors: () => void;
  assertNoRuntimeErrors: () => void;
  clear: () => void;
}

export interface SnapshotLandmarkPayload {
  index: number;
  name: string;
  x: number;
  y: number;
  z: number;
  visibility: number;
}

export interface SnapshotJointAnglePayload {
  joint: string;
  label: string;
  degrees: number | null;
}

export interface SnapshotJsonPayload {
  schemaVersion: string;
  timestamp: string;
  timestampMs: number;
  poseDetected: boolean;
  model: ModelVariant;
  mirrored: boolean;
  smoothing: {
    enabled: boolean;
    strength: number;
  };
  landmarks: SnapshotLandmarkPayload[];
  activeJointAngles: SnapshotJointAnglePayload[];
}

export interface ElementFlipState {
  transform: string;
  scale: string;
  transformM11: number;
  isMirroredX: boolean;
  isUnflipped: boolean;
}

export interface CanvasBackdropSample {
  width: number;
  height: number;
  clientWidth: number;
  clientHeight: number;
  sampledPoints: number;
  opaqueDarkPixels: number;
  nonTransparentPixels: number;
}

export interface TaskModelCacheOptions {
  delayByModel?: Partial<Record<ModelVariant, number>>;
  failModel?: ModelVariant | null;
}

export interface TaskModelCacheController {
  setFailModel: (model: ModelVariant | null) => void;
  getFailModel: () => ModelVariant | null;
}

interface CachedTaskAsset {
  status: number;
  contentType: string;
  body: Buffer;
}

const taskModelBufferCache = new Map<string, CachedTaskAsset>();

/**
 * Attach listeners for uncaught page errors and console.error messages.
 * Must be called BEFORE `page.goto(...)` so startup/script-eval errors are captured.
 */
export function attachRuntimeErrorTracker(page: Page): RuntimeErrorTracker {
  const pageErrors: string[] = [];
  const consoleErrors: string[] = [];

  page.on('pageerror', (error: Error) => {
    pageErrors.push(error.stack ?? error.message);
  });

  page.on('console', (msg: ConsoleMessage) => {
    if (msg.type() === 'error') {
      const location = msg.location();
      const locSuffix = location?.url ? ` (${location.url}:${location.lineNumber})` : '';
      consoleErrors.push(`${msg.text()}${locSuffix}`);
    }
  });

  return {
    get pageErrors() {
      return pageErrors;
    },
    get consoleErrors() {
      return consoleErrors;
    },
    assertNoPageErrors() {
      expect(
        pageErrors,
        `Expected 0 uncaught pageerror events, saw:\n${pageErrors.join('\n')}`,
      ).toEqual([]);
    },
    assertNoConsoleErrors() {
      expect(
        consoleErrors,
        `Expected 0 console.error messages, saw:\n${consoleErrors.join('\n')}`,
      ).toEqual([]);
    },
    assertNoRuntimeErrors() {
      this.assertNoPageErrors();
      this.assertNoConsoleErrors();
    },
    clear() {
      pageErrors.length = 0;
      consoleErrors.length = 0;
    },
  };
}

/**
 * Attaches runtime error tracking before navigating to `path` (default `/`).
 */
export async function gotoWithErrorTracking(
  page: Page,
  path = '/',
): Promise<RuntimeErrorTracker> {
  const tracker = attachRuntimeErrorTracker(page);
  await page.goto(path);
  return tracker;
}

/**
 * Installs a deterministic, React-StrictMode-safe `getUserMedia` (and optional `enumerateDevices`)
 * error mock via `page.addInitScript`. Call BEFORE `page.goto(...)`.
 */
export async function installMediaErrorMock(
  page: Page,
  initialError: MediaMockErrorSpec | null,
): Promise<void> {
  await page.addInitScript((spec: MediaMockErrorSpec | null) => {
    const win = window as unknown as E2EMediaWindow;
    win.__e2eMediaMockError = spec;

    const mediaDevices = navigator.mediaDevices;
    if (!mediaDevices) {
      return;
    }

    const originalGetUserMedia = mediaDevices.getUserMedia.bind(mediaDevices);
    Object.defineProperty(mediaDevices, 'getUserMedia', {
      configurable: true,
      writable: true,
      value: async (constraints?: MediaStreamConstraints): Promise<MediaStream> => {
        const activeError = win.__e2eMediaMockError;
        if (activeError) {
          throw new DOMException(activeError.message, activeError.name);
        }
        return originalGetUserMedia(constraints);
      },
    });

    if (typeof mediaDevices.enumerateDevices === 'function') {
      const originalEnumerateDevices = mediaDevices.enumerateDevices.bind(mediaDevices);
      Object.defineProperty(mediaDevices, 'enumerateDevices', {
        configurable: true,
        writable: true,
        value: async (): Promise<MediaDeviceInfo[]> => {
          const activeError = win.__e2eMediaMockError;
          if (activeError?.emptyDevices) {
            return [];
          }
          return originalEnumerateDevices();
        },
      });
    }
  }, initialError);
}

/**
 * Simulates a browser/context without `navigator.mediaDevices` (e.g. insecure or unsupported context).
 */
export async function installMissingMediaDevicesMock(page: Page): Promise<void> {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      get: () => undefined,
    });
  });
}

/**
 * Delays `getUserMedia` resolution by `delayMs` so loading states can be observed reliably.
 */
export async function installDelayedMediaMock(page: Page, delayMs: number): Promise<void> {
  await page.addInitScript((waitMs: number) => {
    const mediaDevices = navigator.mediaDevices;
    if (!mediaDevices) {
      return;
    }
    const originalGetUserMedia = mediaDevices.getUserMedia.bind(mediaDevices);
    Object.defineProperty(mediaDevices, 'getUserMedia', {
      configurable: true,
      writable: true,
      value: async (constraints?: MediaStreamConstraints): Promise<MediaStream> => {
        await new Promise((resolve) => setTimeout(resolve, waitMs));
        return originalGetUserMedia(constraints);
      },
    });
  }, delayMs);
}

export async function setMediaErrorMock(
  page: Page,
  nextError: MediaMockErrorSpec | null,
): Promise<void> {
  await page.evaluate((spec: MediaMockErrorSpec | null) => {
    const win = window as unknown as E2EMediaWindow;
    win.__e2eMediaMockError = spec;
  }, nextError);
}

export async function clearMediaErrorMock(page: Page): Promise<void> {
  await setMediaErrorMock(page, null);
}

export interface NoFakeDeviceSession {
  browser: Browser;
  context: BrowserContext;
  page: Page;
  tracker: RuntimeErrorTracker;
  close: () => Promise<void>;
}

/**
 * Launches an isolated headless Chromium instance WITHOUT `--use-fake-device-for-media-stream`,
 * WITHOUT `--use-fake-ui-for-media-stream`, and WITHOUT camera permissions, so tests can verify
 * real headless Chromium no-device behavior (`NotSupportedError` status screen).
 */
export async function launchNoFakeDeviceSession(
  baseURL = DEFAULT_BASE_URL,
): Promise<NoFakeDeviceSession> {
  const browser = await chromium.launch({
    headless: true,
    args: [...WEBGL_SWIFTSHADER_ARGS],
  });
  try {
    const context = await browser.newContext({
      viewport: { width: 1280, height: 720 },
    });
    const page = await context.newPage();
    const tracker = attachRuntimeErrorTracker(page);
    await page.goto(baseURL);
    return {
      browser,
      context,
      page,
      tracker,
      async close() {
        await context.close().catch(() => undefined);
        await browser.close().catch(() => undefined);
      },
    };
  } catch (error) {
    await browser.close().catch(() => undefined);
    throw error;
  }
}

/**
 * Opaque-box worker-level HTTP cache for MediaPipe `.task` model files.
 * Caches each `.task` buffer once per worker while preserving CORS `*` headers
 * and supporting optional controlled delays or HTTP 500 error injection.
 */
export async function enableTaskModelCache(
  page: Page,
  options: TaskModelCacheOptions = {},
): Promise<TaskModelCacheController> {
  let activeFailModel: ModelVariant | null = options.failModel ?? null;

  await page.route(
    (url) => url.pathname.endsWith('.task'),
    async (route: Route) => {
      const requestUrl = route.request().url();
      const matchedModel: ModelVariant | null = requestUrl.includes('heavy')
        ? 'heavy'
        : requestUrl.includes('full')
          ? 'full'
          : requestUrl.includes('lite')
            ? 'lite'
            : null;

      const delayMs = matchedModel ? (options.delayByModel?.[matchedModel] ?? 0) : 0;
      if (delayMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }

      if (matchedModel && activeFailModel === matchedModel) {
        await route
          .fulfill({
            status: 500,
            contentType: 'text/plain',
            headers: {
              'access-control-allow-origin': '*',
              'cache-control': 'no-store',
            },
            body: 'Simulated .task download error',
          })
          .catch(() => undefined);
        return;
      }

      const cached = taskModelBufferCache.get(requestUrl);
      if (cached) {
        await route
          .fulfill({
            status: cached.status,
            contentType: cached.contentType,
            headers: { 'access-control-allow-origin': '*' },
            body: cached.body,
          })
          .catch(() => undefined);
        return;
      }

      try {
        const response = await route.fetch({ maxRedirects: 10 });
        const body = await response.body();
        const contentType =
          response.headers()['content-type'] ?? 'application/octet-stream';
        if (response.ok()) {
          taskModelBufferCache.set(requestUrl, {
            status: response.status(),
            contentType,
            body,
          });
        }
        await route
          .fulfill({
            status: response.status(),
            contentType,
            headers: { 'access-control-allow-origin': '*' },
            body,
          })
          .catch(() => undefined);
      } catch {
        await route.abort().catch(() => undefined);
      }
    },
  );

  return {
    setFailModel(model: ModelVariant | null) {
      activeFailModel = model;
    },
    getFailModel() {
      return activeFailModel;
    },
  };
}

/**
 * Waits until the camera stream is active, the canvas is visible, and the status overlay
 * has cleared (indicating both camera capture and PoseLandmarker initialization succeeded).
 */
export async function waitForAppReady(
  page: Page,
  timeoutMs = 45_000,
): Promise<void> {
  await expect(page.getByTestId('pose-canvas')).toBeVisible({ timeout: timeoutMs });
  await expect(page.getByTestId('status-overlay')).toBeHidden({ timeout: timeoutMs });
  await page.waitForFunction(
    () => {
      const video = document.querySelector<HTMLVideoElement>('[data-testid="pose-video"]');
      return Boolean(video && video.readyState >= 2 && video.videoWidth > 0);
    },
    undefined,
    { timeout: timeoutMs },
  );
}

export async function waitForStatusOverlay(
  page: Page,
  timeoutMs = 20_000,
): Promise<Locator> {
  const overlay = page.getByTestId('status-overlay');
  await expect(overlay).toBeVisible({ timeout: timeoutMs });
  return overlay;
}

/**
 * Reads active/pressed/checked state across either `<button aria-pressed>` or `<input type="checkbox">`.
 */
export async function isControlActive(locator: Locator): Promise<boolean> {
  return locator.evaluate((el) => {
    if (el instanceof HTMLInputElement && (el.type === 'checkbox' || el.type === 'radio')) {
      return el.checked;
    }
    const pressed = el.getAttribute('aria-pressed');
    if (pressed !== null) {
      return pressed === 'true';
    }
    const checked = el.getAttribute('aria-checked');
    if (checked !== null) {
      return checked === 'true';
    }
    const selected = el.getAttribute('aria-selected');
    if (selected !== null) {
      return selected === 'true';
    }
    return el.getAttribute('data-active') === 'true';
  });
}

export async function setToggleState(locator: Locator, desired: boolean): Promise<void> {
  const current = await isControlActive(locator);
  if (current !== desired) {
    await locator.click();
    await expect.poll(() => isControlActive(locator)).toBe(desired);
  }
}

export async function setJointSelection(
  page: Page,
  selectedJoints: readonly JointId[],
): Promise<void> {
  const wanted = new Set<JointId>(selectedJoints);
  for (const jointId of ALL_JOINT_IDS) {
    await setToggleState(page.getByTestId(`joint-toggle-${jointId}`), wanted.has(jointId));
  }
}

export async function switchModelAndWait(
  page: Page,
  targetModel: ModelVariant,
  timeoutMs = 60_000,
): Promise<void> {
  const btn = page.getByTestId(`model-${targetModel}`);
  await btn.click();
  await expect.poll(() => isControlActive(btn), { timeout: timeoutMs }).toBe(true);
  await expect(page.getByTestId('status-overlay')).toBeHidden({ timeout: timeoutMs });
  await expect(page.getByTestId('hud-inference-ms')).toHaveText(/\d+(\.\d+)?\s*ms/i, {
    timeout: timeoutMs,
  });
}

export async function setRangeSliderValue(locator: Locator, value: string): Promise<void> {
  await locator.fill(value);
}

/**
 * Inspects both CSS `transform` (`m11`) and standalone CSS `scale` (`-1 ...`) so Mirror
 * assertions work regardless of whether `scale-x-[-1]` compiles to `transform` or `scale`.
 */
export async function getElementFlipState(locator: Locator): Promise<ElementFlipState> {
  return locator.evaluate((el) => {
    const styles = window.getComputedStyle(el);
    const transform = styles.transform || 'none';
    const scale = styles.scale || 'none';

    let transformM11 = 1;
    if (transform !== 'none' && transform !== '') {
      transformM11 = new DOMMatrix(transform).m11;
    }

    const scaleTrimmed = scale.trim();
    const scaleStartsNegativeOne = scaleTrimmed.startsWith('-1');
    const scaleStartsNegative = scaleTrimmed.startsWith('-');
    const isMirroredX =
      Math.abs(transformM11 + 1) < 1e-3 || scaleStartsNegativeOne;
    const isUnflipped =
      (transform === 'none' || Math.abs(transformM11 - 1) < 1e-3) &&
      (scaleTrimmed === 'none' || scaleTrimmed === '' || !scaleStartsNegative);

    return {
      transform,
      scale,
      transformM11,
      isMirroredX,
      isUnflipped,
    };
  });
}

export async function isVideoVisuallyHidden(page: Page): Promise<boolean> {
  return page.getByTestId('pose-video').evaluate((el) => {
    const styles = window.getComputedStyle(el);
    return (
      styles.opacity === '0' ||
      styles.visibility === 'hidden' ||
      styles.display === 'none'
    );
  });
}

export async function sampleCanvasBackdrop(page: Page): Promise<CanvasBackdropSample> {
  return page.getByTestId('pose-canvas').evaluate((el) => {
    const canvas = el as HTMLCanvasElement;
    const ctx = canvas.getContext('2d');
    if (!ctx || canvas.width === 0 || canvas.height === 0) {
      return {
        width: canvas.width,
        height: canvas.height,
        clientWidth: canvas.clientWidth,
        clientHeight: canvas.clientHeight,
        sampledPoints: 0,
        opaqueDarkPixels: 0,
        nonTransparentPixels: 0,
      };
    }

    let sampledPoints = 0;
    let opaqueDarkPixels = 0;
    let nonTransparentPixels = 0;

    for (let gx = 1; gx <= 5; gx += 1) {
      for (let gy = 1; gy <= 5; gy += 1) {
        const px = Math.min(canvas.width - 1, Math.floor((canvas.width * gx) / 6));
        const py = Math.min(canvas.height - 1, Math.floor((canvas.height * gy) / 6));
        const data = ctx.getImageData(px, py, 1, 1).data;
        const r = data[0] ?? 0;
        const g = data[1] ?? 0;
        const b = data[2] ?? 0;
        const a = data[3] ?? 0;
        sampledPoints += 1;
        if (a > 0) {
          nonTransparentPixels += 1;
        }
        if (a >= 200 && r < 50 && g < 70 && b < 90) {
          opaqueDarkPixels += 1;
        }
      }
    }

    return {
      width: canvas.width,
      height: canvas.height,
      clientWidth: canvas.clientWidth,
      clientHeight: canvas.clientHeight,
      sampledPoints,
      opaqueDarkPixels,
      nonTransparentPixels,
    };
  });
}

export async function getBoundingBoxDelta(page: Page): Promise<{
  widthDelta: number;
  heightDelta: number;
  videoWidth: number;
  videoHeight: number;
  canvasWidth: number;
  canvasHeight: number;
}> {
  const videoBox = await page.getByTestId('pose-video').boundingBox();
  const canvasBox = await page.getByTestId('pose-canvas').boundingBox();
  if (!videoBox || !canvasBox) {
    throw new Error('Expected both pose-video and pose-canvas to have bounding boxes');
  }
  return {
    widthDelta: Math.abs(videoBox.width - canvasBox.width),
    heightDelta: Math.abs(videoBox.height - canvasBox.height),
    videoWidth: videoBox.width,
    videoHeight: videoBox.height,
    canvasWidth: canvasBox.width,
    canvasHeight: canvasBox.height,
  };
}

async function readDownloadText(download: Download): Promise<string> {
  const failure = await download.failure();
  expect(failure).toBeNull();

  const filePath = await download.path().catch(() => null);
  if (filePath) {
    return fs.readFile(filePath, 'utf-8');
  }

  const stream = await download.createReadStream();
  if (!stream) {
    throw new Error(`Failed to open download read stream for "${download.suggestedFilename()}"`);
  }
  const chunks: Buffer[] = [];
  for await (const chunk of stream) {
    chunks.push(
      Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as Uint8Array | string),
    );
  }
  return Buffer.concat(chunks).toString('utf-8');
}

export interface DownloadedJsonResult<T = unknown> {
  filename: string;
  rawText: string;
  data: T;
}

/**
 * Triggers a browser download action, reads the downloaded file/stream cleanly,
 * and parses its JSON payload.
 */
export async function readDownloadedJson<T = unknown>(
  page: Page,
  triggerAction: () => Promise<void>,
): Promise<DownloadedJsonResult<T>> {
  const downloadPromise = page.waitForEvent('download');
  await triggerAction();
  const download = await downloadPromise;
  const filename = download.suggestedFilename();
  const rawText = await readDownloadText(download);
  const data = JSON.parse(rawText) as T;
  return { filename, rawText, data };
}

/**
 * Clicks `snapshot-json-button`, verifies filename & full PoseSnapshot JSON schema,
 * and returns the parsed payload for additional caller assertions.
 */
export async function captureAndValidateSnapshot(
  page: Page,
): Promise<SnapshotJsonPayload> {
  const { filename, data } = await readDownloadedJson<SnapshotJsonPayload>(page, async () => {
    await page.getByTestId('snapshot-json-button').click();
  });

  expect(filename).toMatch(/^pose-snapshot-[^\s/\\:]+\.json$/);
  expect(data.schemaVersion).toBe('1.0.0');
  expect(typeof data.timestamp).toBe('string');
  const parsedDateMs = Date.parse(data.timestamp);
  expect(Number.isNaN(parsedDateMs)).toBe(false);
  expect(new Date(data.timestamp).toISOString()).toBe(data.timestamp);

  expect(typeof data.timestampMs).toBe('number');
  expect(Number.isFinite(data.timestampMs)).toBe(true);
  expect(typeof data.poseDetected).toBe('boolean');
  expect(['lite', 'full', 'heavy']).toContain(data.model);
  expect(typeof data.mirrored).toBe('boolean');

  expect(typeof data.smoothing).toBe('object');
  expect(data.smoothing).not.toBeNull();
  expect(typeof data.smoothing.enabled).toBe('boolean');
  expect(typeof data.smoothing.strength).toBe('number');
  expect(Number.isFinite(data.smoothing.strength)).toBe(true);
  expect(data.smoothing.strength).toBeGreaterThanOrEqual(0);
  expect(data.smoothing.strength).toBeLessThanOrEqual(1);

  expect(Array.isArray(data.landmarks)).toBe(true);
  expect(data.landmarks).toHaveLength(33);
  data.landmarks.forEach((lm, idx) => {
    expect(lm.index).toBe(idx);
    expect(typeof lm.name).toBe('string');
    expect(lm.name.length).toBeGreaterThan(0);
    for (const key of ['x', 'y', 'z', 'visibility'] as const) {
      expect(typeof lm[key]).toBe('number');
      expect(Number.isFinite(lm[key])).toBe(true);
    }
    expect(lm.visibility).toBeGreaterThanOrEqual(0);
    expect(lm.visibility).toBeLessThanOrEqual(1);
  });

  expect(Array.isArray(data.activeJointAngles)).toBe(true);
  for (const entry of data.activeJointAngles) {
    expect(typeof entry.joint).toBe('string');
    expect(typeof entry.label).toBe('string');
    if (entry.degrees !== null) {
      expect(typeof entry.degrees).toBe('number');
      expect(Number.isFinite(entry.degrees)).toBe(true);
      expect(entry.degrees).toBeGreaterThanOrEqual(0);
      expect(entry.degrees).toBeLessThanOrEqual(180);
    }
  }

  return data;
}
