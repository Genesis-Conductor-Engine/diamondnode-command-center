#!/usr/bin/env node
/**
 * WebGPU page screencast → stdout (MJPEG pipe for ffmpeg image2pipe)
 * Opens http://127.0.0.1:8790/ with WebGPU enabled in Chromium.
 */
import { chromium } from 'playwright';

const URL = process.env.WEBGPU_STREAM_URL || 'http://127.0.0.1:8790/';
const WIDTH = parseInt(process.env.SOTA_STREAM_WIDTH || '1920', 10);
const HEIGHT = parseInt(process.env.SOTA_STREAM_HEIGHT || '1080', 10);
const FPS = parseInt(process.env.SOTA_STREAM_FPS || '30', 10);
const INTERVAL_MS = Math.floor(1000 / FPS);

const args = [
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan',
  '--use-angle=vulkan',
  '--disable-dev-shm-usage',
  '--no-sandbox',
  '--window-size=1920,1080',
];

const browser = await chromium.launch({ headless: true, args });
const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT } });
page.on('pageerror', (e) => process.stderr.write(`pageerror: ${e}\n`));
page.on('console', (m) => { if (m.type() === 'error') process.stderr.write(`console: ${m.text()}\n`); });
await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 45000 });
await page.waitForTimeout(2500);

let running = true;
process.on('SIGINT', () => { running = false; });
process.on('SIGTERM', () => { running = false; });

while (running) {
  const t0 = Date.now();
  try {
    const buf = await page.screenshot({ type: 'jpeg', quality: 88 });
    process.stdout.write(buf);
  } catch (e) {
    process.stderr.write(`capture error: ${e}\n`);
  }
  const elapsed = Date.now() - t0;
  const wait = Math.max(0, INTERVAL_MS - elapsed);
  if (wait) await new Promise((r) => setTimeout(r, wait));
}

await browser.close();