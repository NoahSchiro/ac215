# Web UI for the gaze tracking maze

React + TypeScript + Vite frontend.

Run the following commands from the `ui` directory.

## Install dependencies

```bash
cd ui
npm ci
```

## Run locally

```bash
npm run dev
```

Open the local URL printed in the terminal, usually
http://localhost:5173. Press Ctrl+C to stop the development server.

## Build

```bash
npm run build
```

This runs TypeScript checks and creates the production bundle in `dist/`.

## Test

```bash
npm run test
```

This runs the Vitest test suite once using the jsdom environment.

## Lint

```bash
npm run lint
```

The webcam only works on `localhost` or over HTTPS. Current Chrome and Safari
are the supported browsers.

## Structure

```
src/
  App.tsx             view switcher: home | maze | calibration
  home/               HomeScreen: choose maze or calibration
  maze/               MazeScreen: placeholder, replaced by #8
  calibration/
    points.ts         the 9 targets, sample types, burst constants
    capture.ts        video frame -> JPEG blob, burst capture
    useCamera.ts      owns the webcam MediaStream
    Calibration.tsx   the calibration screen
    fullscreen.ts     best-effort enter/exit full screen
    Crosshair.tsx     target marker
  test-setup.ts       jest-dom matchers + Testing Library cleanup
```

Each screen lives in its own folder so work on different screens does not
conflict. `App.tsx` only switches views and holds the last
`CalibrationResult`.

## Calibration

Clicking **Calibration** on the home screen opens the webcam first, while the
page is still windowed, so the browser's permission prompt stays visible.
(Safari attaches the prompt to the toolbar, which full screen hides.) Once the
camera is ready a **Camera ready** screen appears; clicking **Begin** (or
pressing Space) enters full screen and shows the first crosshair. Targets sit
on a 3x3 grid, inset 10% from each edge, ordered left to right then top to
bottom.

- **Space** captures a burst of 3 frames 100 ms apart, labels each with the
  current target, and advances to the next crosshair.
- **Escape** cancels and returns home.
- After the 9th target the camera is stopped, full screen exits, and the home
  screen shows the frame count with a **Recalibrate** option.
- If the camera is denied or missing, an error screen offers **Retry** and
  **Back**.

Each `CalibrationSample` holds the full webcam frame as a JPEG `Blob`, the
target as normalized `(x, y)` in `[0, 1]` with `(0, 0)` at the top-left, the
target index, the viewport size, and a timestamp. This is the same label space
the gaze model predicts in. Samples stay in browser memory and are never
uploaded; fitting the per-user model is a later issue.

Full screen is best effort: if the browser refuses, calibration runs windowed.

## Testing

Component tests use `@testing-library/react` on jsdom. jsdom has no camera,
canvas, or full-screen API, so tests mock `navigator.mediaDevices`,
`HTMLCanvasElement.prototype.toBlob`, and `document.requestFullscreen`.
Testing Library's automatic cleanup is registered in `src/test-setup.ts`
because Vitest runs without globals.
