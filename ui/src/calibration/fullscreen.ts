/** Best-effort full screen helpers. Browsers may refuse; callers carry on. */

export async function enterFullscreen(): Promise<void> {
  try {
    await document.documentElement.requestFullscreen?.()
  } catch {
    // Full screen is a nicety; calibration works windowed too.
  }
}

export async function exitFullscreen(): Promise<void> {
  if (!document.fullscreenElement) return
  try {
    await document.exitFullscreen()
  } catch {
    // Nothing to do if the browser refuses.
  }
}
