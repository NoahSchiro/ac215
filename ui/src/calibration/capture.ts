/**
 * Still-frame capture from a live <video> element.
 *
 * Kept free of React so it can be unit tested with a fake canvas.
 */

const JPEG_QUALITY = 0.92

/**
 * Draw the current video frame to an offscreen canvas and encode it as JPEG.
 * Rejects if the video has no dimensions yet (stream not ready).
 */
export function captureFrame(video: HTMLVideoElement): Promise<Blob> {
  const width = video.videoWidth
  const height = video.videoHeight
  if (width === 0 || height === 0) {
    return Promise.reject(new Error('Video has no frame to capture yet'))
  }

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height

  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return Promise.reject(new Error('Could not get 2D canvas context'))
  }
  ctx.drawImage(video, 0, 0, width, height)

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) resolve(blob)
        else reject(new Error('Canvas produced no image'))
      },
      'image/jpeg',
      JPEG_QUALITY,
    )
  })
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

/**
 * Capture `count` frames spaced `intervalMs` apart.
 * Frames are captured sequentially; a failure in any frame rejects the burst.
 */
export async function captureBurst(
  video: HTMLVideoElement,
  count: number,
  intervalMs: number,
): Promise<Blob[]> {
  const frames: Blob[] = []
  for (let i = 0; i < count; i++) {
    if (i > 0) await sleep(intervalMs)
    frames.push(await captureFrame(video))
  }
  return frames
}
