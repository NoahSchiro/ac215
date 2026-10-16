import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { captureBurst, captureFrame } from './capture'

type ToBlobCb = (blob: Blob | null) => void

function fakeVideo(width: number, height: number): HTMLVideoElement {
  const video = document.createElement('video')
  Object.defineProperty(video, 'videoWidth', { value: width })
  Object.defineProperty(video, 'videoHeight', { value: height })
  return video
}

describe('captureFrame', () => {
  const drawImage = vi.fn()
  let toBlobImpl: (cb: ToBlobCb) => void
  let created: HTMLCanvasElement[]

  beforeEach(() => {
    created = []
    toBlobImpl = (cb) => cb(new Blob(['x'], { type: 'image/jpeg' }))

    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(
      () => ({ drawImage }) as unknown as CanvasRenderingContext2D,
    )
    vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation(
      function (this: HTMLCanvasElement, cb: ToBlobCb) {
        created.push(this)
        toBlobImpl(cb)
      },
    )
  })

  afterEach(() => {
    vi.restoreAllMocks()
    drawImage.mockReset()
  })

  it('sizes the canvas to the video and draws the frame', async () => {
    const video = fakeVideo(640, 480)
    const blob = await captureFrame(video)

    expect(blob.type).toBe('image/jpeg')
    expect(created).toHaveLength(1)
    expect(created[0].width).toBe(640)
    expect(created[0].height).toBe(480)
    expect(drawImage).toHaveBeenCalledTimes(1)
    expect(drawImage).toHaveBeenCalledWith(video, 0, 0, 640, 480)
  })

  it('requests JPEG encoding', async () => {
    const spy = vi.mocked(HTMLCanvasElement.prototype.toBlob)
    await captureFrame(fakeVideo(10, 10))
    expect(spy.mock.calls[0][1]).toBe('image/jpeg')
    expect(spy.mock.calls[0][2]).toBeGreaterThan(0)
  })

  it('rejects when the video has no dimensions', async () => {
    await expect(captureFrame(fakeVideo(0, 0))).rejects.toThrow(/no frame/)
    expect(drawImage).not.toHaveBeenCalled()
  })

  it('rejects when the canvas yields no blob', async () => {
    toBlobImpl = (cb) => cb(null)
    await expect(captureFrame(fakeVideo(10, 10))).rejects.toThrow(/no image/)
  })

  it('rejects when no 2D context is available', async () => {
    vi.mocked(HTMLCanvasElement.prototype.getContext).mockReturnValue(null)
    await expect(captureFrame(fakeVideo(10, 10))).rejects.toThrow(/context/)
  })
})

describe('captureBurst', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(
      () => ({ drawImage: vi.fn() }) as unknown as CanvasRenderingContext2D,
    )
    vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation(
      (cb: ToBlobCb) => cb(new Blob(['x'], { type: 'image/jpeg' })),
    )
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('returns the requested number of frames', async () => {
    const promise = captureBurst(fakeVideo(10, 10), 3, 100)
    await vi.advanceTimersByTimeAsync(200)
    const frames = await promise
    expect(frames).toHaveLength(3)
    frames.forEach((f) => expect(f).toBeInstanceOf(Blob))
  })

  it('waits the interval between frames', async () => {
    const toBlob = vi.mocked(HTMLCanvasElement.prototype.toBlob)
    const promise = captureBurst(fakeVideo(10, 10), 3, 100)

    await vi.advanceTimersByTimeAsync(0)
    expect(toBlob).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(99)
    expect(toBlob).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(1)
    expect(toBlob).toHaveBeenCalledTimes(2)
    await vi.advanceTimersByTimeAsync(100)
    expect(toBlob).toHaveBeenCalledTimes(3)

    await expect(promise).resolves.toHaveLength(3)
  })

  it('captures a single frame with no delay when count is 1', async () => {
    const promise = captureBurst(fakeVideo(10, 10), 1, 500)
    await vi.advanceTimersByTimeAsync(0)
    await expect(promise).resolves.toHaveLength(1)
  })

  it('rejects the whole burst if a frame fails', async () => {
    const promise = captureBurst(fakeVideo(0, 0), 3, 100)
    await expect(promise).rejects.toThrow(/no frame/)
  })
})
