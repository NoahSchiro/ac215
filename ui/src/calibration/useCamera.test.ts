import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useCamera } from './useCamera'

function fakeStream(trackCount = 1) {
  const tracks = Array.from({ length: trackCount }, () => ({ stop: vi.fn() }))
  const stream = { getTracks: () => tracks } as unknown as MediaStream
  return { stream, tracks }
}

function domError(name: string) {
  const e = new Error(name)
  e.name = name
  return e
}

let getUserMedia: ReturnType<typeof vi.fn>

beforeEach(() => {
  getUserMedia = vi.fn()
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia },
  })
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('useCamera', () => {
  it('starts idle with no stream', () => {
    const { result } = renderHook(() => useCamera())
    expect(result.current.status).toBe('idle')
    expect(result.current.stream).toBeNull()
    expect(result.current.error).toBeNull()
  })

  it('goes requesting then ready on success', async () => {
    const { stream } = fakeStream()
    let resolve!: (s: MediaStream) => void
    getUserMedia.mockReturnValue(new Promise<MediaStream>((r) => (resolve = r)))

    const { result } = renderHook(() => useCamera())
    let pending!: Promise<void>
    act(() => {
      pending = result.current.start()
    })
    expect(result.current.status).toBe('requesting')

    await act(async () => {
      resolve(stream)
      await pending
    })
    expect(result.current.status).toBe('ready')
    expect(result.current.stream).toBe(stream)
    expect(getUserMedia).toHaveBeenCalledWith({
      video: { facingMode: 'user' },
      audio: false,
    })
  })

  it('reports denied on NotAllowedError', async () => {
    getUserMedia.mockRejectedValue(domError('NotAllowedError'))
    const { result } = renderHook(() => useCamera())
    await act(() => result.current.start())
    expect(result.current.status).toBe('denied')
    expect(result.current.error).toMatch(/denied/i)
    expect(result.current.stream).toBeNull()
  })

  it('reports unavailable on NotFoundError', async () => {
    getUserMedia.mockRejectedValue(domError('NotFoundError'))
    const { result } = renderHook(() => useCamera())
    await act(() => result.current.start())
    expect(result.current.status).toBe('unavailable')
    expect(result.current.error).toMatch(/no camera/i)
  })

  it('reports unavailable on an unknown error', async () => {
    getUserMedia.mockRejectedValue(new Error('boom'))
    const { result } = renderHook(() => useCamera())
    await act(() => result.current.start())
    expect(result.current.status).toBe('unavailable')
    expect(result.current.error).toBeTruthy()
  })

  it('reports unavailable when the browser lacks mediaDevices', async () => {
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: undefined,
    })
    const { result } = renderHook(() => useCamera())
    await act(() => result.current.start())
    expect(result.current.status).toBe('unavailable')
    expect(result.current.error).toMatch(/support/i)
  })

  it('stop() ends every track and returns to idle', async () => {
    const { stream, tracks } = fakeStream(2)
    getUserMedia.mockResolvedValue(stream)
    const { result } = renderHook(() => useCamera())
    await act(() => result.current.start())
    expect(result.current.status).toBe('ready')

    act(() => result.current.stop())
    tracks.forEach((t) => expect(t.stop).toHaveBeenCalledTimes(1))
    expect(result.current.status).toBe('idle')
    expect(result.current.stream).toBeNull()
  })

  it('does not request again while a stream is active', async () => {
    const { stream } = fakeStream()
    getUserMedia.mockResolvedValue(stream)
    const { result } = renderHook(() => useCamera())
    await act(() => result.current.start())
    await act(() => result.current.start())
    expect(getUserMedia).toHaveBeenCalledTimes(1)
  })

  it('can start again after stop()', async () => {
    const { stream } = fakeStream()
    getUserMedia.mockResolvedValue(stream)
    const { result } = renderHook(() => useCamera())
    await act(() => result.current.start())
    act(() => result.current.stop())
    await act(() => result.current.start())
    expect(getUserMedia).toHaveBeenCalledTimes(2)
    expect(result.current.status).toBe('ready')
  })

  it('makes only one request when start() is called twice while pending', async () => {
    const { stream } = fakeStream()
    let resolve!: (s: MediaStream) => void
    getUserMedia.mockReturnValue(new Promise<MediaStream>((r) => (resolve = r)))

    const { result } = renderHook(() => useCamera())
    let first!: Promise<void>
    act(() => {
      first = result.current.start()
      void result.current.start()
    })
    expect(getUserMedia).toHaveBeenCalledTimes(1)

    await act(async () => {
      resolve(stream)
      await first
    })
    expect(result.current.status).toBe('ready')
  })

  it('stops a stream that resolves after unmount', async () => {
    const { stream, tracks } = fakeStream()
    let resolve!: (s: MediaStream) => void
    getUserMedia.mockReturnValue(new Promise<MediaStream>((r) => (resolve = r)))

    const { result, unmount } = renderHook(() => useCamera())
    let pending!: Promise<void>
    act(() => {
      pending = result.current.start()
    })
    unmount()
    resolve(stream)
    await pending
    expect(tracks[0].stop).toHaveBeenCalledTimes(1)
  })

  it('stops a stream that resolves after stop() was called', async () => {
    const { stream, tracks } = fakeStream()
    let resolve!: (s: MediaStream) => void
    getUserMedia.mockReturnValue(new Promise<MediaStream>((r) => (resolve = r)))

    const { result } = renderHook(() => useCamera())
    let pending!: Promise<void>
    act(() => {
      pending = result.current.start()
    })
    act(() => result.current.stop())
    await act(async () => {
      resolve(stream)
      await pending
    })
    expect(tracks[0].stop).toHaveBeenCalledTimes(1)
    expect(result.current.status).toBe('idle')
    expect(result.current.stream).toBeNull()
  })

  it('survives a Strict Mode style remount with a single stream', async () => {
    const { stream, tracks } = fakeStream()
    let resolve!: (s: MediaStream) => void
    getUserMedia.mockReturnValue(new Promise<MediaStream>((r) => (resolve = r)))

    const { result, rerender } = renderHook(() => useCamera())
    let first!: Promise<void>
    act(() => {
      first = result.current.start()
    })
    // Simulate Strict Mode: effects cleaned up and re-run, start() called again.
    rerender()
    act(() => {
      void result.current.start()
    })
    expect(getUserMedia).toHaveBeenCalledTimes(1)

    await act(async () => {
      resolve(stream)
      await first
    })
    expect(result.current.status).toBe('ready')
    expect(tracks[0].stop).not.toHaveBeenCalled()
  })

  it('stops the stream on unmount', async () => {
    const { stream, tracks } = fakeStream()
    getUserMedia.mockResolvedValue(stream)
    const { result, unmount } = renderHook(() => useCamera())
    await act(() => result.current.start())
    unmount()
    expect(tracks[0].stop).toHaveBeenCalledTimes(1)
  })
})
