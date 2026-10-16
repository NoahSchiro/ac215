import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { enterFullscreen, exitFullscreen } from './fullscreen'

let fullscreenElement: Element | null = null
const request = vi.fn()
const exit = vi.fn()

beforeEach(() => {
  fullscreenElement = null
  request.mockReset().mockResolvedValue(undefined)
  exit.mockReset().mockResolvedValue(undefined)
  document.documentElement.requestFullscreen = request
  document.exitFullscreen = exit
  Object.defineProperty(document, 'fullscreenElement', {
    configurable: true,
    get: () => fullscreenElement,
  })
})

afterEach(() => vi.restoreAllMocks())

describe('enterFullscreen', () => {
  it('requests full screen on the document element', async () => {
    await enterFullscreen()
    expect(request).toHaveBeenCalledTimes(1)
  })

  it('swallows a refusal', async () => {
    request.mockRejectedValue(new Error('denied'))
    await expect(enterFullscreen()).resolves.toBeUndefined()
  })

  it('does nothing when the API is missing', async () => {
    // @ts-expect-error simulating an older browser
    document.documentElement.requestFullscreen = undefined
    await expect(enterFullscreen()).resolves.toBeUndefined()
  })
})

describe('exitFullscreen', () => {
  it('exits when in full screen', async () => {
    fullscreenElement = document.documentElement
    await exitFullscreen()
    expect(exit).toHaveBeenCalledTimes(1)
  })

  it('does not call exit when not in full screen', async () => {
    await exitFullscreen()
    expect(exit).not.toHaveBeenCalled()
  })

  it('swallows a refusal', async () => {
    fullscreenElement = document.documentElement
    exit.mockRejectedValue(new Error('nope'))
    await expect(exitFullscreen()).resolves.toBeUndefined()
  })
})
