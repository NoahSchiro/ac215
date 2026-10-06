import { describe, expect, it } from 'vitest'
import {
  CALIBRATION_POINTS,
  FRAMES_PER_POINT,
  FRAME_INTERVAL_MS,
} from './points'

describe('CALIBRATION_POINTS', () => {
  it('has exactly nine targets', () => {
    expect(CALIBRATION_POINTS).toHaveLength(9)
  })

  it('keeps every coordinate strictly inside the viewport', () => {
    for (const p of CALIBRATION_POINTS) {
      expect(p.x).toBeGreaterThan(0)
      expect(p.x).toBeLessThan(1)
      expect(p.y).toBeGreaterThan(0)
      expect(p.y).toBeLessThan(1)
    }
  })

  it('has no duplicate targets', () => {
    const keys = new Set(CALIBRATION_POINTS.map((p) => `${p.x},${p.y}`))
    expect(keys.size).toBe(9)
  })

  it('includes the four corners and the center', () => {
    const has = (x: number, y: number) =>
      CALIBRATION_POINTS.some((p) => p.x === x && p.y === y)
    expect(has(0.1, 0.1)).toBe(true)
    expect(has(0.9, 0.1)).toBe(true)
    expect(has(0.1, 0.9)).toBe(true)
    expect(has(0.9, 0.9)).toBe(true)
    expect(has(0.5, 0.5)).toBe(true)
  })

  it('is ordered left to right, then top to bottom', () => {
    expect(CALIBRATION_POINTS[0]).toEqual({ x: 0.1, y: 0.1 })
    expect(CALIBRATION_POINTS[1]).toEqual({ x: 0.5, y: 0.1 })
    expect(CALIBRATION_POINTS[2]).toEqual({ x: 0.9, y: 0.1 })
    expect(CALIBRATION_POINTS[3]).toEqual({ x: 0.1, y: 0.5 })
    expect(CALIBRATION_POINTS[4]).toEqual({ x: 0.5, y: 0.5 })
    expect(CALIBRATION_POINTS[8]).toEqual({ x: 0.9, y: 0.9 })
  })

  it('is symmetric about the center', () => {
    for (const p of CALIBRATION_POINTS) {
      const mirrored = CALIBRATION_POINTS.some(
        (q) => Math.abs(q.x - (1 - p.x)) < 1e-9 && Math.abs(q.y - (1 - p.y)) < 1e-9,
      )
      expect(mirrored).toBe(true)
    }
  })
})

describe('burst settings', () => {
  it('captures at least one frame per target', () => {
    expect(Number.isInteger(FRAMES_PER_POINT)).toBe(true)
    expect(FRAMES_PER_POINT).toBeGreaterThanOrEqual(1)
  })

  it('spaces frames by a positive interval', () => {
    expect(FRAME_INTERVAL_MS).toBeGreaterThan(0)
  })
})
