import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import type { CalibrationResult } from './calibration/points'

type CalibrationProps = {
  onComplete: (r: CalibrationResult) => void
  onCancel: () => void
}
let calibrationProps: CalibrationProps | null = null

vi.mock('./calibration/Calibration', () => ({
  Calibration: (props: CalibrationProps) => {
    calibrationProps = props
    return <div data-testid="calibration-screen" />
  },
}))

beforeEach(() => {
  calibrationProps = null
})

afterEach(() => {
  vi.restoreAllMocks()
})

function fakeResult(count: number): CalibrationResult {
  return {
    samples: Array.from({ length: count }, (_, i) => ({
      image: new Blob(['x']),
      target: { x: 0.5, y: 0.5 },
      pointIndex: i,
      viewportWidth: 800,
      viewportHeight: 600,
      timestamp: 0,
    })),
    completedAt: 0,
  }
}

describe('App', () => {
  it('shows the home screen with Maze and Calibration options', () => {
    render(<App />)
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Gaze Maze')
    expect(screen.getByRole('button', { name: /^Maze/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^Calibration/ })).toBeInTheDocument()
  })

  it('opens the maze and returns home', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Maze/ }))
    expect(screen.getByText(/coming soon/)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /Back to home/ }))
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Gaze Maze')
  })

  it('opens the calibration screen', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    expect(screen.getByTestId('calibration-screen')).toBeInTheDocument()
  })

  it('returns home with the result after calibration completes', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))

    await act(async () => calibrationProps!.onComplete(fakeResult(27)))

    expect(screen.getByText(/27 frames captured/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^Recalibrate/ })).toBeInTheDocument()
  })

  it('returns home without a result when calibration is cancelled', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))

    await act(async () => calibrationProps!.onCancel())

    expect(screen.queryByText(/frames captured/)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^Calibration/ })).toBeInTheDocument()
  })

  it('replaces the previous result on recalibration', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /^Calibration/ }))
    await act(async () => calibrationProps!.onComplete(fakeResult(27)))

    fireEvent.click(screen.getByRole('button', { name: /^Recalibrate/ }))
    await act(async () => calibrationProps!.onComplete(fakeResult(9)))
    expect(screen.getByText(/9 frames captured/)).toBeInTheDocument()
  })
})
