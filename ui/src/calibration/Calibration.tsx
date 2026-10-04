import type { CalibrationResult } from './points'

interface Props {
  onComplete: (result: CalibrationResult) => void
  onCancel: () => void
}

/** Placeholder until the camera and calibration flow are added. */
export function Calibration({ onCancel }: Props) {
  return (
    <main className="home">
      <h1>Calibration</h1>
      <p>Calibration is coming soon.</p>
      <button type="button" onClick={onCancel}>
        Back to home
      </button>
    </main>
  )
}
