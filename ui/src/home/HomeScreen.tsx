import type { CalibrationResult } from '../calibration/points'
import './home.css'

interface Props {
  calibration: CalibrationResult | null
  onStartMaze: () => void
  onStartCalibration: () => void
}

/** Entry screen: choose between the maze and gaze calibration. */
export function HomeScreen({ calibration, onStartMaze, onStartCalibration }: Props) {
  return (
    <main className="home">
      <h1>Gaze Maze</h1>
      <p className="home-tagline">Steer a ball through a maze using only your eyes.</p>

      <div className="home-actions">
        <button type="button" className="home-card" onClick={onStartMaze}>
          <h2>Maze</h2>
          <p>Play the maze.</p>
        </button>
        <button type="button" className="home-card" onClick={onStartCalibration}>
          <h2>{calibration ? 'Recalibrate' : 'Calibration'}</h2>
          <p>
            {calibration
              ? `${calibration.samples.length} frames captured. Run again to replace them.`
              : 'Teach the model where you are looking. Takes about a minute.'}
          </p>
        </button>
      </div>

      <p className="home-note">
        Calibration uses your webcam. Frames stay in this browser tab and are never uploaded.
      </p>
    </main>
  )
}
