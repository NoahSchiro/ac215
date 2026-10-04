import { useCallback, useState } from 'react'
import { Calibration } from './calibration/Calibration'
import type { CalibrationResult } from './calibration/points'
import { HomeScreen } from './home/HomeScreen'
import { MazeScreen } from './maze/MazeScreen'

type View = 'home' | 'maze' | 'calibration'

function App() {
  const [view, setView] = useState<View>('home')
  const [calibration, setCalibration] = useState<CalibrationResult | null>(null)

  const goHome = useCallback(() => setView('home'), [])

  // Full screen is entered by the calibration screen itself, after the
  // camera permission prompt has been answered. Safari hides the prompt
  // behind a full-screen page.
  const startCalibration = useCallback(() => setView('calibration'), [])

  const finishCalibration = useCallback((result: CalibrationResult) => {
    setCalibration(result)
    setView('home')
  }, [])

  const cancelCalibration = useCallback(() => {
    setView('home')
  }, [])

  switch (view) {
    case 'maze':
      return <MazeScreen onBack={goHome} />
    case 'calibration':
      return <Calibration onComplete={finishCalibration} onCancel={cancelCalibration} />
    default:
      return (
        <HomeScreen
          calibration={calibration}
          onStartMaze={() => setView('maze')}
          onStartCalibration={startCalibration}
        />
      )
  }
}

export default App
