import { useEffect, useState } from 'react'
import './App.css'
import Maze from './maze/Maze'

function App() {
  const [theme, setTheme] = useState<'light' | 'dark'>('light')

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  return (
    <main>
      <h1>Gaze Tracking Maze</h1>

      <button
        className="theme-toggle"
        onClick={() =>
          setTheme((current) => (current === 'light' ? 'dark' : 'light'))
        }
      >
        Switch to {theme === 'light' ? 'dark' : 'light'} mode
      </button>

      <Maze />
    </main>
  )
}

export default App