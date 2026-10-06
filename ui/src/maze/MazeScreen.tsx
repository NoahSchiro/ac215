import './maze.css'

interface Props {
  onBack: () => void
}

/**
 * Placeholder for the maze view. Issue #8 replaces the contents of this
 * component; the App switcher and the onBack contract stay the same.
 */
export function MazeScreen({ onBack }: Props) {
  return (
    <main className="maze">
      <h1>Maze</h1>
      <p>The maze is coming soon.</p>
      <button type="button" onClick={onBack}>
        Back to home
      </button>
    </main>
  )
}
