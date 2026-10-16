import Maze from './Maze'

interface Props {
  onBack: () => void
}

export function MazeScreen({ onBack }: Props) {
  return (
    <main className="maze-screen">
      <Maze />

      <button
        type="button"
        className="back-button"
        onClick={onBack}
      >
        Back to Home
      </button>
    </main>
  )
}
