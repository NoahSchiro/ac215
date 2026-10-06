import Maze from './Maze'

interface Props {
  onBack: () => void
}

export function MazeScreen({ onBack }: Props) {
  return (
    <main className="maze-screen">

      <Maze />

      <button type="button" onClick={onBack}>
        Back to home
      </button>
    </main>
  )
}