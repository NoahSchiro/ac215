import { useMemo } from 'react'
import { generateMaze } from './mazeGenerator'
import './Maze.css'

// Number of rows and columns in the maze.
const ROWS = 15
const COLS = 15

export default function Maze() {
  // Generate the maze once when the component is first created.
  // useMemo prevents a new random maze from being generated
  // every time React re-renders the component.
  const maze = useMemo(() => generateMaze(ROWS, COLS), [])

  return (
    <div 
    className="maze" 
    role="group"
    aria-label="Maze"
    style={{gridTemplateColumns: `repeat(${COLS}, 1fr)`,
    gridTemplateRows: `repeat(${ROWS}, 1fr)`,
    }}>
      {maze.map((row) =>
        row.map((cell) => (
          <div
            key={`${cell.row}-${cell.col}`}
            className="maze-cell"
            style={{
              gridRow: cell.row + 1,
              gridColumn: cell.col + 1,

              // Draw only the walls that still exist.
              borderTop: cell.walls.top
                ? '2px solid var(--maze-wall)'
                : 'none',

              borderRight: cell.walls.right
                ? '2px solid var(--maze-wall)'
                : 'none',

              borderBottom: cell.walls.bottom
                ? '2px solid var(--maze-wall)'
                : 'none',

              borderLeft: cell.walls.left
                ? '2px solid var(--maze-wall)'
                : 'none',
            }}
          />
        )),
      )}
    </div>
  )
}