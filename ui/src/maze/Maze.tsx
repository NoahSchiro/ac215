import { useCallback, useEffect, useMemo, useState } from 'react'
import { generateMaze } from './mazeGenerator'
import './Maze.css'

const ROWS = 15
const COLS = 15
const MOVE_INTERVAL = 160

type Direction = 'up' | 'down' | 'left' | 'right'

const KEY_DIRECTIONS: Record<string, Direction> = {
  ArrowUp: 'up',
  ArrowDown: 'down',
  ArrowLeft: 'left',
  ArrowRight: 'right',
}

export default function Maze() {
  const maze = useMemo(() => generateMaze(ROWS, COLS), [])

  const [player, setPlayer] = useState({
    row: 0,
    col: 0,
    rotation: 0,
    hasMoved: false,
  })

  const movePlayer = useCallback(
    (direction: Direction) => {
      setPlayer((current) => {
        const cell = maze[current.row][current.col]
        let nextRow = current.row
        let nextCol = current.col

        // Check for a wall before moving.
        switch (direction) {
          case 'up':
            if (cell.walls.top) return current
            nextRow -= 1
            break
          case 'down':
            if (cell.walls.bottom) return current
            nextRow += 1
            break
          case 'left':
            if (cell.walls.left) return current
            nextCol -= 1
            break
          case 'right':
            if (cell.walls.right) return current
            nextCol += 1
            break
        }

        // Keep the player inside the maze.
        if (
          nextRow < 0 ||
          nextRow >= ROWS ||
          nextCol < 0 ||
          nextCol >= COLS
        ) {
          return current
        }

        return {
          row: nextRow,
          col: nextCol,
          rotation:
            current.rotation +
            (direction === 'right' || direction === 'down' ? 90 : -90),
          hasMoved: true,
        }
      })
    },
    [maze],
  )

  // Move repeatedly while an arrow key is held.
  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | undefined
    let activeKey: string | null = null

    function stopMoving() {
      clearInterval(timer)
      timer = undefined
      activeKey = null
    }

    function handleKeyDown(event: KeyboardEvent) {
      const direction = KEY_DIRECTIONS[event.key]
      if (!direction) return

      event.preventDefault()

      // Our timer handles repeated movement.
      if (event.repeat || activeKey === event.key) return

      stopMoving()
      activeKey = event.key

      movePlayer(direction)
      timer = setInterval(() => movePlayer(direction), MOVE_INTERVAL)
    }

    function handleKeyUp(event: KeyboardEvent) {
      if (event.key === activeKey) stopMoving()
    }

    function handleVisibilityChange() {
      if (document.hidden) stopMoving()
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)
    window.addEventListener('blur', stopMoving)
    document.addEventListener('visibilitychange', handleVisibilityChange)

    return () => {
      stopMoving()
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
      window.removeEventListener('blur', stopMoving)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [movePlayer])
  const hasWon = player.row === ROWS - 1 && player.col === COLS - 1

  return (
    <div
      className="maze"
      role="group"
      aria-label="Maze"
      style={{
        gridTemplateColumns: `repeat(${COLS}, minmax(0, 1fr))`,
        gridTemplateRows: `repeat(${ROWS}, minmax(0, 1fr))`,
      }}
    >
      {maze.map((row) =>
        row.map((cell) => (
          <div
            key={`${cell.row}-${cell.col}`}
            className="maze-cell"
            style={{
              gridRow: cell.row + 1,
              gridColumn: cell.col + 1,
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

      {!hasWon && (
        <div
          className="maze-cheese"
          role="img"
          aria-label="Cheese goal"
          style={{
            gridRow: ROWS,
            gridColumn: COLS,
          }}
        >
          🧀
        </div>
      )}

      <div
        className="maze-player"
        style={{
          left: `${((player.col + 0.5) / COLS) * 100}%`,
          top: `${((player.row + 0.5) / ROWS) * 100}%`,
          width: `${75 / COLS}%`,
          height: `${75 / ROWS}%`,
        }}
      >
        <img
          src={
            !player.hasMoved || hasWon
              ? '/pavlos_smile.png'
              : '/pavlos_frown.png'
          }
          alt="Pavlos player"
          draggable={false}
          style={{
            transform: `rotate(${hasWon ? 0 : player.rotation}deg)`,
          }}
        />
      </div>
    </div>
  )
}