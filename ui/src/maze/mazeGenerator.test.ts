import { describe, expect, it } from 'vitest'
import { createGrid, generateMaze } from './mazeGenerator'

describe('mazeGenerator', () => {
  it('creates a grid with the correct dimensions', () => {
    const maze = createGrid(5, 7)

    expect(maze.length).toBe(5)
    expect(maze[0].length).toBe(7)
  })

  it('creates cells with all four walls', () => {
    const maze = createGrid(2, 2)

    expect(maze[0][0].walls).toEqual({
      top: true,
      right: true,
      bottom: true,
      left: true,
    })
  })

  it('visits every cell during maze generation', () => {
    const maze = generateMaze(10, 10)

    const everyCellVisited = maze
      .flat()
      .every((cell) => cell.visited)

    expect(everyCellVisited).toBe(true)
  })

  it('removes walls to create passages', () => {
    const maze = generateMaze(10, 10)

    const hasPassage = maze.flat().some((cell) =>
      Object.values(cell.walls).some((wall) => wall === false),
    )

    expect(hasPassage).toBe(true)
  })

  it('keeps walls consistent between neighboring cells', () => {
    const maze = generateMaze(10, 10)

    for (let row = 0; row < maze.length; row++) {
      for (let col = 0; col < maze[row].length; col++) {
        const cell = maze[row][col]

        // Compare right wall with next cell's left wall
        if (col < maze[row].length - 1) {
          expect(cell.walls.right).toBe(
            maze[row][col + 1].walls.left,
          )
        }

        // Compare bottom wall with next cell's top wall
        if (row < maze.length - 1) {
          expect(cell.walls.bottom).toBe(
            maze[row + 1][col].walls.top,
          )
        }
      }
    }
  })
})