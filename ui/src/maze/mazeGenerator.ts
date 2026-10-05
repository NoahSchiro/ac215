// Defines the structure of one cell in the maze.
export interface Cell {
  // Position of the cell within the maze grid.
  row: number
  col: number

  // Tracks whether the maze generation algorithm has already visited this cell.
  visited: boolean

  // Each boolean represents whether that wall currently exists.
  // true = wall exists, false = wall has been removed.
  walls: {
    top: boolean
    right: boolean
    bottom: boolean
    left: boolean
  }
}

// A Maze is represented as a 2D array of Cell objects.
export type Maze = Cell[][]


// Creates the initial maze grid before any paths have been carved.
export function createGrid(rows: number, cols: number): Maze {
  // Create the requested number of rows.
  return Array.from({ length: rows }, (_, row) =>
    // For each row, create the requested number of columns.
    Array.from({ length: cols }, (_, col) => ({
      // Store the location of each cell.
      row,
      col,

      // No cells have been explored yet.
      visited: false,

      // Every cell begins with all four walls intact.
      walls: {
        top: true,
        right: true,
        bottom: true,
        left: true,
      },
    })),
  )
}


// Finds all neighboring cells that have not yet been visited.
// These are the possible cells the DFS algorithm can move to next.
function getUnvisitedNeighbors(
  maze: Maze,
  cell: Cell,
): Cell[] {
  // Stores all valid unvisited neighbors.
  const neighbors: Cell[] = []

  // Get the current cell's position.
  const { row, col } = cell

  // Get the dimensions of the maze.
  const rows = maze.length
  const cols = maze[0].length

  // Check the cell above.
  // row > 0 prevents us from going outside the top of the maze.
  if (row > 0 && !maze[row - 1][col].visited) {
    neighbors.push(maze[row - 1][col])
  }

  // Check the cell to the right.
  // col < cols - 1 prevents us from going outside the right edge.
  if (col < cols - 1 && !maze[row][col + 1].visited) {
    neighbors.push(maze[row][col + 1])
  }

  // Check the cell below.
  // row < rows - 1 prevents us from going outside the bottom.
  if (row < rows - 1 && !maze[row + 1][col].visited) {
    neighbors.push(maze[row + 1][col])
  }

  // Check the cell to the left.
  // col > 0 prevents us from going outside the left edge.
  if (col > 0 && !maze[row][col - 1].visited) {
    neighbors.push(maze[row][col - 1])
  }

  // Return all neighboring cells that DFS is allowed to visit.
  return neighbors
}


// Removes the wall between two adjacent cells.
function removeWall(current: Cell, next: Cell): void {
  // Determine where the next cell is relative to the current cell.
  const rowDifference = next.row - current.row
  const colDifference = next.col - current.col

  // If next is above current:
  // remove current's top wall and next's bottom wall.
  if (rowDifference === -1) {
    current.walls.top = false
    next.walls.bottom = false
  }

  // If next is to the right:
  // remove current's right wall and next's left wall.
  else if (colDifference === 1) {
    current.walls.right = false
    next.walls.left = false
  }

  // If next is below current:
  // remove current's bottom wall and next's top wall.
  else if (rowDifference === 1) {
    current.walls.bottom = false
    next.walls.top = false
  }

  // If next is to the left:
  // remove current's left wall and next's right wall.
  else if (colDifference === -1) {
    current.walls.left = false
    next.walls.right = false
  }
}


// Generates the maze using randomized depth-first search (DFS)
// with backtracking.
export function generateMaze(rows: number, cols: number): Maze {
  // Start with a grid where every cell has all four walls.
  const maze = createGrid(rows, cols)

  // Begin maze generation from the top-left cell.
  const start = maze[0][0]
  start.visited = true

  // The stack keeps track of the current DFS path.
  // It also allows us to backtrack when we reach a dead end.
  const stack: Cell[] = [start]

  // Continue until DFS has completely explored the maze.
  while (stack.length > 0) {
    // The last cell in the stack is our current position.
    const current = stack[stack.length - 1]

    // Find neighboring cells that have not been visited yet.
    const neighbors = getUnvisitedNeighbors(maze, current)

    // If at least one unvisited neighbor exists, continue forward.
    if (neighbors.length > 0) {
      // Randomly select one of the available neighbors.
      // This randomness causes different maze layouts to be generated.
      const randomIndex = Math.floor(Math.random() * neighbors.length)
      const next = neighbors[randomIndex]

      // Carve a passage between the current cell and selected neighbor.
      removeWall(current, next)

      // Mark the new cell as visited so DFS does not process it again.
      next.visited = true

      // Add the new cell to the DFS path.
      stack.push(next)
    } else {
      // If there are no available neighbors, we reached a dead end.
      // Remove the current cell from the stack to backtrack.
      stack.pop()
    }
  }

  // Create an entrance on the left side of the top-left cell.
  maze[0][0].walls.left = false

  // Create an exit on the right side of the bottom-right cell.
  maze[rows - 1][cols - 1].walls.right = false

  return maze
}