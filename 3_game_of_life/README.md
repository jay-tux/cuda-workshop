# Challenge #3: Conway's Game of Life
*Difficulty: medium*

## Problem
Conway's Game of Life is a 0-player game (cellular automaton), where the state of the "game" evolves over time.
Normally, we use an infinite grid, but for simplicity, this challenge assumes a grid of the same size as your screen (in terminal cells).
Each cell is either alive or dead.
For every cell $C_{ij}$ in the grid, only the immediate neighbors of $C_{ij}$ can affect its state:
- If $C_{ij}$ is alive, and it has 2 or 3 alive neighbors, it survives. Otherwise, it dies.
- If $C_{ij}$ is dead, and it has exactly 3 alive neighbors, it becomes alive.

Below is a table with what we consider neighbors of $C_{ij}$:

| $C_{i-1,j-1}$ | $C_{i,j-1}$ | $C_{i+1,j-1}$ |
|---------------|-------------|---------------|
| $C_{i-1,j}$   | $C_{ij}$    | $C_{i+1,j}$   |
| $C_{i-1,j+1}$ | $C_{i,j+1}$ | $C_{i+1,j+1}$ |

For the implementation, we will use "wrap-around", meaning that the top row is adjacent to the bottom one, and the leftmost column is adjacent to the rightmost one.
Finally, we use a double-buffered approach, meaning that we have two grids: one for the current (input) state, and one for the next (output) state.

## Implementation tasks
1. In the `kernel.hpp` file, implement the `c2_game_of_life::buffer::at` function.
   - Given a row and column, this function should return a pointer to the cell in the buffer.
   - Remember that we use wrap-around, so you should handle the edge cases.
   - The buffer is stored row-major, where each cell is a single byte (`char`).
2. In the `kernel.cu` file, implement the state-transition kernel itself (`__global__ void do_step(...)`).
   - At this point, you should be able to determine what part of the work each thread in the kernel should do.
   - You can use the `c2_game_of_life::buffer::at` function (as `in.at()` or `out.at()`) you implemented in the previous step.
   - Do not modify the `in` buffer, but rather write your output to the `out` buffer.
3. In the `main.cu` file, you still need to clean up two `TODO()`'s in the `main` function.
   - The first is responsible for allocating the buffers - don't forget that we need to access the memory on both CPU and GPU.
     - You're not allowed to add `cudaMemcpy` calls to this file.
   - The second is responsible for cleaning up said buffers at the end of the program.