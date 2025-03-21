# Challenge #1: matrix multiplication
*Difficulty: easy*

## Problem
Given two matrices $A$ and $B$, each of dimension $n \times n$, compute the matrix product $C = A \cdot B$.
Mathematically, this is defined as
$$
C_{ij} = \sum_{k=1}^n A_{ik} \cdot B_{kj}
$$

(***Before you start coding***) Where is the parallelism in this problem? I.e. what can be run in parallel?

## Implementation tasks
*All `TODO` markers are in the `kernel.cu` file.*

1. Implement the kernel function itself (`__global__ void kernel_entry(...)`). 
    - Recall from the examples how you can get the row and column index of the current thread.
    - You can use the `matrix::at(col, row)` function to access the elements of the matrices (e.g. `a.at(col, row)`).
    - Each (GPU) thread running the `kernel_entry` function should compute one element of the resulting matrix $C$.
      - Mathematically, this means that thread $(row, col)$ should compute $C_{row, col}$.
2. Implement the helper functions as shown during the demo:
    - `__host__ matrix kernel::make_gpu(unsigned n)`: Create a new $n \times n$ matrix on the GPU. Make sure to allocate enough memory!
    - `__host__ matrix kernel::copy_to_gpu(const matrix &cpu)`: Copy a matrix from the CPU to the GPU.
    - `__host__ matrix kernel::copy_to_cpu(const matrix &gpu)`: Copy a matrix from the GPU to the CPU.
    - `__host__ void kernel::cleanup(matrix &gpu)`: Clean up all resources used by a matrix on the GPU.
    - `__host__ matrix kernel::matmul(const matrix &a_gpu, const matrix &b_gpu)`: Compute the matrix product of two matrices on the GPU. Note that, at this point, both `a_gpu` and `b_gpu` live on the GPU.
    - `__host__ matrix kernel::full_program(const matrix &a_cpu, const matrix &b_cpu)`: This (driver) function should combine all the above to compute $a_cpu \cdot b_cpu$ on the GPU. It should return the result as a matrix on the CPU.
3. To test your implementation:
    - A driver `main.cu` file is provided, which takes 2 arguments (the size $n$ of the matrices, and a seed for the PRNG).
    - The `./expected/` directory contains two sample outputs files (for $n=4,s=0$ and $n=16,s=4$). You can compare the output of your implementation with them using `diff`.