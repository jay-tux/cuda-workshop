//
// Created by jay on 3/20/25.
//

#include "kernel.hpp"
#include "cuda_wrapper.hpp"

__host__ __device__ int &matrix::at(const unsigned col, const unsigned row) {
  return data[row * n + col];
}

__host__ __device__ const int &matrix::at(const unsigned col, const unsigned row) const {
  return data[row * n + col];
}

__global__ void kernel_entry(const matrix a, const matrix b, matrix c) {
  TODO("matrix multiplication kernel (GPU-side)")
}

__host__ matrix kernel::make_gpu(unsigned n) {
  TODO("GPU-side matrix allocation")
}


__host__ matrix kernel::copy_to_gpu(const matrix& cpu) {
  TODO("CPU-to-GPU matrix copy")
}

__host__ matrix kernel::copy_to_cpu(const matrix& gpu) {
  TODO("CPU-to-GPU matrix copy")
}

__host__ void kernel::matmul(const matrix& a_gpu, const matrix& b_gpu, matrix& c_gpu) {
  TODO("CPU-side kernel launch")
}

__host__ void kernel::cleanup(matrix& gpu) {
  TODO("CPU-side matrix cleanup (of GPU resources)")
}

__host__ matrix kernel::full_program(const matrix& a_cpu, const matrix& b_cpu) {
  TODO("combine kernel::* methods to run the full matrix multiply program")
}
