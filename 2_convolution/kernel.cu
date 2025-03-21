//
// Created by jay on 3/12/25.
//

#include "cuda_wrapper.hpp"
#include "kernel.hpp"

using namespace c1_convolution;

__global__ void convolution_gpu(const byte *__restrict__ image, byte *__restrict__ out, const int i_w, const int i_h, const conv_kernel kernel) {
  TODO("get x,y coordinates and check if they are within bounds");

  TODO("use 2 nested for-loops to compute the convolution for 1 pixel");

  TODO("don't forget to clamp your output values to [0, 255]");

  TODO("store the result");
}

__host__ void c1_convolution::do_convolution(const image &in_cpu, const image &out_cpu, const conv_kernel &kernel_cpu) {
  constexpr auto block = dim3(32, 32);
  const auto grid = dim3(in_cpu.w / block.x + 1, in_cpu.h / block.y + 1);

  byte *in_gpu, *out_gpu; float *kernel_gpu;
  TODO("driver program which does all CPU-side setup, calls the kernel, and does all CPU-side cleanup (don't forget to copy to the output image)");
}