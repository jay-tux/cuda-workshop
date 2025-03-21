#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include "cuda_wrapper.hpp"

using type = unsigned;

__global__ void kernel(const type *__restrict__ a, const type *__restrict__ b, type *__restrict__ c, const size_t n) {
  TODO("get thread index");
  TODO("check if index is within bounds");
  TODO("perform addition");
}

std::mt19937 rng{std::random_device{}()};

int main(const int argc, const char **argv) {
  if (argc != 3 || argv[1] == std::string("-h")) {
    std::cout << "Usage: " << argv[0] << " <vector length> <max>\n";
    return 0;
  }

  char *end = nullptr;
  const size_t n = std::strtoull(argv[1], &end, 10);
  const int max = static_cast<int>(std::strtol(argv[2], &end, 10));

  auto *a = new type[n], *b = new type[n], *c = new type[n];
  for (size_t i = 0; i < n; i++) {
    a[i] = static_cast<type>(rng()) % max;
    b[i] = static_cast<type>(rng()) % max;
  }

  type *a_gpu, *b_gpu, *c_gpu;
  // 1. Allocate GPU buffers
  TODO("allocate GPU buffers");

  // 2. Copy input data to GPU
  TODO("copy input");

  // 3. Perform computation
  //      -> using 1024 threads per block
  constexpr int block_size = 1024;
  //      -> using enough blocks to cover the entire vector
  const int grid_size = TODO("number of blocks");
  //      -> actual invocation
  TODO("GPU kernel invocation");
  //      -> wait for kernel to finish
  TODO("wait for kernel to finish");

  // 4. Copy output data to CPU
  TODO("copy output");

  // 5. Clean up GPU resources
  TODO("clean up");

  constexpr size_t stride = 4;
  for (size_t i = 0; i < n; i += stride) {
    for (size_t j = 0; j < stride && j + i < n; j++) {
      std::printf("%5d + %5d = %5d\t\t", a[i + j], b[i + j], c[i + j]);
    }
    std::printf("\n");
  }

  delete [] a; delete [] b; delete [] c;
}
