//
// Created by jay on 3/18/25.
//

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <iostream>
#include "cuda_wrapper.hpp"

using byte = unsigned char;

struct pixel { byte r, g, b; };

__host__ __device__ pixel pixel_at(const byte *data, const int w, const int x, const int y) {
  const byte r = data[3 * (w * y + x) + 0];
  const byte g = data[3 * (w * y + x) + 1];
  const byte b = data[3 * (w * y + x) + 2];
  return {r, g, b};
}

__global__ void kernel(const byte *__restrict__ data, byte *__restrict__ out, int w, int h) {
  // GIMP lightness formula: 1/2 * (max(R, G, B) + min(R, G, B))
  const int x = TODO("calculate x-coordinate");
  const int y = TODO("calculate y-coordinate");
  if (x >= w || y >= h) return;

  const byte lightness = TODO("lightness formula");
  out[w * y + x] = lightness;
}

int main(const int argc, const char **argv) {
  if (argc != 3 || argv[1] == std::string("-h")) {
    std::cout << "Usage: " << argv[0] << " <input image file> <output image file>\n";
    return 0;
  }

  // load image
  int w, h, n;
  byte *data_cpu = stbi_load(argv[1], &w, &h, &n, 3);
  if (!data_cpu) {
    std::cerr << "Could not load image: " << argv[1] << "\n";
    return 1;
  }

  byte *data_gpu, *out;
  // 1. Allocate GPU buffers
  TODO("allocate GPU buffers");

  // 2. Copy input data to GPU
  TODO("copy input data");

  // 3. Perform computation
  //      -> using 1024 threads per block, but in 2D
  constexpr dim3 block_size{32, 32}; // 32 * 32 = 1024
  //      -> using enough blocks to cover the entire image
  const dim3 grid_size{TODO("calculate 2D grid size")};
  TODO("kernel invocation");

  // 4. Copy output data to CPU
  // ! This is not needed -> unified memory allocated by cudaMallocManaged is accessible by both CPU and GPU

  stbi_write_png(argv[2], w, h, 1, out, w);

  // 5. Clean up GPU resources
  // ! We have to wait with freeing the output buffer until the output image is written!
  TODO("free GPU buffers");
  stbi_image_free(data_cpu);
}