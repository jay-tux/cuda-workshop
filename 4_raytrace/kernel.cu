//
// Created by jay on 3/18/25.
//

#include "kernel.hpp"
#include "float3.hpp"

#include <cuda_wrapper.hpp>

__device__ constexpr float clamp(const float x, const float min, const float max) {
  TODO("Return the value of x clamped to the range [min, max] (both inclusive)");
}

__device__ void image::set_pixel(const int x, const int y, const float3 color) {
  TODO("Set the pixel at coordinates (x, y) to the provided color. Don't forget to clamp the color values to the range [0, 1], and convert them to byte values.");
}

__host__ void image::cleanup() {
  TODO("Clean up the pointer");
  data = nullptr;
  width = 0; height = 0;
}

template <uint8_t bounces_left>
__device__ float3 colorizer(const scene_gpu &scene, const ray &r) {
  float dist{};
  size_t hit{};
  float3 normal{};
  material mat{};
  float3 color = {0, 0, 0}; // black

  if (kernel::intersect(scene, r, dist, hit, normal, mat)) {
    float3 hit_at, view_direction;
    TODO("Compute the hit_at and view_direction vectors");
    for (size_t p_id = 0; p_id < scene.num_points; p_id++) {
      color = color + scene.points[p_id].shade(mat, hit_at, normal, view_direction);
    }

    if constexpr(bounces_left > 0) { // Make sure we don't exceed the maximum number of bounces
      if (mat.reflect_factor > epsilon) {
        ray reflected_ray;
        TODO("Compute the reflected ray");
        const float3 reflect_color = colorizer<bounces_left - 1>(scene, reflected_ray);

        color = lerp(color, reflect_color, mat.reflect_factor);
      }

      if (mat.transparency > epsilon) {
        ray passthrough_ray;
        TODO("Compute the passthrough ray");
        const float3 passthrough_color = colorizer<bounces_left - 1>(scene, passthrough_ray);

        color = lerp(color, passthrough_color, mat.transparency);
      }
    }
  }

  return color;
}

__global__ void dump(const scene_gpu scene) { scene.dump_gpu(); }

__global__ void renderer(const scene_gpu scene, image img, kernel::mode m) {
  int x, y;
  TODO("Get the x and y coordinates of the pixel to render");

  ray r;
  TODO("Construct the ray that goes through the pixel");

  float dist = INFINITY;
  size_t hit{};
  float3 _n{};
  material _m{};

  if (m == kernel::mode::ID || m == kernel::mode::DIST) {
    if (kernel::intersect(scene, r, dist, hit, _n, _m)) {
      if (m == kernel::mode::ID) {
        img.set_pixel(x, y, color_to_id[hit % 32]);
      }
      else { // mode::DIST
        img.set_pixel(x, y, {1.0f / dist, 1.0f / dist, 1.0f / dist });
      }
    }
    else {
      img.set_pixel(x, y, {0, 0, 0}); // black
    }
  }
  else { // mode::COLOR
    img.set_pixel(x, y, colorizer<bounces>(scene, r));
  }
}

__host__ void kernel::render(const scene_gpu& scene, image &img, const mode m) {
  // cuCheckAsync((dump<<<1, 1>>>(scene))); // You can use this to check if the scene got copied to the GPU correctly

  TODO("Launch the renderer kernel with the correct block and grid dimensions");
}

__device__ bool kernel::intersect(const scene_gpu& scene, const ray& ray, float& dist, size_t& obj_id, float3 &normal, material& mat) {
  dist = INFINITY;

  TODO("Check if any planes intersect the ray. If the intersection is the closest so far, update dist, obj_id, normal, and mat.");
  TODO("Check if any spheres intersect the ray. If the intersection is the closest so far, update dist, obj_id, normal, and mat.");
  TODO("Check if any triangles intersect the ray. If the intersection is the closest so far, update dist, obj_id, normal, and mat.");

  return dist != INFINITY;
}

