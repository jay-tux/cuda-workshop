//
// Created by jay on 3/18/25.
//

#include <iostream>

#include "scene.hpp"
#include "float3.hpp"
#include "cuda_wrapper.hpp"
#include "matrix.hpp"
#include "kernel.hpp"

void scene_cpu::setup_cam() {
  cam_forward = normalize(cam_forward);
  cam_up = normalize(cam_up);
  cam_right = cross(cam_up, cam_forward);
  cam_up = cross(cam_forward, cam_right);
}

__device__ bool plane::intersect(const ray &ray, float &t, float3 &normal) const {
  TODO("check if the ray is parallel to the plane");
  TODO("calculate the intersection point");
  float t;

  TODO("colorizing stage: calculate the normal at the intersection point");

  return t >= 0.0f;
}

__device__ bool sphere::intersect(const ray &ray, float &t, float3 &normal) const {
  float t0, t1;
  TODO("Calculate t0, t1");
  bool t0_valid, t1_valid;
  TODO("Check which of t0, t1 are valid");

  if (t0_valid) {
    if (t1_valid) { // t0, t1 both okay
      t = min(t0, t1);
    }
    else { // t0 okay, t1 not okay
      t = t0;
    }
  }
  else {
    if (t1_valid) { // t0 not okay, t1 okay
      t = t1;
    }
    else { // neither okay
      return false;
    }
  }

  TODO("Coloring stage: calculate the normal at the intersection point");
  return true;
}

__device__ bool triangle::intersect(const ray &ray, float &t, float3 &normal) const {
  TODO("Compute alpha, beta, gamma");
  TODO("Check if the intersection point is inside the triangle, and if so, set t");

  TODO("Colorizing stage: calculate the normal at the intersection point");
}

__device__ float3 point_light::shade(const material &mat, const float3 &point, const float3 &normal, const float3 &view_dir) const {
  TODO("compute the diffuse component");
  float3 diffuse;
  TODO("compute the specular component");
  float3 specular;
  TODO("compute the attenuation of the light")
  float attenuation;

  return intensity * attenuation * (diffuse + specular);
}

__host__ void scene_gpu::cleanup() {
  TODO("Warmup...");
}

__host__ scene_gpu scene_cpu::to_gpu() const {
  TODO("Warmup...");
}

std::ostream &operator<<(std::ostream &os, const float3 &v) {
  return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
}

__host__ void scene_cpu::dump() const {
  std::cout << " === Scene (CPU) ===\n"
            << "Camera data:\n"
            << "  -> Position: " << camera << "\n"
            << "  -> Forward: " << cam_forward << "\n"
            << "  -> Up: " << cam_up << "\n"
            << "  -> Right: " << cam_right << "\n"
            << "\n"
            << "  -> Planes (" << planes.size() << "):\n";
  for (size_t i = 0; i < planes.size(); i++) {
    const auto & [point, normal, id, material_id] = planes[i];
    std::cout << "    -> Plane #" << i << " [ID " << id << "]:\n"
              << "      -> Point: " << point << "\n"
              << "      -> Normal: " << normal << "\n"
              << "      -> Material ID: " << material_id << "\n";
  }
  std::cout << "\n"
            << "  -> Spheres (" << spheres.size() << "):\n";
  for (size_t i = 0; i < spheres.size(); i++) {
    const auto & [center, radius, id, material_id] = spheres[i];
    std::cout << "    -> Sphere #" << i << " [ID " << id << "]:\n"
              << "      -> Center: " << center << "\n"
              << "      -> Radius: " << radius << "\n"
              << "      -> Material ID: " << material_id << "\n";
  }
  std::cout << "\n"
              << "  -> Triangles (" << triangles.size() << "):\n";
  for (size_t i = 0; i < triangles.size(); i++) {
    const auto & [v0, v1, v2, normal, id, material_id] = triangles[i];
    std::cout << "    -> Triangle #" << i << " [ID " << id << "]:\n"
              << "      -> v0: " << v0 << "\n"
              << "      -> v1: " << v1 << "\n"
              << "      -> v2: " << v2 << "\n"
              << "      -> Normal: " << normal << "\n"
              << "      -> Material ID: " << material_id << "\n";
  }
  std::cout << "\n"
            << "  -> Materials (" << materials.size() << "):\n";
  for (size_t i = 0; i < materials.size(); i++) {
    const auto &[color, phong_exponent, reflect_factor, transparency, id] = materials[i];
    std::cout << "    -> Material #" << i << " [ID " << id << "]:\n"
              << "      -> Color: " << color << "\n"
              << "      -> Phong exponent: " << phong_exponent << "\n"
              << "      -> Reflection factor: " << reflect_factor << "\n"
              << "      -> Transparency: " << transparency << "\n";
  }
  std::cout << "\n"
            << "  -> Point lights (" << points.size() << "):\n";
  for (size_t i = 0; i < points.size(); i++) {
    const auto &[point, color, attenuation, intensity] = points[i];
    std::cout << "    -> Point light #" << i << ":\n"
              << "      -> Point: " << point << "\n"
              << "      -> Color: " << color << "\n"
              << "      -> Attenuation: " << attenuation << "\n"
              << "      -> Intensity: " << intensity << "\n";
  }

  std::cout << "=== End Scene ===\n";
}

__host__ void scene_gpu::dump() const {
  std::cout << " === Scene (GPU; host) ===\n"
            << "Camera data:\n"
            << "  -> Position: " << camera << "\n"
            << "  -> Forward: " << cam_forward << "\n"
            << "  -> Up: " << cam_up << "\n"
            << "  -> Right: " << cam_right << "\n"
            << "\n"
            << "  -> Planes (" << num_planes << "):\n";
  for (size_t i = 0; i < num_planes; i++) {
    const auto & [point, normal, id, material_id] = planes[i];
    std::cout << "    -> Plane #" << i << " [ID " << id << "]:\n"
              << "      -> Point: " << point << "\n"
              << "      -> Normal: " << normal << "\n"
              << "      -> Material ID: " << material_id << "\n";
  }
  std::cout << "\n"
            << "  -> Spheres (" << num_spheres << "):\n";
  for (size_t i = 0; i < num_spheres; i++) {
    const auto & [center, radius, id, material_id] = spheres[i];
    std::cout << "    -> Sphere #" << i << " [ID " << id << "]:\n"
              << "      -> Center: " << center << "\n"
              << "      -> Radius: " << radius << "\n"
              << "      -> Material ID: " << material_id << "\n";
  }
  std::cout << "\n"
              << "  -> Triangles (" << num_triangles << "):\n";
  for (size_t i = 0; i < num_triangles; i++) {
    const auto & [v0, v1, v2, normal, id, material_id] = triangles[i];
    std::cout << "    -> Triangle #" << i << " [ID " << id << "]:\n"
              << "      -> v0: " << v0 << "\n"
              << "      -> v1: " << v1 << "\n"
              << "      -> v2: " << v2 << "\n"
              << "      -> Normal: " << normal << "\n"
              << "      -> Material ID: " << material_id << "\n";
  }
  std::cout << "\n"
          << "  -> Materials (" << num_materials << "):\n";
  for (size_t i = 0; i < num_materials; i++) {
    const auto &[color, phong_exponent, reflect_factor, transparency, id] = materials[i];
    std::cout << "    -> Material #" << i << " [ID " << id << "]:\n"
              << "      -> Color: " << color << "\n"
              << "      -> Phong exponent: " << phong_exponent << "\n"
              << "      -> Reflection factor: " << reflect_factor << "\n"
              << "      -> Transparency: " << transparency << "\n";
  }
  std::cout << "\n"
            << "  -> Point lights (" << num_points << "):\n";
  for (size_t i = 0; i < num_points; i++) {
    const auto &[point, color, attenuation, intensity] = points[i];
    std::cout << "    -> Point light #" << i << ":\n"
              << "      -> Point: " << point << "\n"
              << "      -> Color: " << color << "\n"
              << "      -> Attenuation: " << attenuation << "\n"
              << "      -> Intensity: " << intensity << "\n";
  }

  std::cout << "=== End Scene ===\n";
}

__device__ void scene_gpu::dump_gpu() const {
  printf(" === Scene (GPU; device) ===\n"
            "Camera data:\n"
            "  -> Position: (%f, %f, %f)\n"
            "  -> Forward: (%f, %f, %f)\n"
            "  -> Up: (%f, %f, %f)\n"
            "  -> Right: (%f, %f, %f)\n"
            "\n"
            "  -> Planes (%ld):\n",
            camera.x, camera.y, camera.z,
            cam_forward.x, cam_forward.y, cam_forward.z,
            cam_up.x, cam_up.y, cam_up.z,
            cam_right.x, cam_right.y, cam_right.z,
            num_planes
  );
  for (size_t i = 0; i < num_planes; i++) {
    const auto & [point, normal, id, material_id] = planes[i];
    const auto & [r, g, b] = color_to_id[id % 32];
    printf("    -> Plane #%ld [ID %ld (%f, %f, %f)]:\n"
              "      -> Point: (%f, %f, %f)\n"
              "      -> Normal: (%f, %f, %f)\n"
              "      -> Material ID: %ld\n",
              i, id, r, g, b,
              point.x, point.y, point.z,
              normal.x, normal.y, normal.z,
              material_id
    );
  }

  printf("\n  -> Spheres (%ld):\n", num_spheres);
  for (size_t i = 0; i < num_spheres; i++) {
    const auto & [center, radius, id, material_id] = spheres[i];
    const auto & [r, g, b] = color_to_id[id % 32];
    printf("    -> Sphere #%ld [ID %ld (%f, %f, %f)]:\n"
              "      -> Center: (%f, %f, %f)\n"
              "      -> Radius: %f\n"
              "      -> Material ID: %ld\n",
              i, id, r, g, b,
              center.x, center.y, center.z,
              radius,
              material_id
    );
  }

  printf("\n  -> Triangles (%ld):\n", num_triangles);
  for (size_t i = 0; i < num_triangles; i++) {
    const auto & [v0, v1, v2, normal, id, material_id] = triangles[i];
    const auto & [r, g, b] = color_to_id[id % 32];
    printf("    -> Triangle #%ld [ID %ld (%f, %f, %f)]:\n"
              "      -> v0: (%f, %f, %f)\n"
              "      -> v1: (%f, %f, %f)\n"
              "      -> v2: (%f, %f, %f)\n"
              "      -> Normal: (%f, %f, %f)\n"
              "      -> Material ID: %ld\n",
              i, id, r, g, b,
              v0.x, v0.y, v0.z,
              v1.x, v1.y, v1.z,
              v2.x, v2.y, v2.z,
              normal.x, normal.y, normal.z,
              material_id
    );
  }

  printf("\n  -> Materials (%ld):\n", num_materials);
  for (size_t i = 0; i < num_materials; i++) {
    const auto &[color, phong_exponent, reflect_factor, transparency, id] = materials[i];
    printf("    -> Material #%ld [ID %ld]:\n"
              "      -> Color: (%f, %f, %f)\n"
              "      -> Phong exponent: %f\n"
              "      -> Reflection factor: %f\n"
              "      -> Transparency: %f\n",
              i, id,
              color.x, color.y, color.z,
              phong_exponent, reflect_factor, transparency
    );
  }
  printf("\n  -> Point lights (%ld):\n", num_points);
  for (size_t i = 0; i < num_points; i++) {
    const auto &[point, color, attenuation, intensity] = points[i];
    printf("    -> Point light #%ld:\n"
              "      -> Point: (%f, %f, %f)\n"
              "      -> Color: (%f, %f, %f)\n"
              "      -> Attenuation: (%f, %f, %f)\n"
              "      -> Intensity: %f\n",
              i,
              point.x, point.y, point.z,
              color.x, color.y, color.z,
              attenuation.x, attenuation.y, attenuation.z,
              intensity
    );
  }

  printf("=== End Scene ===\n");
}