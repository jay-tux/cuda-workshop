# Starter code for the CUDA workshop

In this repository, you will find the starter code for both the live-coding examples (0a and 0b), and the four challenges (1, 2, 3, and 4):
- Examples:
  - `0a_vec_add`: Simple vector addition
  - `0b_grayscale`: Loads in an image and converts it to grayscale using GIMP's lightness formula
- Challenges:
  - `1_matmul`: Matrix multiplication
  - `2_convolution`: 2D convolution
  - `3_game_of_life`: Conway's Game of Life simulation
  - `4_raytrace`: A simple raytracer
  
(The names of these correspond to the CMake targets, and the directories in which the code and, in the case of the challenges, the challenge assignment can be found.)

Recommended development environment:
- CLion IDE (with the Conan plugin for dependency management)
- Conan package manager (for dependency management)
- CMake 3.28 or newer
- CUDA 12 or newer (tested on GTX 1080 and RTX 4070)

Target audience: this workshop is aimed at people who have some (limited) experience with C++, and are interested in GPGPU (general purpose GPU computing) using CUDA.

This repository has a [companion repository](https://github.com/jay-tux/cuda-workshop-solutions), which contains both the solutions and the slides for the workshop host.
Additionally, you can find some slides with more background in [this repository](https://github.com/jay-tux/cuda-presentation).