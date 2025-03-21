# Challenge #2: convolution
*Difficulty: medium*

## Problem
Convolutions are essentially filters that can be applied to an image.
In this challenge, you will implement a simple 2D convolution on the GPU.

We define the convolution of an (RGB) image $I$ with a kernel $K$ on a per-pixel basis ($conv(I, x, y)$).
You can assume that $K$ is a square ($n \cdot n$) matrix, with an odd $n$.
Below is the pseudocode for the convolution of a single pixel:

```
conv(I, x, y):
    result <- (0, 0, 0)
    for i = -n/2 to n/2:
        for j = -n/2 to n/2:
            result.r += I(x+i, y+j).r * K(i, j)
            result.g += I(x+i, y+j).g * K(i, j)
            result.b += I(x+i, y+j).b * K(i, j)
```

In this challenge, we will assume that all pixels outside the image's boundary are black `(0, 0, 0)`.

## Implementation tasks
*All `TODO` markers are in the `kernel.cu` file.*

1. Implement the kernel function itself (`__global__ void convolution_gpu(...)`).
   - We use simple `byte *` pointers to represent the image. You will have to compute the correct index for each pixel's RGB values.
     - Each pixel takes 3 bytes `... r g b ...`.
     - The image is stored in row-major order, meaning that pixel `(x, y)` is next to pixel `(x + 1, y)`.
     - The `__restrict__` keyword is used to tell the compiler that the memory regions of the "restricted" pointers don't overlap. It's just a hint to aid optimization.
   - The convolution itself should be computed using floating-point values (as the kernel values are floating-point numbers).
     - Don't forget to cast the input RGB values to `float` before multiplication. 
     - Don't forget to clamp your outputs to the `[0, 255]` range.
     - Don't forget to re-cast the final result back to `byte`.
   - For the convolution kernel, you can use `conv_kernel::operator()(x, y)` to access the kernel values (e.g. `k(x, y)`).
     - Hint: if you're stuck on the image offsets, you can maybe use this for inspiration?
2. Implement the `c1_convolution::do_convolution(...)` function.
   - This function is responsible for all the CPU-side work:
     1. Setup of GPU resources
     2. Calling the kernel
     3. Cleanup of GPU resources
3. To test your implementation:
   - A driver program is provided again. It takes 3 arguments: the path to the input image, the path to the convolution kernel matrix, and the path to the output image.
   - In the `./examples/` directory, you can find an example input image (`./examples/camel.png`) and a bunch of example kernel matrices. The expected output images are provided in the `./examples/outputs/` directory.