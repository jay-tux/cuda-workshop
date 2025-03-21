# Challenge #4: Raytracing
*Difficulty: hard*

## Problem
In this challenge, we're going to implement a simple raytracer.
Raytracing is a rendering technique where we "shoot" rays from the camera into the scene, bouncing off of the objects, and gradually computing the color for a pixel.
As there's quite a bit of math involved as well, we'll go through it bit by bit.

## Implementation tasks
To test your application, you can use the provided starting code, when compiled to an executable.
To run this executable, pass it (in order) the input scene file, the output image file (PNG), and the mode.
There are three modes to run:
- `id`: renders the scene with the object ID for each hit (a color per object);
- `dist`: renders the scene with the distance to the hit point (a grayscale image, where white is closer and black is further);
- `color`: the full rendering pass. This will only work when all tasks are finished.

### Warmup tasks
1. In `scene.cu`, implement `scene_cpu::to_gpu()`. This function should simply copy the entire scene to the GPU.
2. In `scene.cu`, implement `scene_gpu::cleanup()`. This function should clean up the resources used on the GPU by the scene.
3. In `kernel.cu`, there are two helper functions (`clamp(...)` and `image::set_pixel(...)`). Implement these functions.
4. In `kernel.cu`, implement `kernel::render(...)`. This function should call through to the `renderer(...)` function (but on the GPU).

### Raycasting tasks
This is the first part - we need to be able to detect if a given ray intersects the scene.
We represent a ray as $P + t\vec{D}$, with $P$ being the starting point, and $\vec{D}$ being the direction of vector.
A point $X$ is on the ray as long as it satisfies $P + t\vec{D} = X$, with $t \geq 0$.

#### Computing the ray
*This has to be implemented in `renderer(...)` in `kernel.cu`, by writing to the local variable `r`.*

The first step is to compute the ray for a given pixel.
The starting point of the ray is the same for each pixel (the camera position).
After computing the x- and y-coordinates of the pixel, we need to compute the direction vector $\vec{D}$.

For this, we have the forward (`scene.cam_forward`), right (`scene.cam_right`), and up (`scene.cam_up`) vectors of the camera.
The rough steps for this task are:
1. Get the x- and y-coordinates;
2. Compute the aspect ratio of the screen (based on width and height);
3. Compute the "right"-factor of the ray (based on the ratio between the x-coordinate and the image width, don't forget to use the aspect ratio);
4. Compute the "up"-factor of the ray (based on the ratio between the y-coordinate and the image height);
5. Combine all vectors to get the direction vector $\vec{D}$ (by a simple sum).
6. **Important:** Normalize the direction vector.

Everything else in the `renderer` function is already implemented.

#### Ray-plane intersection
*This has to be implemented as `plane::intersect(...)` in `scene.cu`.*

We represent a plane as $(O, \vec{N})$, with $O$ any point on the plane, and $\vec{N}$ the normal of the plane.

Because planes are infinite, we only need to check two things:
1. Is the ray parallel to the plane? If so, there are no intersections.
2. Is the intersection point behind the origin of the ray? If so, there are no intersections.

If both $\vec{D}$ and $\vec{N}$ are normalized, we can use the following formula to check if they are parallel:
$$
\vec{D}\cdot\vec{N} = 0
$$

After that check, we can compute the intersection point as:
$$
t = \frac{\vec{N}\cdot(O - P)}{\vec{D}\cdot\vec{N}}
$$

If $t \geq 0$, we have an intersection.

*Note: due to floating-point rounding, you want to consider the predefined `epsilon` value instead of 0 itself.*  
*Note: one of the TODO's in the function is for the colorizing stage. We will defer this to a later part of the challenge.*

#### Ray-sphere intersection
*This has to be implemented as `sphere::intersect(...)` in `scene.cu`.*

We represent a sphere as $(C, r)$, with $C$ being the center point and $r$ being the radius.
The intersection between the ray and the sphere are all points $X$ (0, 1, or 2) satisfying the following system of equations:
$$
\begin{cases}
    P + t\vec{D} = X \\
    \|X - C\|^2 = r^2
\end{cases} \iff (P + t\vec{D} - C) \cdot (P + t\vec{D} - C) = r^2
$$

To solve this, we need the roots of the quadratic equation $\vec{D}\cdot\vec{D}t^2 + 2\vec{D}\cdot(P - C)t + (P - C)\cdot(P - C) - r^2 = 0$.
After simplifying, we end up with
$$
\begin{align*}
d &= (-\vec{D})\cdot(P - C) \\
s &= d^2 - (\vec{D}\cdot\vec{D})((P - C)\cdot(P - C) - r^2) \\
\\
t_0 &= \frac{d - \sqrt{s}}{\vec{D}\cdot\vec{D}} \\
t_1 &= \frac{d + \sqrt{s}}{\vec{D}\cdot\vec{D}}
\end{align*}
$$

We need to make sure that the intersection points are in front of the ray origin (i.e. $t_0, t_1 \geq 0$).
We discard those who are not in front of the ray, and return the closest one.

*Note: due to floating-point rounding, you want to consider the predefined `epsilon` value instead of 0 itself, and you need to ensure that both $t_0, t_1$ are finite (using `isfinite()`).*  
*Note: one of the TODO's in the function is for the colorizing stage. We will defer this to a later part of the challenge.* 

#### Ray-triangle intersection
*This has to be implemented as `triangle::intersect(...)` in `scene.cu`.*

We represent a triangle as $(V0, V1, V2)$, with $V0, V1, V2$ being the vertices of the triangle.
The mathematical foundation for this is a bit more complex.
The main idea is that we first compute the intersection point with the plane of the triangle, and then check if that point is inside the triangle.

As the goal of this challenge is not to do a lot of math, we will provide you with the algorithm (known as the Moeller-Trumbore algorithm):
$$
\begin{align*}
a &= V1 - V0 \\
b &= V1 - V2 \\
c &= \vec{D} \\
d &= V1 - P \\
\\
A &= \begin{pmatrix} a_x & b_x & d_x \\ a_y & b_y & d_y \\ a_z & b_z & d_z \end{pmatrix} \\
B &= \begin{pmatrix} a_x & b_x & c_x \\ a_y & b_y & c_y \\ a_z & b_z & c_z \end{pmatrix} \\
A1 &= \begin{pmatrix} d_x & b_x & c_x \\ d_y & b_y & c_y \\ d_z & b_z & c_z \end{pmatrix} \\
A2 &= \begin{pmatrix} a_x & d_x & c_x \\ a_y & d_y & c_y \\ a_z & d_z & c_z \end{pmatrix} \\
\\
\alpha &= \det(B) \\
\beta &= \frac{\det(A1)}{\alpha} \\
\gamma &= \frac{\det(A2)}{\alpha} \\
t &= \frac{\det(A)}{\alpha}
\end{align*}
$$

The intersection is valid if $\beta \geq 0 \land \gamma \geq 0 \land \beta + \gamma \leq 1 \land t \geq 0$.
If so, $t$ is the distance from the ray origin to the intersection point.

*Note: due to floating-point rounding, you want to consider the predefined `epsilon` value instead of 0 itself, and you need to ensure that $t$ is finite (using `isfinite()`).*  
*Note: you can use the `matrix` struct to represent the matrices (their constructor takes columns, and they have a `matrix::determinant` member function).*  
*Note: one of the TODO's in the function is for the colorizing stage. We will defer this to a later part of the challenge.*



#### Raycasting driver
In `kernel.cu`, finish the `kernel::intersect(...)` function.

1. Loop over all planes, checking for intersections.
2. Loop over all spheres, checking for intersections.
3. Loop over all triangles, checking for intersections.

Importantly, you need to keep track of:
- The $t$ value of the closest intersection point (write the final value to the reference-parameter `dist`).
- The object information for the closest hit:
  - The object ID (which you can get from the plane/sphere/triangle as `obj.id);
  - The normal at the intersection point (use pass-by-reference for this, and let the intersect function of the plane/sphere/triangle figure it out (see later tasks));
  - The material of the object (which you can get by using `scene.materials[obj.material_idx]`).

**Important:** use local variables, don't always overwrite your "final" result.

***After this, you can already use the `id` and `dist` modes.** (However, you might have to comment out some TODO's...)*

### Shading tasks
In this second part of the challenge, we will implement the shading of the scene (based on Phong-shading).

#### Normal computation
To shade the scene, we need the normal to each object at the point of the hit. 
The normal is a vector pointing perpendicularly outward of the object.

To do this, modify the `plane::intersect`, `sphere::intersect`, and `triangle::intersect` functions in `scene.cu`.
For the `plane` and `triangle`, you can use the member variable `normal` directly (copying it to the by-reference parameter `normal`; you can only really access it using `this->normal`!).
For the `sphere`, the normal is a unit vector pointing away from the center through the intersection point.

#### Phong shading
*This has to be implemented in `point_light::shade(...)` in `scene.cu`.*

The Phong shading model is a simple model to compute the color of a point on a surface.
It requires looping over all lights in the scene (we only use point-lights), computing their influence on the final result, and adding them up.
The looping and adding has already been implemented for you in the driver function.
All you still need to do is compute the influence of a single light.

Phong shading combines three components:
1. Ambient (background) light - we ignore this;
2. Diffuse light - the light that is reflected equally in all directions (typical for rougher objects);
3. Specular light - the light that is reflected in a mirror-like way (typical for shiny objects).

***Important: all vectors have to be normalized, otherwise the dot-products will not work as expected.***

##### Diffuse lighting
For diffuse lighting, we need the following components:
- The normal at the intersection point ($\vec{N}$);
- The direction from the intersection point to the light source (the inverse direction of the light ray hitting the object) ($\vec{L}$);
- The color of the object ($C_M$);
- The color of the light source ($C_L$).

The intensity of the reflected light ($d$) is proportional to the cosine of the angle between the normal and the light direction:
$$
\begin{align*}
d &= max(0, \vec{N}\cdot\vec{L}) \\
C_D &= d * (C_M \circ C_L)
\end{align*}
$$

With $C_D$ being the diffuse color.
You can use `mul_elem` to get the Hadamard-product (or element-wise product, $\circ$) of two vectors

##### Specular lighting
For specular lighting, we need the following components:
- The normal at the intersection point ($\vec{N}$);
- The direction from the intersection point to the camera (the inverse direction of the ray hitting the object) ($\vec{V}$);
- The direction from the intersection point to the light source (the inverse direction of the light ray hitting the object) ($\vec{L}$);
- The color of the object ($C_M$);
- The color of the light source ($C_L$);
- The shininess of the object ($p$, commonly called the "Phong exponent").

The intensity of the reflected light ($s$) is proportional to the cosine between the reflection of the light direction and the camera direction:
$$
\begin{align*}
s &= max(0, \text{refl}(\vec{L}, \vec{N})\cdot\vec{V})^p \\
C_S &= s * C_L
\end{align*}
$$

With $C_S$ being the specular color.
You can use `reflect` to get the reflection of a vector (which will be a normalized vector).

##### Attenuation
The intensity of a point light at a point is inversely proportional to the square of the distance between the light and the point.
We need to take this into account when computing the diffuse and specular light.

Each point light has two additional parameters:
- `light.intensity`: the intensity of the light ($i$);
- `light.attenuation`: the attenuation factors.

The formula for attenuation ($\alpha$) requires three factors: $K_c$ (constant), $K_l$ (linear), and $K_q$ (quadratic); as well as the distance between the light and the point ($d$):
$$
\alpha = \frac{1}{K_c + K_l \cdot d + K_q \cdot d^2}
$$

We can now compute the final color of the point:
$$
C = i \cdot \alpha \cdot (C_D + C_S)
$$

For the distance between two points, you can use the `length(a, b)` function.

#### Reflections & transparency
A benefit to raytracing is that it can easily handle reflections and transparency, just by recursion (shooting additional rays).

As a GPU prefers to know the depth of recursion at compile-time, we will limit the recursion to a fixed depth (e.g. 3).
For this, we use the compile-time constant `bounces` and the template non-type argument `bounces_left` (if you are not familiar with these, don't worry, it's not very important for the challenge itself).

As you can see, the function `colorizer(...)` in the `kernel.cu` file has already been largely implemented for you.
What still needs to happen are the following smaller steps (in order of the TODO's in the file):
1. If there is an intersection, compute the intersection point and view direction.
2. If there is an intersection, the object is reflective, and there are bounces left, compute the reflection ray.
3. If there is an intersection, the object is transparent, and there are bounces left, compute the passthrough ray.

As you can see, the loop over the lights has already been implemented.
Similarly, the recursive calls and color combining (by using linear interpolation) have also already been implemented.

***Important:*** to avoid strange artefacts, you should add a small offset to the intersection point in the direction of the next bounce (e.g. `intersection + <direction> * epsilon`).

## Checking your output
The `./examples/` directory contains a few example scenes you can use to test your implementation.
Each of these scenes (`*.ini` files) also has an associated directory with the expected output (`*.png` files):
- `id.png`: the object ID for each hit (for the `id` mode);
- `dist.png`: the distance to the hit point (for the `dist` mode);
- `color.png`: the full rendering pass (for the `color` mode).