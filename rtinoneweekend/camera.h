#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include "hittable_list.h"

namespace camera_h{
  //it is inefficient to call camera.ray_color, so just global
  __device__ color ray_color(const ray& r, hittable_list **world) {
    hit_record rec;
    if ((*world)->hit(r, { 0.f, infinity }, rec)) {
      return 0.5f * (rec.normal + color(1.f, 1.f, 1.f));
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
  }

  // rand seed for each pixel
  __global__ void render_init(int width, int height, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;
    int pixel_index = j * width + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
  }

  //a kernel(__global__) cannot be a member function
  __global__ void renderKernel(vec3 *fb, int width, int height, int nSamples,
                               vec3 to_pixel00, vec3 viewport_u, vec3 viewport_v, vec3 camera_center,
                               hittable_list **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;
    int pixel_index = (j * width + i);
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < nSamples; ++s) {
      //instead of pixel_delta, divide everytime for better accuracy
      auto ray_direction = to_pixel00 + ((i + curand_uniform(&local_rand_state)) * viewport_u) / width + ((j + curand_uniform(&local_rand_state)) * viewport_v) / height;
      ray r(camera_center, ray_direction);
      col += ray_color(r, world);
    }
    fb[pixel_index] = col/nSamples;
  }
}

class camera {
public:
  float  aspect_ratio=1.f;
  int    nx = 10;
  int    samples_per_pixel = 10;
  void render(hittable_list **d_world) {
    initialize();

    // Render
    int tx = 8; 
    int ty = 8;
    std::cout << "Rendering a " << nx << "x" << ny << " image ";
    std::cout << "in " << tx << "x" << ty << " blocks.\n";


    // allocate FB
    vec3 *fb;
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
    // allocate random state
    curandState *d_rand_state; //random seed per pixel
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    camera_h::render_init <<<blocks, threads >>> (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    camera_h::renderKernel <<<blocks, threads >>> (fb, nx, ny, samples_per_pixel,to_pixel00, viewport_u, viewport_v, center, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cout << "took " << timer_seconds << " seconds.\n";

    // Write to bmp
    std::vector<uint8_t> imageBuffer; imageBuffer.reserve(num_pixels * 3);
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        int pixel_index = (j * nx + i);
        write_color(imageBuffer, fb[pixel_index]);
      }
    }

    if (stbi_write_bmp("rendered.bmp", nx, ny, 3, imageBuffer.data()) == 1) {
      std::cout << "image out success\n";
    }
    else {
      std::cout << "image out fail\n";
    }

    // cleanup
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));
  }
private:
  int    ny;
  point3 center;
  point3 pixel00_loc;
  vec3   viewport_u;
  vec3   viewport_v;
  vec3   to_pixel00;
  void initialize() {
    // Calculate the image height, and ensure that it's at least 1.
    ny = int(nx / aspect_ratio);
    ny = (ny < 1) ? 1 : ny;

    center = point3(0, 0, 0);

    // Determine viewport dimensions
    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(nx) / ny);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    viewport_u = vec3(viewport_width, 0, 0);
    viewport_v = vec3(0, -viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    auto pixel_delta_u = viewport_u / nx;
    auto pixel_delta_v = viewport_v / ny;

    // Calculate the location of the upper left pixel.
    auto viewport_upper_left = center
      - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
    pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    to_pixel00 = pixel00_loc - center;
  }
};
