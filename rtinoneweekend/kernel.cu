#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define __STDC_LIB_EXT1__
#include "stb_image_write.h"

#include <iostream>
#include <ctime>
#include <vector>

#include "color.h"
#include "ray.h"
#include "vec3.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
      file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

__device__ float hit_sphere(const point3& center, float radius, const ray& r) {
  vec3 oc = r.origin() - center;
  auto a = dot(r.direction(), r.direction());
  auto b = 2.0f * dot(r.direction(), oc);
  auto c = dot(oc, oc) - radius * radius;
  auto discriminant = b * b - 4.f * a * c;

  if (discriminant < 0.f) {
    return -1.f;
  }
  else {
    return (-b - sqrtf(discriminant)) / (2.f * a);
  }
}

__device__ color ray_color(const ray& r) {
  auto t = hit_sphere(point3(0, 0, -1), 0.5f, r);
  if (t > 0.0f) {
    vec3 N = unit_vector(r.at(t) - vec3(0, 0, -1));
    return 0.5f * color(N.x() + 1, N.y() + 1, N.z() + 1);
  }

  vec3 unit_direction = unit_vector(r.direction());
  auto a = 0.5f * (unit_direction.y() + 1.0f);
  return (1.0f - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}

__global__ void render(vec3 *fb, int width, int height, vec3 to_pixel00, vec3 viewport_u, vec3 viewport_v, vec3 camera_center) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i >= width || j >= height) return;
  int pixel_index = (j * width + i);

  //instead of pixel_delta, divide everytime for better accuracy
  auto ray_direction = to_pixel00 + (i * viewport_u)/width + (j * viewport_v)/height; 
  ray r(camera_center, ray_direction);
  fb[pixel_index] = ray_color(r);
}

int main() {
  // Image
  auto aspect_ratio = 16.0 / 9.0;
  int nx = 1200;
  int tx = 8;
  int ty = 8;

  // Calculate the image height, and ensure that it's at least 1.
  int ny = int(nx / aspect_ratio);
  ny = (ny < 1) ? 1 : ny;

  // Camera

  auto focal_length = 1.0;
  auto viewport_height = 2.0;
  auto viewport_width = viewport_height * (double(nx) / ny);
  auto camera_center = point3(0, 0, 0);

  // Calculate the vectors across the horizontal and down the vertical viewport edges.
  auto viewport_u = vec3(viewport_width, 0, 0);
  auto viewport_v = vec3(0, -viewport_height, 0);

  // Calculate the horizontal and vertical delta vectors from pixel to pixel.
  auto pixel_delta_u = viewport_u / nx;
  auto pixel_delta_v = viewport_v / ny;

  // Calculate the location of the upper left pixel.
  auto viewport_upper_left = camera_center
    - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
  auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
  auto to_pixel00 = pixel00_loc - camera_center;

  // Render
  std::cerr << "Rendering a " << nx << "x" << ny << " image ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  int num_pixels = nx * ny;
  std::vector<uint8_t> imageBuffer; imageBuffer.reserve(num_pixels * 3);
  size_t fb_size = num_pixels * sizeof(vec3);

  // allocate FB
  vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  clock_t start, stop;
  start = clock();
  // Render our buffer
  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);
  render << <blocks, threads >> > (fb, nx, ny, to_pixel00, viewport_u, viewport_v, camera_center);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cout << "took " << timer_seconds << " seconds.\n";

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

  checkCudaErrors(cudaFree(fb));
}
