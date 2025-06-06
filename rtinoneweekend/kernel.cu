#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define __STDC_LIB_EXT1__
#include "stb_image_write.h"

#include <iostream>
#include <ctime>
#include <vector>

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

__global__ void render(vec3 *fb, int width, int height) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i >= width || j >= height) return;
  int pixel_index = (j * width + i);
  fb[pixel_index] = vec3(float(i) / width, float(j) / height, 0.2f);
}

int main() {
  int nx = 1200;
  int ny = 600;
  int tx = 8;
  int ty = 8;

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
  render << <blocks, threads >> > (fb, nx, ny);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cout << "took " << timer_seconds << " seconds.\n";

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      int pixel_index = (j * nx + i);
      uint8_t ir = uint8_t(255.999 * fb[pixel_index].r());
      uint8_t ig = uint8_t(255.999 * fb[pixel_index].g());
      uint8_t ib = uint8_t(255.999 * fb[pixel_index].b());

      imageBuffer.push_back(ir);
      imageBuffer.push_back(ig);
      imageBuffer.push_back(ib);
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
