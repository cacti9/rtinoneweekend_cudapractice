#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define __STDC_LIB_EXT1__
#include "stb_image_write.h"

#include <iostream>
#include <ctime>
#include <vector>

#include "rtweekend.h"
#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include <new>


__global__ void create_world(hittable **d_list, hittable_list **d_world) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(d_list) = new sphere(vec3(0, 0, -1), 0.5);
    *(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100);
    *d_world = new hittable_list(d_list, 2);
  }
}

__global__ void free_world(hittable **d_list, hittable_list **d_world) {
  delete *(d_list);
  delete *(d_list + 1);
  delete *d_world;
}

int main() {
  // make our world of hitables
  hittable **d_list = nullptr;
  hittable_list **d_world = nullptr;
  checkCudaErrors(cudaMalloc((void **)&d_list, 2 * sizeof(hittable *)));
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable_list *)));
  create_world << <1, 1 >> > (d_list, d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  camera cam;
  cam.aspect_ratio = 16.f / 9.f;
  cam.nx = 400;
  cam.render(d_world);

  // clean up
  checkCudaErrors(cudaDeviceSynchronize());
  free_world<<<1,1>>>(d_list,d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));

  // useful for cuda-memcheck --leak-check full
  cudaDeviceReset();
}
