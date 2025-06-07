#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>


// Constants

constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr float pi = 3.1415926535897932385;

// Common Headers

#include "color.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"

// Utility Functions

__device__ inline float degrees_to_radians(float degrees) {
  return degrees * pi / 180.0;
}

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
