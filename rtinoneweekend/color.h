#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "interval.h"
#include "vec3.h"

#include <vector>

using color = vec3;

void write_color(std::vector<uint8_t>& imageBuffer, const color& pixel_color) {
  auto r = pixel_color.x();
  auto g = pixel_color.y();
  auto b = pixel_color.z();

  static const interval intensity(0.000, 0.999);
  uint8_t rbyte = uint8_t(256 * intensity.clamp(r));
  uint8_t gbyte = uint8_t(256 * intensity.clamp(g));
  uint8_t bbyte = uint8_t(256 * intensity.clamp(b));

  imageBuffer.push_back(rbyte);
  imageBuffer.push_back(gbyte);
  imageBuffer.push_back(bbyte);
}
