#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "vec3.h"

#include <vector>

using color = vec3;

void write_color(std::vector<uint8_t>& imageBuffer, const color& pixel_color) {
  auto r = pixel_color.x();
  auto g = pixel_color.y();
  auto b = pixel_color.z();

  uint8_t ir = uint8_t(255.999 * r);
  uint8_t ig = uint8_t(255.999 * g);
  uint8_t ib = uint8_t(255.999 * b);

  imageBuffer.push_back(ir);
  imageBuffer.push_back(ig);
  imageBuffer.push_back(ib);
}
