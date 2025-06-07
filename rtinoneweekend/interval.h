#pragma once

class interval {
public:
  float min, max;

  __device__ interval() : min(+infinity), max(-infinity) {} // Default interval is empty

  __host__ __device__ interval(float min, float max) : min(min), max(max) {}

  __device__ float size() const {
    return max - min;
  }

  __device__ bool contains(float x) const {
    return min <= x && x <= max;
  }

  __device__ bool surrounds(float x) const {
    return min < x && x < max;
  }

  __host__ float clamp(float x) const {
    if (x < min) return min;
    if (x > max) return max;
    return x;
  }

  static const interval empty, universe;
};

const interval interval::empty = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity);
