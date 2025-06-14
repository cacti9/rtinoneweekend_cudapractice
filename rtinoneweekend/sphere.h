#pragma once

#include "hittable.h"
#include "vec3.h"

class sphere : public hittable {
public:
  __device__ sphere(const point3& center, float radius) : center(center), radius(fmaxf(0, radius)) {}

  __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
    vec3 oc = center - r.origin();
    auto a = r.direction().length_squared();
    auto h = dot(r.direction(), oc);
    auto c = oc.length_squared() - radius * radius;

    auto discriminant = h * h - a * c;
    if (discriminant < 0.f)
      return false;

    auto sqrtd = sqrtf(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (h - sqrtd) / a;
    if (!ray_t.surrounds(root)) {
      root = (h + sqrtd) / a;
      if (!ray_t.surrounds(root)) 
        return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);

    return true;
  }

private:
  point3 center;
  float radius;
};
