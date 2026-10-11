#pragma once

#include "vec3.cuh"

enum ObjectType {
    OBJ_SPHERE = 0,
    OBJ_PLANE = 1
};

struct Sphere {
    Vec3 position;
    float radius;
    Vec3 color;
    float reflection;
    float diffuse_c;
    float specular_c;

    __host__ __device__ Sphere() 
        : position(), radius(1.0f), color(1.0f, 1.0f, 1.0f), 
          reflection(0.5f), diffuse_c(1.0f), specular_c(1.0f) {}

    __host__ __device__ Sphere(Vec3 pos, float r, Vec3 col, float refl = 0.5f, float diff = 1.0f, float spec = 1.0f)
        : position(pos), radius(r), color(col), 
          reflection(refl), diffuse_c(diff), specular_c(spec) {}
};

struct Plane {
    Vec3 position;
    Vec3 normal;
    Vec3 color0;
    Vec3 color1;
    float reflection;
    float diffuse_c;
    float specular_c;

    __host__ __device__ Plane()
        : position(0.0f, -0.5f, 0.0f), normal(0.0f, 1.0f, 0.0f),
          color0(1.0f, 1.0f, 1.0f), color1(0.0f, 0.0f, 0.0f),
          reflection(0.25f), diffuse_c(0.75f), specular_c(0.5f) {}

    __host__ __device__ Plane(Vec3 pos, Vec3 n, Vec3 c0, Vec3 c1, float refl = 0.25f, float diff = 0.75f, float spec = 0.5f)
        : position(pos), normal(n), color0(c0), color1(c1),
          reflection(refl), diffuse_c(diff), specular_c(spec) {}

    __host__ __device__ Vec3 get_color(const Vec3& M) const {
        // Python: (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2)
        // In Python: int(x) truncates toward 0, and % 2 in Python is Python modulo (non-negative)
        // Let's implement exact matching logic
        int m0 = static_cast<int>(M.x * 2.0f);
        int m2 = static_cast<int>(M.z * 2.0f);
        int rem0 = ((m0 % 2) + 2) % 2;
        int rem2 = ((m2 % 2) + 2) % 2;
        return (rem0 == rem2) ? color0 : color1;
    }
};

struct RenderParams {
    int width;
    int height;
    int depth_max;
    float ambient;
    float specular_k;
    Vec3 camera_O;
    Vec3 light_L;
    Vec3 light_color;
    float S[4]; // Screen bounds: x0, y0, x1, y1
};

