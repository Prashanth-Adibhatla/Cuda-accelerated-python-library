#pragma once

#include "vec3.cuh"
#include "ray.cuh"
#include "scene_objects.cuh"
#include <vector>

// Full scene render in parallel on GPU (replacing the nested loop)
void render_scene_cuda(
    const RenderParams& params,
    const std::vector<Sphere>& spheres,
    const std::vector<Plane>& planes,
    float* output_image // Pointer to (height * width * 3) float buffer
);

// Individual parallel CUDA functions implementing the ray tracer methods for N elements:
void parallel_normalize_cuda(const float* in_vecs, float* out_vecs, int count);
void parallel_intersect_spheres_cuda(
    const float* ray_origins, const float* ray_dirs,
    const Sphere* spheres, int num_spheres,
    float* out_distances, int num_rays
);
void parallel_intersect_planes_cuda(
    const float* ray_origins, const float* ray_dirs,
    const Plane* planes, int num_planes,
    float* out_distances, int num_rays
);

