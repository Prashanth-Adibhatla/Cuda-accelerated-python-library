#include "../include/cuda_raytracer.cuh"
#include <cstdio>
#include <cfloat>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while (0)

// -------------------------------------------------------------------------
// Device Intersection & Shading Functions
// -------------------------------------------------------------------------

__device__ inline float intersect_plane_dev(const Vec3& O, const Vec3& D, const Vec3& P, const Vec3& N) {
    float denom = dot(D, N);
    if (fabsf(denom) < 1e-6f) {
        return 1e30f;
    }
    float d = dot(P - O, N) / denom;
    if (d < 0.0f) {
        return 1e30f;
    }
    return d;
}

__device__ inline float intersect_sphere_dev(const Vec3& O, const Vec3& D, const Vec3& S, float R) {
    float a = dot(D, D);
    Vec3 OS = O - S;
    float b = 2.0f * dot(D, OS);
    float c = dot(OS, OS) - R * R;
    float disc = b * b - 4.0f * a * c;

    if (disc > 0.0f) {
        float distSqrt = sqrtf(disc);
        float q = (b < 0.0f) ? (-b - distSqrt) * 0.5f : (-b + distSqrt) * 0.5f;
        float t0 = q / a;
        float t1 = c / q;
        if (t0 > t1) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        if (t1 >= 0.0f) {
            return (t0 < 0.0f) ? t1 : t0;
        }
    }
    return 1e30f;
}

// -------------------------------------------------------------------------
// 2D Ray Tracing Kernel (Replaces the nested Python loops)
// -------------------------------------------------------------------------

__global__ void raytrace_render_kernel(
    RenderParams params,
    const Sphere* spheres,
    int num_spheres,
    const Plane* planes,
    int num_planes,
    float* output_image
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column index (0 to w - 1)
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row index in screen space (0 to h - 1)

    if (i >= params.width || j >= params.height) {
        return;
    }

    // Screen coordinates corresponding to np.linspace(S[0], S[2], w) and np.linspace(S[1], S[3], h)
    float u = (params.width > 1) ? (float)i / (float)(params.width - 1) : 0.0f;
    float v = (params.height > 1) ? (float)j / (float)(params.height - 1) : 0.0f;

    float x = params.S[0] + u * (params.S[2] - params.S[0]);
    float y = params.S[1] + v * (params.S[3] - params.S[1]);

    Vec3 Q(x, y, 0.0f);
    Vec3 D = normalize(Q - params.camera_O);

    Vec3 rayO = params.camera_O;
    Vec3 rayD = D;
    Vec3 col(0.0f, 0.0f, 0.0f);
    float reflection = 1.0f;

    for (int depth = 0; depth < params.depth_max; ++depth) {
        float t_closest = 1e30f;
        int hit_type = -1; // 0 = sphere, 1 = plane
        int hit_idx = -1;

        // Find closest intersection among all spheres
        for (int s = 0; s < num_spheres; ++s) {
            float t_obj = intersect_sphere_dev(rayO, rayD, spheres[s].position, spheres[s].radius);
            if (t_obj < t_closest) {
                t_closest = t_obj;
                hit_type = OBJ_SPHERE;
                hit_idx = s;
            }
        }

        // Find closest intersection among all planes
        for (int p = 0; p < num_planes; ++p) {
            float t_obj = intersect_plane_dev(rayO, rayD, planes[p].position, planes[p].normal);
            if (t_obj < t_closest) {
                t_closest = t_obj;
                hit_type = OBJ_PLANE;
                hit_idx = p;
            }
        }

        // If no intersection was found, terminate ray
        if (hit_type == -1 || t_closest >= 1e29f) {
            break;
        }

        // Intersection point
        Vec3 M = rayO + rayD * t_closest;
        Vec3 N;
        Vec3 base_color;
        float diff_c = 1.0f;
        float spec_c = 1.0f;
        float obj_refl = 0.5f;

        if (hit_type == OBJ_SPHERE) {
            const Sphere& sp = spheres[hit_idx];
            N = normalize(M - sp.position);
            base_color = sp.color;
            diff_c = sp.diffuse_c;
            spec_c = sp.specular_c;
            obj_refl = sp.reflection;
        } else {
            const Plane& pl = planes[hit_idx];
            N = pl.normal;
            base_color = pl.get_color(M);
            diff_c = pl.diffuse_c;
            spec_c = pl.specular_c;
            obj_refl = pl.reflection;
        }

        Vec3 toL = normalize(params.light_L - M);
        Vec3 toO = normalize(params.camera_O - M);

        // Shadow check: test if ray from M + N*0.0001 towards light intersects ANY other object
        Vec3 shadow_O = M + N * 0.0001f;
        bool in_shadow = false;

        for (int s = 0; s < num_spheres; ++s) {
            if (hit_type == OBJ_SPHERE && s == hit_idx) {
                continue;
            }
            float t_sh = intersect_sphere_dev(shadow_O, toL, spheres[s].position, spheres[s].radius);
            if (t_sh < 1e29f) {
                in_shadow = true;
                break;
            }
        }

        if (!in_shadow) {
            for (int p = 0; p < num_planes; ++p) {
                if (hit_type == OBJ_PLANE && p == hit_idx) {
                    continue;
                }
                float t_sh = intersect_plane_dev(shadow_O, toL, planes[p].position, planes[p].normal);
                if (t_sh < 1e29f) {
                    in_shadow = true;
                    break;
                }
            }
        }

        // In Ray-Tracer.py: if in shadow, trace_ray returns None, breaking the reflection loop
        if (in_shadow) {
            break;
        }

        // Shading: Ambient + Lambert (diffuse) + Blinn-Phong (specular)
        Vec3 col_ray(params.ambient, params.ambient, params.ambient);

        float dot_N_toL = fmaxf(dot(N, toL), 0.0f);
        col_ray += diff_c * dot_N_toL * base_color;

        Vec3 half_vec = normalize(toL + toO);
        float dot_N_half = fmaxf(dot(N, half_vec), 0.0f);
        float spec_term = powf(dot_N_half, params.specular_k);
        col_ray += spec_c * spec_term * params.light_color;

        col += reflection * col_ray;

        // Reflection ray
        rayO = M + N * 0.0001f;
        rayD = normalize(rayD - 2.0f * dot(rayD, N) * N);
        reflection *= obj_refl;
    }

    // Write to output image buffer
    // Image coordinate convention matching Python: img[h - j - 1, i, :]
    int out_row = params.height - j - 1;
    int out_col = i;
    int pixel_idx = (out_row * params.width + out_col) * 3;

    output_image[pixel_idx + 0] = fminf(fmaxf(col.x, 0.0f), 1.0f);
    output_image[pixel_idx + 1] = fminf(fmaxf(col.y, 0.0f), 1.0f);
    output_image[pixel_idx + 2] = fminf(fmaxf(col.z, 0.0f), 1.0f);
}

// -------------------------------------------------------------------------
// Host Scene Rendering Function
// -------------------------------------------------------------------------

void render_scene_cuda(
    const RenderParams& params,
    const std::vector<Sphere>& spheres,
    const std::vector<Plane>& planes,
    float* output_image
) {
    int num_spheres = static_cast<int>(spheres.size());
    int num_planes = static_cast<int>(planes.size());
    size_t image_size = params.width * params.height * 3 * sizeof(float);

    // Device memory allocation
    Sphere* d_spheres = nullptr;
    Plane* d_planes = nullptr;
    float* d_output = nullptr;

    if (num_spheres > 0) {
        CUDA_CHECK(cudaMalloc(&d_spheres, num_spheres * sizeof(Sphere)));
        CUDA_CHECK(cudaMemcpy(d_spheres, spheres.data(), num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice));
    }

    if (num_planes > 0) {
        CUDA_CHECK(cudaMalloc(&d_planes, num_planes * sizeof(Plane)));
        CUDA_CHECK(cudaMemcpy(d_planes, planes.data(), num_planes * sizeof(Plane), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&d_output, image_size));
    CUDA_CHECK(cudaMemset(d_output, 0, image_size));

    // Launch 2D grid of threads
    dim3 block(16, 16);
    dim3 grid(
        (params.width + block.x - 1) / block.x,
        (params.height + block.y - 1) / block.y
    );

    raytrace_render_kernel<<<grid, block>>>(
        params,
        d_spheres,
        num_spheres,
        d_planes,
        num_planes,
        d_output
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(output_image, d_output, image_size, cudaMemcpyDeviceToHost));

    // Cleanup
    if (d_spheres) cudaFree(d_spheres);
    if (d_planes) cudaFree(d_planes);
    if (d_output) cudaFree(d_output);
}

// -------------------------------------------------------------------------
// Batch CUDA Kernels for Individual Methods
// -------------------------------------------------------------------------

__global__ void batch_normalize_kernel(const float* in_vecs, float* out_vecs, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        Vec3 v(in_vecs[idx * 3 + 0], in_vecs[idx * 3 + 1], in_vecs[idx * 3 + 2]);
        Vec3 res = normalize(v);
        out_vecs[idx * 3 + 0] = res.x;
        out_vecs[idx * 3 + 1] = res.y;
        out_vecs[idx * 3 + 2] = res.z;
    }
}

void parallel_normalize_cuda(const float* in_vecs, float* out_vecs, int count) {
    size_t size = count * 3 * sizeof(float);
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));
    CUDA_CHECK(cudaMemcpy(d_in, in_vecs, size, cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (count + block - 1) / block;
    batch_normalize_kernel<<<grid, block>>>(d_in, d_out, count);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out_vecs, d_out, size, cudaMemcpyDeviceToHost));
    cudaFree(d_in);
    cudaFree(d_out);
}

__global__ void batch_intersect_spheres_kernel(
    const float* origins, const float* dirs,
    const Sphere* spheres, int num_spheres,
    float* out_dist, int num_rays
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < num_rays) {
        Vec3 O(origins[r * 3 + 0], origins[r * 3 + 1], origins[r * 3 + 2]);
        Vec3 D(dirs[r * 3 + 0], dirs[r * 3 + 1], dirs[r * 3 + 2]);
        float min_d = 1e30f;
        for (int s = 0; s < num_spheres; ++s) {
            float d = intersect_sphere_dev(O, D, spheres[s].position, spheres[s].radius);
            if (d < min_d) min_d = d;
        }
        out_dist[r] = (min_d < 1e29f) ? min_d : INFINITY;
    }
}

void parallel_intersect_spheres_cuda(
    const float* ray_origins, const float* ray_dirs,
    const Sphere* spheres, int num_spheres,
    float* out_distances, int num_rays
) {
    size_t vec_size = num_rays * 3 * sizeof(float);
    size_t dist_size = num_rays * sizeof(float);

    float *d_orig, *d_dirs, *d_out;
    Sphere* d_spheres;

    CUDA_CHECK(cudaMalloc(&d_orig, vec_size));
    CUDA_CHECK(cudaMalloc(&d_dirs, vec_size));
    CUDA_CHECK(cudaMalloc(&d_out, dist_size));
    CUDA_CHECK(cudaMalloc(&d_spheres, num_spheres * sizeof(Sphere)));

    CUDA_CHECK(cudaMemcpy(d_orig, ray_origins, vec_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dirs, ray_dirs, vec_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spheres, spheres, num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (num_rays + block - 1) / block;
    batch_intersect_spheres_kernel<<<grid, block>>>(d_orig, d_dirs, d_spheres, num_spheres, d_out, num_rays);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out_distances, d_out, dist_size, cudaMemcpyDeviceToHost));

    cudaFree(d_orig);
    cudaFree(d_dirs);
    cudaFree(d_spheres);
    cudaFree(d_out);
}

__global__ void batch_intersect_planes_kernel(
    const float* origins, const float* dirs,
    const Plane* planes, int num_planes,
    float* out_dist, int num_rays
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < num_rays) {
        Vec3 O(origins[r * 3 + 0], origins[r * 3 + 1], origins[r * 3 + 2]);
        Vec3 D(dirs[r * 3 + 0], dirs[r * 3 + 1], dirs[r * 3 + 2]);
        float min_d = 1e30f;
        for (int p = 0; p < num_planes; ++p) {
            float d = intersect_plane_dev(O, D, planes[p].position, planes[p].normal);
            if (d < min_d) min_d = d;
        }
        out_dist[r] = (min_d < 1e29f) ? min_d : INFINITY;
    }
}

void parallel_intersect_planes_cuda(
    const float* ray_origins, const float* ray_dirs,
    const Plane* planes, int num_planes,
    float* out_distances, int num_rays
) {
    size_t vec_size = num_rays * 3 * sizeof(float);
    size_t dist_size = num_rays * sizeof(float);

    float *d_orig, *d_dirs, *d_out;
    Plane* d_planes;

    CUDA_CHECK(cudaMalloc(&d_orig, vec_size));
    CUDA_CHECK(cudaMalloc(&d_dirs, vec_size));
    CUDA_CHECK(cudaMalloc(&d_out, dist_size));
    CUDA_CHECK(cudaMalloc(&d_planes, num_planes * sizeof(Plane)));

    CUDA_CHECK(cudaMemcpy(d_orig, ray_origins, vec_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dirs, ray_dirs, vec_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_planes, planes, num_planes * sizeof(Plane), cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (num_rays + block - 1) / block;
    batch_intersect_planes_kernel<<<grid, block>>>(d_orig, d_dirs, d_planes, num_planes, d_out, num_rays);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(out_distances, d_out, dist_size, cudaMemcpyDeviceToHost));

    cudaFree(d_orig);
    cudaFree(d_dirs);
    cudaFree(d_planes);
    cudaFree(d_out);
}

