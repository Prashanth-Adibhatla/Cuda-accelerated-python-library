#pragma once

#include <cuda_runtime.h>
#include <cmath>

struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ Vec3 operator+(const Vec3& b) const {
        return Vec3(x + b.x, y + b.y, z + b.z);
    }

    __host__ __device__ Vec3 operator-(const Vec3& b) const {
        return Vec3(x - b.x, y - b.y, z - b.z);
    }

    __host__ __device__ Vec3 operator*(float s) const {
        return Vec3(x * s, y * s, z * s);
    }

    __host__ __device__ friend Vec3 operator*(float s, const Vec3& v) {
        return Vec3(v.x * s, v.y * s, v.z * s);
    }

    __host__ __device__ Vec3 operator*(const Vec3& b) const {
        return Vec3(x * b.x, y * b.y, z * b.z);
    }

    __host__ __device__ Vec3 operator/(float s) const {
        float inv = 1.0f / s;
        return Vec3(x * inv, y * inv, z * inv);
    }

    __host__ __device__ Vec3& operator+=(const Vec3& b) {
        x += b.x; y += b.y; z += b.z;
        return *this;
    }

    __host__ __device__ Vec3& operator*=(float s) {
        x *= s; y *= s; z *= s;
        return *this;
    }

    __host__ __device__ Vec3& operator*=(const Vec3& b) {
        x *= b.x; y *= b.y; z *= b.z;
        return *this;
    }

    __host__ __device__ float dot(const Vec3& b) const {
        return x * b.x + y * b.y + z * b.z;
    }

    __host__ __device__ float norm_sq() const {
        return x * x + y * y + z * z;
    }

    __host__ __device__ float norm() const {
        return sqrtf(norm_sq());
    }

    __host__ __device__ Vec3 normalized() const {
        float n = norm();
        if (n > 1e-8f) {
            float inv = 1.0f / n;
            return Vec3(x * inv, y * inv, z * inv);
        }
        return Vec3(0.0f, 0.0f, 0.0f);
    }

    __host__ __device__ Vec3 cross(const Vec3& b) const {
        return Vec3(
            y * b.z - z * b.y,
            z * b.x - x * b.z,
            x * b.y - y * b.x
        );
    }
};

__host__ __device__ inline float dot(const Vec3& a, const Vec3& b) {
    return a.dot(b);
}

__host__ __device__ inline Vec3 normalize(const Vec3& v) {
    return v.normalized();
}

