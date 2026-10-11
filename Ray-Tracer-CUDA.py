import time
import numpy as np
import matplotlib.pyplot as plt

try:
    import raytracer_cuda
    CUDA_AVAILABLE = True
except ImportError:
    print("[WARNING] 'raytracer_cuda' module not found.")
    print("Please build the library first by running: python setup.py build_ext --inplace (or run build.bat)")
    CUDA_AVAILABLE = False


def run_cuda_raytracer():
    # -----------------------------
    # Scene and Camera Setup
    # -----------------------------
    w = 2160
    h = 1440
    r = float(w) / h
    # Screen coordinates: x0, y0, x1, y1
    S = (-1.0, -1.0 / r + 0.25, 1.0, 1.0 / r + 0.25)

    # Camera & Light
    O = [0.0, 0.35, -1.0]
    L = [-4.0, 5.0, -10.0]
    color_light = [1.0, 1.0, 1.0]

    # Material & Lighting Parameters
    ambient = 0.05
    specular_k = 50.0
    depth_max = 5

    # Spheres definition
    spheres = [
        {
            "position": [0.75, 0.1, 1.0],
            "radius": 0.6,
            "color": [0.0, 0.0, 1.0],
            "reflection": 0.5,
            "diffuse_c": 1.0,
            "specular_c": 1.0,
        },
        {
            "position": [-0.75, 0.1, 2.25],
            "radius": 0.6,
            "color": [0.5, 0.223, 0.5],
            "reflection": 0.5,
            "diffuse_c": 1.0,
            "specular_c": 1.0,
        },
        {
            "position": [-2.75, 0.1, 3.5],
            "radius": 0.6,
            "color": [1.0, 0.572, 0.184],
            "reflection": 0.5,
            "diffuse_c": 1.0,
            "specular_c": 1.0,
        },
    ]

    # Plane definition (Checkerboard)
    planes = [
        {
            "position": [0.0, -0.5, 0.0],
            "normal": [0.0, 1.0, 0.0],
            "color0": [1.0, 1.0, 1.0],
            "color1": [0.0, 0.0, 0.0],
            "reflection": 0.25,
            "diffuse_c": 0.75,
            "specular_c": 0.5,
        }
    ]

    if not CUDA_AVAILABLE:
        print("[ERROR] Cannot render: raytracer_cuda library is not yet compiled.")
        return

    print("=" * 60)
    print(f"Rendering {w}x{h} ({w * h:,} pixels) using CUDA parallel ray tracing...")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. Full Scene Rendering (Replaces the Python nested for loop)
    # -------------------------------------------------------------
    # Warm-up run for GPU initialization
    _ = raytracer_cuda.render_scene(
        w,
        h,
        depth_max,
        ambient,
        specular_k,
        O,
        L,
        color_light,
        list(S),
        spheres,
        planes,
    )

    # Timed benchmark run
    start_time = time.perf_counter()

    img = raytracer_cuda.render_scene(
        w,
        h,
        depth_max,
        ambient,
        specular_k,
        O,
        L,
        color_light,
        list(S),
        spheres,
        planes,
    )

    render_time = time.perf_counter() - start_time
    total_pixels = w * h
    throughput = (total_pixels / render_time) / 1e6

    print(f"[RESULT] Render completed successfully in {render_time*1000:.2f} ms ({render_time:.6f} s)")
    print(f"[PERFORMANCE] Throughput: {total_pixels / render_time:,.0f} pixels/sec ({throughput:.2f} Mpixels/sec)")

    # -------------------------------------------------------------
    # 2. Demonstration of Individual Parallel CUDA Functions
    # -------------------------------------------------------------
    print("\n--- Testing Individual Parallel CUDA Functions ---")
    
    # (a) Parallel vector normalize
    sample_vecs = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [1.0, 1.0, 1.0]
    ], dtype=np.float32)
    norm_res = raytracer_cuda.parallel_normalize(sample_vecs)
    print(f"parallel_normalize output for 3 vectors:\n{norm_res}")

    # (b) Parallel sphere intersection test
    origins = np.array([[0.0, 0.1, 0.0], [0.75, 0.1, -1.0]], dtype=np.float32)
    dirs = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    dists = raytracer_cuda.parallel_intersect_spheres(origins, dirs, spheres)
    print(f"parallel_intersect_spheres distances: {dists}")

    # -------------------------------------------------------------
    # 3. Save and Display the Rendered Image
    # -------------------------------------------------------------
    output_filename = "fig_cuda.png"
    plt.imsave(output_filename, img)
    print(f"\n[OUTPUT] Saved rendered image to '{output_filename}'")

    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    plt.title(f"CUDA Parallel Ray Tracer - {w}x{h} in {render_time*1000:.2f} ms")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_cuda_raytracer()

