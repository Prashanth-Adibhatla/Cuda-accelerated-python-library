import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

def find_cuda():
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and os.path.isdir(cuda_path):
        return cuda_path
    candidates = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None

def find_vcvars64():
    candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None

class CUDABuildExt(build_ext):
    def build_extension(self, ext):
        cuda_dir = find_cuda()
        if not cuda_dir:
            raise RuntimeError("CUDA Toolkit not found! Please ensure CUDA Toolkit is installed.")
        
        nvcc = os.path.join(cuda_dir, "bin", "nvcc.exe")
        if not os.path.isfile(nvcc):
            nvcc = "nvcc"

        vcvars64 = find_vcvars64()
        py_include = os.path.dirname(sys.executable)
        python_include = os.path.join(py_include, "include")
        if not os.path.exists(python_include):
            import sysconfig
            python_include = sysconfig.get_path('include')
        
        python_lib_dir = os.path.join(py_include, "libs")
        if not os.path.exists(python_lib_dir):
            import sysconfig
            python_lib_dir = os.path.join(sysconfig.get_config_var('installed_base'), 'libs')

        pybind_include = pybind11.get_include()
        cuda_include = os.path.join(cuda_dir, "include")
        cuda_lib_dir = os.path.join(cuda_dir, "lib", "x64")
        project_include = os.path.abspath("include")

        out_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Ensure correct Windows extension: .pyd
        if not out_path.endswith(".pyd"):
            out_path = os.path.splitext(out_path)[0] + ".pyd"

        sources = [os.path.abspath(s) for s in ext.sources]

        # Construct NVCC build command
        cmd = [
            f'"{nvcc}"',
            "-shared",
            "-O3",
            "--std=c++17",
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_89,code=sm_89",
            "-Xcompiler", "/MD,/EHsc,/O2,/W3,/Zc:preprocessor",
            f'-I"{project_include}"',
            f'-I"{cuda_include}"',
            f'-I"{pybind_include}"',
            f'-I"{python_include}"',
            f'-L"{cuda_lib_dir}"',
            f'-L"{python_lib_dir}"',
            "-lcudart",
            f'-o "{out_path}"',
        ]
        cmd.extend([f'"{s}"' for s in sources])

        full_cmd_str = " ".join(cmd)
        if vcvars64:
            build_script = f'call "{vcvars64}" && {full_cmd_str}'
        else:
            build_script = full_cmd_str

        print("Executing build command:\n", build_script)
        ret = subprocess.call(build_script, shell=True)
        if ret != 0:
            raise RuntimeError(f"Compilation failed with exit code {ret}")

setup(
    name="raytracer_cuda",
    version="1.0.0",
    description="CUDA-accelerated numerical kernels and ray-tracing benchmarks using Pybind11",
    ext_modules=[
        Extension(
            "raytracer_cuda",
            sources=["src/cuda_raytracer.cu", "src/numeric_cuda.cu", "src/bindings.cpp"],
        )
    ],
    cmdclass={"build_ext": CUDABuildExt},
)

