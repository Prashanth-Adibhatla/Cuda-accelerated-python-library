@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo  Building raytracer_cuda Python Extension
echo  Includes CUDA numerical kernels and ray-tracing benchmarks
echo ===================================================

REM 1. Locate CUDA Toolkit
if defined CUDA_PATH (
    set "CUDA_DIR=%CUDA_PATH%"
) else (
    set "CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
)

if not exist "!CUDA_DIR!\bin\nvcc.exe" (
    echo [ERROR] nvcc.exe not found in !CUDA_DIR!\bin
    exit /b 1
)

echo [INFO] Found CUDA at: !CUDA_DIR!

REM 2. Locate Visual Studio vcvars64.bat
set "VCVARS="
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)

if defined VCVARS (
    echo [INFO] Initializing MSVC environment via: !VCVARS!
    call "!VCVARS!"
)

REM 3. Run setup.py to build in-place
echo [INFO] Running: python setup.py build_ext --inplace
python setup.py build_ext --inplace

if %ERRORLEVEL% EQU 0 (
    echo ===================================================
    echo  Build Successful! Module raytracer_cuda is ready.
    echo ===================================================
) else (
    echo ===================================================
    echo  Build Failed with error code %ERRORLEVEL%.
    echo ===================================================
    exit /b %ERRORLEVEL%
)

