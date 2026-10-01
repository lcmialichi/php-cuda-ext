<p align="center">
  <a href="https://github.com/lcmialichi/php-cuda-ext">
    <img src="https://repository-images.githubusercontent.com/1091968129/520375bf-6506-4732-9834-9c5b51d9888b"
         alt="php-cuda-ext banner"
         width="480">
  </a>
</p>

<h1 align="center">php-cuda-ext</h1>

<p align="center">
  Native PHP extension for GPU computing using NVIDIA CUDA
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PHP-8.0+-purple?logo=php">
  <img src="https://img.shields.io/badge/CUDA-11.0%2B-76B900?logo=nvidia">
  <img src="https://img.shields.io/badge/Platform-Linux-red">
  <img src="https://img.shields.io/badge/License-MIT-blue">
</p>

---

## Project Status

> **Under active development**

- APIs are unstable and may change
- Not recommended for production environments
---

## Overview

`php-cuda-ext` is a native PHP extension that enables **GPU-accelerated numerical computing, machine learning, and data science workloads directly from PHP** using NVIDIA CUDA.

The extension gives PHP developers **first-class access to GPU computing**, allowing applications written in PHP to operate on large-scale tensors, execute parallel numerical algorithms, and scale computational workloads beyond CPU limitations.

With `php-cuda-ext`, PHP is no longer restricted to orchestration or I/O-bound tasks — it becomes a viable environment for:

- Tensor-based computation
- Data science pipelines
- Machine learning primitives
- High-throughput numerical processing
- GPU-accelerated experimentation and research

All computations are executed natively on the GPU, without relying on external runtimes or language bridges.

---

## Design Goals

- No Python dependency
- No bindings to TensorFlow, PyTorch, or similar frameworks
- Native PHP syntax and semantics
- Explicit control over GPU execution
- Emphasis on performance and transparency

Rather than prescribing a fixed machine learning abstraction,
`php-cuda-ext` focuses on providing the fundamental building blocks
required to implement ML and data science systems directly in PHP.

This approach favors flexibility, performance, and transparency over
opinionated high-level APIs.

## Architecture and API direction

`Cuda\CudaArray` currently means a tensor whose operations require CUDA. The
internal `tensor_t` describes shape, dtype, storage and views; `tensor_factory`
creates tensors from PHP values or host buffers, while `tensor_transfer` owns
device-to-host transfers and PHP array conversion. PHP class methods delegate
to these internal components and the CUDA operation layer.

`CudaArray` remains the public name while CUDA is the only execution backend.
A backend-neutral `Tensor` should only become the primary API when CPU storage
and operations actually exist; a compatibility alias or migration path will be
needed for current users. Fused execution graphs will need a separate execution
layer between tensor operations and CUDA kernels. Neither CPU fallback nor
graph execution is implemented yet.

CUDA launch conventions: one-thread-per-element kernels use `cuda_grid_1d`
from `src/cuda/launch_config.cuh` and skip empty inputs. Matrix multiplication
uses `cuda_grid_2d` in `src/cuda/matmul_kernels.cu`; activations, concatenation
and float-buffer kernels have separate files. Reductions are different:
each output needs its own block, so occupancy recommendations select the block
size but must not cap the number of output blocks. Status-returning launchers
use `cuda_launch_status`, synchronizing only when their existing API requires
it; void launchers remain asynchronous. New `.cu` files must be
listed in both `config.m4` and `Makefile.frag`.

The CUDA allocator reserves up to a quarter of `cuda.memory_size` (at most
16 MB) for small blocks, uses a grow-on-demand pool for larger blocks and
caches up to 64 independent allocations. The limit applies to reserved device
bytes, including the small pool and cached allocations. This allocator uses
a mutex and synchronous `cudaMalloc`/`cudaFree`; compare it with CUDA's
stream-ordered allocator on real GPUs before adopting an asynchronous backend.

Runtime failures use `Cuda\Exception` and its `RuntimeException`,
`InvalidArgumentException`, `OutOfMemoryException`, and `CompilationException`
subclasses. Catch `Cuda\Exception` for all extension failures. Invalid input,
CUDA operations, memory exhaustion, and NVRTC compilation use the corresponding
subclass instead of a PHP `Error` or a warning with a sentinel return value.
Diagnostics emitted during extension initialization and compatibility warnings
remain PHP warnings.
Async status queries and batch results still use booleans to represent normal
pending or per-item states. The C allocator test runs before PHPTs when invoking
`./run-tests.sh`; running `make test` directly only runs the PHP tests.

---

## Requirements

- NVIDIA GPU with CUDA capability
- NVIDIA Driver compatible with CUDA Toolkit
- CUDA Toolkit **11.x+** (12.x recommended)
- PHP **8.0+** (GPU test suite verified with 8.1 and 8.3)
- Linux (tested on Ubuntu / Debian-based systems)
- `gcc`, `g++`, `make`, `autoconf`, `phpize`

---

## Installation

Clone the repository:

```bash
git clone https://github.com/lcmialichi/php-cuda-ext.git
cd php-cuda-ext
```

Build without installing or requiring root:

```bash
./compile.sh
./run-tests.sh --require-gpu
```

The output is in `cuda_build-<PHP major.minor>/modules/cuda.so`. To select
another installed PHP version, pass matching tools (for example):

```bash
PHP_BIN=php8.3 PHPIZE=phpize8.3 PHP_CONFIG=php-config8.3 ./compile.sh
PHP_BIN=php8.3 PHP_CONFIG=php-config8.3 ./run-tests.sh --require-gpu
```

`CUDA_HOME` selects a nonstandard toolkit location; `CUDA_ARCH=sm_86` overrides
GPU architecture detection for cross-builds. `./compile.sh --install` is the
explicit install/INI registration step and needs write access to the PHP
extension/INI directories. Build directories are versioned so PHP ABIs do not
overwrite each other. `./run-tests.sh` always runs CPU-side C checks; with
`--require-gpu` it fails early unless the extension loads and a CUDA device is
visible, rather than reporting only skipped PHPTs.

To compare CPU-to-GPU import paths on your own device after building:

```bash
php -n -d extension=./cuda_build-8.1/modules/cuda.so run_benchmarks.php --import
```

The focused benchmark prepares arrays, packed bytes and temporary files before
timing, and reports the average cost of each import method, including file I/O
for `fromFile()`. Both `--import` and the full run export JSON and HTML under
`benchmarks/reports/`; the full run already includes the import cases. Reports
include median and p95 alongside average time to expose timing spikes.
Benchmarks vary with storage and GPU; `pack()` time is excluded. The reported
memory change comes from PHP's `memory_get_usage(true)`, **not GPU VRAM**.
Metadata-only views such as `flatten()` and `reshape()` do not measure GPU
kernel throughput.

For Docker, `docker compose run --rm php_cuda_dev bash -lc './compile.sh && ./run-tests.sh --require-gpu'`
builds and tests with a host NVIDIA GPU. The default image uses Ubuntu 22.04
and its PHP 8.1 packages, without a third-party PHP PPA. To build another
version, choose a CUDA development image whose distribution provides that PHP
version and set `CUDA_IMAGE` and `PHP_VERSION` as Compose build arguments.

The CUDA toolkit supplies `libcuda.so` stubs for **linking only**. Running the
extension requires the host NVIDIA driver to provide the real `libcuda.so.1`;
do not install or ship the toolkit stub as a runtime driver. `gpus: all` in
Compose exposes the driver and device to the container.

Verify installation:
```bash
php -m | grep cuda
```

## Core Concepts
### CudaArray (GPU Tensor)
``CudaArray`` represents an n-dimensional array stored entirely in GPU memory.

- No implicit CPU ↔ GPU transfers
- Contiguous memory layout
- Supports broadcasting and element-wise operations
- Designed for chained expressions

```php
use Cuda\CudaArray;

$a = CudaArray::ones([3, 3], dtype: 'float32');
$b = CudaArray::full([3, 3], 2.0); // default dtype = float32

$result = ($a * 2.0 + $b) ** 2;
```

All operations above are executed on the GPU.

## Data Transfer

For data already packed in row-major order, bypass recursive PHP array conversion:

```php
$tensor = Cuda\CudaArray::fromBuffer(pack('g*', 1, 2, 3, 4), [2, 2], 'float32');
$raw = Cuda\CudaArray::fromFile('/data/weights.f32', [2, 2], 'float32');
$numpy = Cuda\CudaArray::fromNpy('/data/weights.npy');
```

`fromBuffer()` copies directly from the PHP string to the device. `fromFile()`
reads raw bytes directly into pinned host memory for files of at least 1 MB,
falling back to regular host memory if pinning is unavailable. Both require an
exact shape/dtype byte count and native little-endian numeric data. `fromNpy()`
reads NumPy `.npy` v1-v3 C-order arrays with little-endian float32/64, signed
or unsigned 8/16/32/64-bit integers, or bool; Fortran-order, big-endian, and
other dtypes are rejected rather than silently converted.

`Cuda\CudaArray::where($condition, $x, $y)` selects elements on the GPU with
NumPy-style broadcasting across all three tensors. A nonzero condition is true;
`$x` and `$y` must both be `CudaArray` instances with the same dtype. The
one-argument indices form of NumPy/CuPy `where` is not provided.

```php
use Cuda\CudaArray;
// CPU → GPU
$ca = new CudaArray([[1, 2], [3, 4]]);

// GPU-only allocation
$ones  = CudaArray::ones([1024, 1024]);
$zeros = CudaArray::zeros([512]);

// GPU → CPU
$data = $ca->toArray();

// GPU → Contiguous list 
$host = $ca->toHost();

// Contiguous list → CPU memory
$host->toGpu();

// Contiguous list → PHP Array
$host->toArray();

// Save to file (PHP serialization)
file_put_contents('/data/array.ser', serialize($host));

// Load from file
$restored = unserialize(file_get_contents('/data/array.ser')); // Cuda\ContiguousArray

// Convert back to GPU when needed
$gpu_restored = $restored->toGpu(); // Cuda\CudaArray
```

## Supported Operations
### Arithmetic & Math
- add, subtract, multiply, divide, power
- exp, log, sqrt, abs
- sin, cos, tan
### Reductions
- sum(axis)
- min(axis)
- max(axis)
- prod(axis)
- argMax(axis)
- argMin(axis)
### Shape Manipulation
- reshape(shape)
- flatten()
- transpose(axes)
- concat(tensors, axis)

## Supported Data types
- float32, float64
- uint8, uint16, uint32, uint32
- int8, int16, int32, int64
- bool

## Custom CUDA Kernels (JIT)
Custom kernels are written as CUDA C/C++ source strings and compiled to PTX at runtime with NVRTC. Parameter metadata tells the extension how to marshal PHP values and `CudaArray` buffers when launching the kernel.

### Kernel Definition
```php
$src =  'v_add',
  <<<'CUDA'
extern "C" __global__ void v_add(float *a, float *b, float *c, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
    }
}
CUDA;

$compiler = new Cuda\Compiler(source: $src);
$compiler->kernel('v_add',
  [
    ['name' => 'a', 'type' => 'array', 'dtype' => 'float32'],
    ['name' => 'b', 'type' => 'array', 'dtype' => 'float32'],
    ['name' => 'c', 'type' => 'array', 'dtype' => 'float32'],
    ['name' => 'n', 'dtype' => 'int32'],
  ],
  ['#include <math.h>']
);
```

Supported parameter dtypes include `float32`, `float64`, signed and unsigned integer widths, and `bool`. Use `type => 'array'` for `CudaArray` parameters; omit `type` for scalar values.

### Compilation & Execution

```php
$module = $compiler->compile();
$module->initialize();

$n = 1_048_576;

$a = CudaArray::ones([$n]);
$b = CudaArray::full([$n], 5.0);
$c = CudaArray::zeros([$n]);

$module->launch(
    'v_add',
    args: [$a, $b, $c, $n],
    config: [
        'block' => [256, 1, 1],
        'grid'  => [(int)ceil($n / 256), 1, 1]
    ]
);
```

### Asynchronous Execution
```php
$id = $module->launchAsync('v_add', args: [$a, $b, $c, $n]);
$module->sync();
```
Multiple kernels can be queued and synchronized explicitly.

## Examples
Documented examples are available in the /examples directory:
- Tensor creation and basic operations
- Broadcasting and shape manipulation
- Reductions
- Custom JIT kernels
- Asynchronous execution

## Use Cases
- Numerical computing
- Image and signal processing
- Scientific simulations
- Experimental machine learning pipelines
- GPU-accelerated data processing in PHP

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.