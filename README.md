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

---

## Requirements

- NVIDIA GPU with CUDA capability
- NVIDIA Driver compatible with CUDA Toolkit
- CUDA Toolkit **11.x+** (12.x recommended)
- PHP **8.0+**
- Linux (tested on Ubuntu / Debian-based systems)
- `gcc`, `g++`, `make`, `autoconf`, `phpize`

---

## Installation

Clone the repository:

```bash
git clone https://github.com/lcmialichi/php-cuda-ext.git
cd php-cuda-ext
```

Compile and install:

```bash
./compile.sh
```

The script runs:
- ``phpize``
- ``./configure``
- ``make``
- ``make install``

Notice: the script will register cuda extension automatically

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

```php
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
Custom kernels are defined using PHP 8 Attributes and compiled to PTX at runtime.

### Kernel Definition
```php
use Cuda\Attr as Attr;

class Kernels
{
    #[Attr\Kernel(name: 'v_add')]
    public function vectorAdd(
        #[Attr\TensorType] array $a,
        #[Attr\TensorType] array $b,
        #[Attr\TensorType] array &$c,
        #[Attr\IntType] int $n
    ): void {
        $idx = $cuda->globalIdx();
        if ($idx < $n) {
            $c[$idx] = $a[$idx] + $b[$idx];
        }
    }
}
```

### Compilation & Execution

```php
$compiler = new Cuda\Compiler();
$compiler->kernel([new Kernels(), 'vectorAdd']);

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
$id = $module->launchAsync('v_add', args: [...]);
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