
<p align="center">
  <a href="https://github.com/lcmialichi/php-cuda-ext">
    <img src="https://repository-images.githubusercontent.com/1091968129/520375bf-6506-4732-9834-9c5b51d9888b" alt="php-cuda-ext banner" width="500px">
  </a>
</p>

<p align="center">
  <strong>Run high-performance GPU computing directly from PHP using NVIDIA CUDA.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PHP-8.0+-purple?logo=php">
  <img src="https://img.shields.io/badge/CUDA-11.0%2B-76B900?logo=nvidia">
  <img src="https://img.shields.io/badge/License-MIT-blue">
  <img src="https://img.shields.io/badge/Platform-Linux-red">
</p>

<p align="center">
  <em>
    No Python. No TensorFlow. No PyTorch.<br>
    Just native PHP + CUDA.
  </em>
</p>

---

## ⚠️ Project Status — Under development

> **This project is under active development.**

- ❌ Not production-ready
- ⚠️ API may change at any time
- 🧪 Intended for research, experimentation, and advanced use cases

Use at your own risk.

---

## What is php-cuda-ext?

`php-cuda-ext` is a **native PHP extension written in C/C++** that provides **direct access to NVIDIA CUDA** from PHP userland.

It enables **GPU-accelerated computing**, including:
- Tensor operations
- Machine learning primitives
- Custom CUDA kernels (JIT-compiled)
- High-performance numerical workloads

All **without leaving PHP**.

---


- ✅ Direct CUDA access from PHP
- ✅ JIT compilation of CUDA kernels written in PHP
- ✅ Operator overloading for GPU tensors
- ✅ Async kernel execution and stream synchronization
- ❌ No external ML frameworks
- ❌ No Python runtime
- ❌ No bindings to TensorFlow / PyTorch

This is **GPU computing at the PHP language level**.

---

## 🖥️ Requirements

To build and run this extension:

- NVIDIA GPU with CUDA support
- NVIDIA Driver (compatible with CUDA Toolkit)
- CUDA Toolkit **12.x recommended**
- PHP **8.0+** (C extension enabled)
- gcc / g++
- make / autoconf
- Linux (Ubuntu, Debian, Arch, CentOS, etc.)

---

## ✨ Features

- **JIT CUDA Compiler**
  - Write CUDA kernels in PHP using PHP 8 Attributes
  - Compiled to PTX at runtime

- **CudaArray (GPU Tensors)**
  - Multi-dimensional arrays stored entirely on the GPU
  - Optimized memory layout and execution

- **Operator Overloading**
  - Use native operators: `+ - * / **`
  - Expressions execute fully on the GPU

- **Automatic Broadcasting**
  - NumPy-like broadcasting rules

- **Async Execution**
  - Non-blocking kernel launches
  - Stream synchronization and async operation tracking

- **Advanced Memory Control**
  - Shared memory
  - Thread synchronization (`__syncthreads`)

- **GPU Math Library**
  - Trigonometry, logarithms, exponentials, intrinsics

---

## 🛠️ Installation & Compilation

Clone the repository:

```bash
git clone https://github.com/lcmialichi/php-cuda-ext.git
cd php-cuda-ext
```

## Compile and install:
```bash
./compile.sh
```
The script automatically runs:
- phpize
- ./configure
- make
- make install

Verify installation:
```bash
php -m | grep cuda
```

## Quick Start — GPU Tensors (CudaArray):
### Operator Overloading

```php
use Cuda\CudaArray;

$a = CudaArray::ones([3, 3]);
$b = CudaArray::full([3, 3], 2.0);

// Executes entirely on the GPU
$result = ($a * 2.0 + $b) ** 2;
```
Complex mathematical executed on the GPU.

### Basic Operations

```php
$ca = Cuda\CudaArray::ones([4, 4, 4]);

// Slicing & assignment
$ca[0] = ($ca[1] * 2) + $ca[2];

// Reshape
$newCa = $ca->reshape([64]);

// Transfer back to CPU
$array = $newCa->toArray();
```

## CudaArray API Overview

### Math & Comparison
- **Arithmetic**: ``add``, ``subtract``, ``multiply``, ``divide``, ``power``, ``neg``, ``matmul``
- **Functions**: ``exp``, ``log``, ``sqrt``, ``abs``, ``ceil``, ``floor``, ``round``
- **Trigonometry**: ``sin``, ``cos``, ``tan``
- **Comparison**: ``gt``, ``lt``, ``eq``, ``ne``, ``ge``, ``le``
- **info**: ``getNdims``, ``getShape``, ``getSize``,  ``getStrides``

### Reductions
- ``sum(axis)``
- ``min(axis)``
- ``max(axis)``
- ``prod(axis)``
- ``argMax(axis)``
- ``argMin(axis)``

### Shape & Manipulation
- ``reshape(shape)``
- ``flatten()``
- ``transpose(axes)``
- ``concat(tensors, axis)``

## Creating Tensors

```php
// Transfers data from CPU → GPU
$ca = new Cuda\CudaArray([[1, 2], [3, 4]]);

// GPU-only creation (no PHP array transfer)
$ca = Cuda\CudaArray::ones($shape);
$ca = Cuda\CudaArray::zeros($shape);
$ca = Cuda\CudaArray::full($shape, 1.5);
$ca = Cuda\CudaArray::rand($shape, 0, 10);

```

## Custom CUDA Kernels (JIT Compilation)
### Define kernels using PHP 8 Attributes.
```php
use Cuda\Attr as Attr;

class MyKernelDefinitions
{
    #[Attr\Kernel(name: 'v_add')]
    public function vectorAdd(
        #[Attr\TensorType] array $a,
        #[Attr\TensorType] array $b,
        #[Attr\TensorType] array &$c,
        #[Attr\IntType] int $n
    ): void {
        /** @var \Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();
        if ($idx < $n) {
            $c[$idx] = $a[$idx] + $b[$idx];
        }
    }
}
```
### Compile & Execute
```php
$compiler = new Cuda\Compiler();
$defs = new MyKernelDefinitions();

$compiler->kernel([$defs, 'vectorAdd']);
$module = $compiler->compile();
$module->initialize();

$n = 1024 * 1024;
$a = Cuda\CudaArray::ones([$n]);
$b = Cuda\CudaArray::full([$n], 5.0);
$c = Cuda\CudaArray::zeros([$n]);

$module->run(
    'v_add',
    args: [$a, $b, $c, $n],
    config: [
        'block' => [256, 1, 1],
        'grid'  => [(int)ceil($n / 256), 1, 1]
    ]
);

```

### Async Execution
```php
$id = $module->runAsync('v_add', args: [...]);
$module->sync();
```

### Advanced: Shared Memory & Synchronization
```php
// Inside a kernel method
$cuda->__declare_shared($shared, 'float32', 256);
$cuda->sync->threads(); // __syncthreads()
$value = $cuda->math->sqrt($input[$idx]);
```

## CompiledModule API Overview
 - ``initialize()``
 - ``run()``
 - ``runAsync()``
 - ``sync()``
 - ``isFinished(id|null)``
 - ``getAsyncStatus(id|null)``
 - ``wait()``
 - ``getPendingOperations()``
 - ``cancelOperation(id)``
 - ``cleanup()``
 - ``hasKernel()``
 - ``getKernels()``
 - ``getPtx()``

## 📚 Examples & Learning Path

To help you get started, we provide a collection of documented examples in the [`/examples`](./examples) directory. These are organized from high-level abstractions to advanced custom kernel development:

| Example | Description |
| :--- | :--- |
| **[Tensor Basics](./examples/01_basics_cuda_array.php)** | Initialization, VRAM allocation, and operator overloading. |
| **[Math & Reductions](./examples/02_math_and_reductions.php)** | Parallel mathematical functions and data aggregations (sum, max). |
| **[Shape Manipulation](./examples/03_advanced_manipulation.php)** | Reshaping, Transposition, and NumPy-style Broadcasting. |
| **[Custom JIT Kernels](./examples/04_custom_jit_kernels.php)** | Writing CUDA kernels in PHP using Attributes and JIT compilation. |
| **[Asynchronous Execution](./examples/05_jit_async_execution.php)** | Non-blocking kernel launches and stream synchronization. |

Explore the [Examples README](./examples/README.md) for a technical overview of the GPU memory lifecycle and the JIT compilation pipeline.

## 📊 Benchmarks
### Run built-in benchmarks:
```bash
composer update

php run_benchmarks.php
```

## Use Cases
- Machine Learning & AI
- Large-scale numerical computing
- Image & video processing
- Scientific simulations
- Physics engines & procedural generation
- Quantitative finance & risk modeling

## Contributing
Contributions are welcome!
- Bug reports
- Performance improvements
- New kernels
- Documentation

## License
 This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- NVIDIA CUDA platform
- PHP internals community
- Early testers and contributors
---
If you find this project interesting, consider starring the repository.