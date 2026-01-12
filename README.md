### PHP CUDA Extension

A native PHP extension that provides direct access to NVIDIA CUDA functionality, enabling high-performance GPU computing inside PHP applications.

<div align="center">
  <img src="https://img.shields.io/badge/PHP-8.0+-purple?logo=php">
  <img src="https://img.shields.io/badge/CUDA-11.0%2B-76B900?logo=nvidia">  
  <img src="https://img.shields.io/badge/License-MIT-blue">
  <img src="https://img.shields.io/badge/Platform-Linux-red">
</div>

# ⚠️ NOTICE — Under Development

This extension is **actively under development**.  
It is **not production-ready**, may contain bugs, and its API may change at any time.  
Use **only in testing or experimental environments**.

---

## Requirements
To build and run this extension, you need:

- NVIDIA CUDA Toolkit (12.x recommended)
- Compatible NVIDIA GPU driver
- PHP 8.0+ with support for C extensions
- gcc / g++
- make / autoconf
- Linux (Ubuntu, Debian, CentOS, Arch, etc.)

## Features
 - JIT CUDA Compiler: Write CUDA kernels directly in PHP using Attributes. The extension compiles them to PTX at runtime.
 - Operator Overloading: Use standard math operators (+, -, *, /, **) directly on CudaArray objects.
 - High-Performance Tensors: Optimized CudaArray class for multi-dimensional data management on the GPU.
 - Automatic Broadcasting: Seamlessly perform operations between tensors of different (but compatible) shapes.
 - Async Execution: Support for non-blocking kernel execution with runAsync() and stream synchronization.
 - Advanced Memory Control: Direct access to Shared Memory and thread synchronization (__syncthreads) within PHP.
- Mathematical Library: Built-in GPU-accelerated functions for Trigonometry, Logarithms, and Exponentials.

## How to Compile
```bash
git clone https://github.com/lcmialichi/php-cuda-ext.git/
cd php-cuda-ext
```
Compile and install the extension by running:
```bash
./compile.sh
```

The compile script automatically performs:
1. phpize
2. ./configure
3. make
4. make install

Verify that it loaded correctly:
```bash
php -m | grep cuda
```

## Quick start: CudaArray
### Operator Overloading
CudaArray supports native PHP operator overloading, providing an intuitive syntax for GPU-accelerated tensor operations.
```php
$a = Cuda\CudaArray::ones([3, 3]);
$b = Cuda\CudaArray::full([3, 3], 2.0);

// Mathematical expressions execute entirely on the GPU
$result = ($a * 2.0 + $b) ** 2;
```
Operator overloading enables complex mathematical expressions that execute efficiently on the ``GPU``:

### Basic Operations

```php
$ca = Cuda\CudaArray::ones([4, 4, 4]);

// Slicing and Assignment
$ca[0] = ($ca[1] * 2) + $ca[2];

// Reshaping
$newCa = $ca->reshape([64]);

// Transfer back to CPU
$phpArray = $newCa->toArray();

```

## Methods Reference
### Math & Comparison
- **Arithmetic**: ``add``, ``subtract``, ``multiply``, ``divide``, ``power``, ``neg``, ``matmul``
- **Functions**: ``exp``, ``log``, ``sqrt``, ``abs``, ``ceil``, ``floor``, ``round``
- **Trigonometry**: ``sin``, ``cos``, ``tan``
- **Comparison**: ``gt(>)`` , ``lt(<)`` , ``eq(==)`` , ``ne(!=)`` , ``ge(>=)`` , ``le(<=)``

### Reduction
- ``sum(axis)``, ``min(axis)``, ``max(axis)``, ``prod(axis)``, ``argMax(axis)``, ``argMin(axis)``

### Shape & Manipulation
- ``reshape(shape)``, ``flatten()``, ``transpose(axes)``, ``concat(tensors, axis)``

### New Instance

```php
# Notice: when using the constructor, the PHP array is transferred from CPU → GPU
$ca = new Cuda\CudaArray([[1, 2], [3, 4]]);

# Creates a tensor directly on the GPU, without transferring data from PHP
$ca = Cuda\CudaArray::ones($shape);
$ca = Cuda\CudaArray::zeros($shape);
$ca = Cuda\CudaArray::full($shape, 1.5);
$ca = Cuda\CudaArray::rand($shape, 0, 10);
```


## Custom CUDA Kernels (JIT Compilation)
The extension allows you to define custom GPU kernels using PHP syntax. These are compiled JIT (Just-In-Time) into optimized PTX code.

### 1. Define your Kernels
Use PHP 8 Attributes to define the kernel entry point and variable types.
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

### 2. Compile and Execute
```php
$compiler = new \Cuda\Compiler();
$defs = new MyKernelDefinitions();

// Register and compile to PTX
$compiler->kernel([$defs, 'vectorAdd']);
$module = $compiler->compile();
$module->initialize();

// Prepare Tensors
$n = 1024 * 1024;
$a = \Cuda\CudaArray::ones([$n]);
$b = \Cuda\CudaArray::full([$n], 5.0);
$c = \Cuda\CudaArray::zeros([$n]);

// Launch Kernel
$module->run('v_add', 
    args: [$a, $b, $c, $n], 
    config: [
        'block' => [256, 1, 1], 
        'grid' => [(int)ceil($n / 256), 1, 1]
    ]
);

// launch kernel async
$opId = $module->runAsync('v_add', 
    args: [$a, $b, $c, $n], 
    config: [
        'block' => [256, 1, 1], 
        'grid' => [(int)ceil($n / 256), 1, 1]
    ]
);

$module->sync(); 
```

### Advanced: Shared Memory & Sync
You can implement complex algorithms (like Tiled Matrix Multiplication) using shared memory:

```php
// Inside a kernel method
$cuda->__declare_shared($sharedMem, 'float32', 256);
$cuda->sync->threads(); // __syncthreads()
$val = $cuda->math->sqrt($in[$idx]); // GPU Intrinsics
```

### Cuda\CompiledModule Methods:
```php
$module->initialize(); // if you want to initialize before first op
$module->run();
$id = $module->runAsync(); // returns op id
$module->sync();
$module->isFinished(); // $id as an optional arg
$module->getAsyncStatus($id); // $id as an optional arg
$module->wait();
$module->getPendingOperations();
$module->cancelOperation($id);
$module->cleanup();
$module->hasKernel();
$modules->getKernels();
$module->getPtx();

```

## Run Benchmark
You can run benchmark script to see real execution time
```bash
php benchmarks/cuda_array.php
php benchmarks/kernels.php
```

## Use Cases
 - Machine Learning & AI: GPU-accelerated model inference and preprocessing
 - Data Science & Analytics: Large-scale numerical computations
 - Image & Video Processing: Real-time filtering and transformations
 - Scientific Computing: Complex mathematical simulations
 - Game Development: Physics engines and procedural generation
 - Financial Modeling: Risk analysis and quantitative finance

 ## Contributing
 We welcome contributions from the community!
 
 ##  License
 This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

 ## Acknowledgments
  - NVIDIA for the CUDA parallel computing platform
  -  PHP internals developers and community
  - Contributors and early testers

## Support
Star this repository if you find it interesting!

Follow development progress and report issues on GitHub

Keywords: PHP CUDA extension, GPU computing PHP, NVIDIA PHP, tensor operations, machine learning PHP, high-performance computing, GPU acceleration, scientific computing PHP, CUDA tensor, PHP extension development
</div>
