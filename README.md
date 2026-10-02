# PHP GPU Tensors

<p align="center">
    <img src="art/php-gpu-tensors.jpg" alt="PHP GPU Tensors: high-performance computing with PHP and NVIDIA GPUs" width="350">
</p>

Native PHP extension for GPU tensors and NVIDIA CUDA-accelerated numerical
workloads. Build tensor operations and machine-learning data pipelines in PHP,
move data explicitly between host and GPU, and compile custom CUDA C++ kernels
at runtime with NVRTC. No Python runtime required.

**Status:** beta, with the PHP API frozen at `0.1.0`. The freeze does
not imply production readiness: so far, Linux, PHP 8.1 and 8.3, and CUDA 12.3
have been exercised with an NVIDIA RTX A2000. PHP 8.4.26 and 8.5.11 also pass
extension builds with CUDA 12.6.3. PIE 1.5.1 installed and loaded the current
source under PHP 8.4.26 and 8.5.11, and under PHP 8.5 targeting PHP 8.1. The
full GPU suite passes 26/26 tests on PHP 8.5.11 NTS and ZTS, and PHP 8.1.34
ZTS. Four concurrent PHP 8.5 ZTS runtimes also passed tensor and JIT workloads
on an RTX A2000. Other PHP/CUDA versions and GPUs need independent testing.

## Start here

You need a CUDA-capable NVIDIA GPU, a compatible host driver, the CUDA Toolkit
(including NVRTC), PHP development headers (`phpize`, `php-config`), a C/C++
toolchain, `make`, and `autoconf`. The extension builds on Linux. Building
requires the toolkit; running requires the host driver's `libcuda.so.1`.

```bash
git clone https://github.com/lcmialichi/php-gpu-tensors.git
cd php-gpu-tensors
./compile.sh
./run-tests.sh --require-gpu
php -n -d extension=./cuda_build-8.1/modules/cuda.so examples/01_basics_cuda_array.php
```

The build stays in `cuda_build-<PHP major.minor>/modules/cuda.so`; replace
`8.1` above with the PHP version used by `php-config`. To install the extension
and its INI configuration instead, run `./compile.sh --install` with permission
to write to your PHP extension/INI directories. To choose another PHP ABI:

```bash
PHP_BIN=php8.3 PHPIZE=phpize8.3 PHP_CONFIG=php-config8.3 ./compile.sh
PHP_BIN=php8.3 PHP_CONFIG=php-config8.3 ./run-tests.sh --require-gpu
```

### Install with PIE 🥧

PIE 1.5.1 installed the current source under PHP 8.1.34, 8.4.26, and 8.5.11.
The native `hash_file()` `$options` argument was added in PHP 8.1
([PHP.Watch](https://php.watch/codex/hash_file#changes-php-8.1)); the official
PHP 8.1.34 build accepts it, and PIE installation succeeds.

An earlier failure came from the Ubuntu PHP 8.1.2 package used in one test
container, whose `hash_file()` rejected the fourth argument. That result is
specific to that package build and must not be generalized to PHP 8.1 as a
whole. If a distribution's PHP build reproduces it, run PIE under PHP 8.5 and
target PHP 8.1 with matching `--with-php-config=/usr/bin/php-config8.1` and
`--with-phpize-path=/usr/bin/phpize8.1`, or install from source with
`./compile.sh --install`.

The Packagist `0.1.0-beta.2` predates PHP 8.4/8.5 and ZTS support. Use the new
`0.1.0-beta.3` release for those changes.

PIE builds the native extension for the selected PHP installation; it does not
install an NVIDIA driver or CUDA Toolkit. Those must already be available on
the system. GPU execution additionally requires a compatible NVIDIA driver
and visible GPU.

Set `CUDA_HOME` if the toolkit is not at `/usr/local/cuda`, or `CUDA_ARCH=sm_86`
when cross-building. cuBLAS is used when available for compatible larger
matrix products; `CUDA_USE_CUBLAS=no ./compile.sh` builds with the extension's
built-in CUDA matrix kernels instead.

Docker users with the NVIDIA Container Toolkit and a working host driver can
build and test in the development image:

```bash
docker compose run --rm php_cuda_dev bash -lc './compile.sh && ./run-tests.sh --require-gpu'
```

`./run-tests.sh` runs CPU-side C tests and the PHP test suite. Use
`--cpu-only` to run only the host-side C tests in build environments without a
GPU; this does not validate CUDA execution. `--require-gpu` fails immediately
when no GPU is visible, instead of treating skipped GPU tests as success.

## GPU Tensors in PHP

```php
use Cuda\CudaArray;

$input = new CudaArray([[1, 2], [3, 4]], 'float32');
$weights = CudaArray::ones([2, 2]);
$output = $input->add($weights)->multiply(2);
$column_means = $input->mean(0);

print_r($output->toArray()); // [[4, 6], [8, 10]]
echo $output->dtype();       // float32
```

`CudaArray` holds GPU storage; operations return GPU tensors. `toArray()`
transfers to the CPU and expands all values into PHP arrays. Operations include
arithmetic, broadcasting, comparisons, `matmul()` (including batches), shape
views, and `sum()`, `mean()`, `min()`, `max()`, `prod()`, `argMax()`, and
`argMin()` with an optional axis. PHP arithmetic operators also dispatch to
tensor methods. `mean()` reduces all values when called without an axis, or
reduces one dimension when given an axis. It returns `float32` for `float32`
input and `float64` for `float64`, integer, and boolean input.

## PHP GPU Computing for Machine Learning

Use this PHP CUDA extension to build GPU-accelerated numerical steps into PHP
applications: tensor arithmetic, matrix multiplication, broadcasting,
reductions such as `mean()`, and custom CUDA kernels. These primitives can
support machine-learning data preparation and inference workloads while the
data remains in NVIDIA GPU memory. This is a low-level GPU computing library,
not a complete machine-learning framework; model training, automatic
differentiation, and Python interoperability are outside its current scope.

For data already in packed row-major bytes, avoid creating individual PHP
scalars. `fromFile()` reads raw bytes, whereas `fromNpy()` parses NumPy's
`.npy` format:

```php
use Cuda\CudaArray;
use Cuda\HostArray;

$bytes = pack('g*', 1, 2, 3, 4); // little-endian float32
$host = HostArray::fromBuffer($bytes, [2, 2]);
$gpu = $host->toGpu();
$result = CudaArray::where(
    CudaArray::fromBuffer(pack('C*', 1, 0), [2], 'bool'),
    $gpu,
    CudaArray::zeros([2, 2])
);
$cpuCopy = $result->toHost();
$raw = $cpuCopy->toBuffer();
```

`HostArray` is an alias of `Cuda\ContiguousArray`, a contiguous CPU tensor.
Pass `pinned: true` to its constructor or `fromBuffer()` for page-locked host
storage when repeated transfers justify the extra host memory. `where()`
broadcasts its three inputs; its mask treats nonzero values as true, and its
two value tensors must have the same dtype. `.npy` imports support C-order
little-endian numeric and boolean arrays; Fortran order and big-endian data
are rejected.

## Custom kernels

Register the CUDA kernel's argument metadata, compile to PTX, then launch
with explicit grid and block dimensions:

```php
use Cuda\Compiler;
use Cuda\CudaArray;

$source = <<<'CUDA'
extern "C" __global__ void scale(float *data, int factor, int count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) data[index] *= factor;
}
CUDA;

$compiler = new Compiler(source: $source);
$compiler->kernel('scale', [
    ['name' => 'data', 'type' => 'array', 'dtype' => 'float32'],
    ['name' => 'factor', 'dtype' => 'int32'],
    ['name' => 'count', 'dtype' => 'int32'],
]);
$module = $compiler->compile();
$module->initialize();

$data = CudaArray::ones([512]);
$module->launch('scale',
    args: [$data, 3, 512],
    config: ['block' => [256, 1, 1], 'grid' => [2, 1, 1]]
);
print_r(array_slice($data->toArray(), 0, 4)); // [3, 3, 3, 3]
```

`launch()` synchronizes; `launchAsync()` returns an operation ID for `sync()`
or `wait()`. Keep tensors alive until asynchronous work finishes. See
[the JIT examples](examples/04_custom_jit_kernels.php) and
[asynchronous execution](examples/05_jit_async_execution.php).

## API and limits

| API | Purpose |
| --- | --- |
| `Cuda\CudaArray` | GPU allocation, tensor math, reductions, views, imports and `where()` |
| `Cuda\HostArray` / `Cuda\ContiguousArray` | CPU storage, packed buffers, optional pinned memory and `toGpu()` |
| `Cuda\Compiler` / `Cuda\CompiledModule` | NVRTC compilation, cached PTX, synchronous and asynchronous kernels |
| `cuda_get_device_count()` and other `cuda_*` functions | Device selection, properties, memory and synchronization |
| `Cuda\Exception` | Base class for runtime, argument, allocation and compilation errors |

The annotated signatures are in [class stubs](stubs/cuda.stub.php) and
[device function stubs](stubs/cuda_methods.stub.php); runnable examples live
in [examples](examples/README.md). `astype()` currently supports only the
same dtype. GPU data has no CPU fallback. The project does not yet provide a
stable API or automatic kernel fusion.

## Contribute

Contributions can be code, tests, documentation, runnable examples, or reports
from another PHP/CUDA/GPU combination. Browse
[open issues](https://github.com/lcmialichi/php-gpu-tensors/issues),
[report a bug](https://github.com/lcmialichi/php-gpu-tensors/issues/new?template=bug_report.yml),
or [propose a feature](https://github.com/lcmialichi/php-gpu-tensors/issues/new?template=feature_request.yml).
Start with the [contribution guide](CONTRIBUTING.md); documentation, examples,
and host-side tests are possible without an NVIDIA GPU.

## Benchmarks

The benchmark suite is maintained in the separate
[PHP GPU Tensors Benchmarks repository](https://github.com/lcmialichi/php-gpu-tensors-benchmarks).
It includes focused `--matmul` and `--import` runs and JSON/HTML reports.

For possible directions, see the [roadmap](ROADMAP.md). A feature does not
need to be listed there to be worth discussing.

Licensed under the [MIT License](LICENSE).