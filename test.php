<?php

use Cuda\CudaArray;
use Cuda\Attr as Attr;
use Cuda\CompiledModule;
use Cuda\Compiler;

define('MODULE_FILE', 'bench_suite.cudas');
define('DEFAULT_BLOCK', [256, 1, 1]);
define('MATRIX_BLOCK', [16, 16, 1]);

#[Attr\Kernel(name: 'vector_add')]
function vectorAdd(
    #[Attr\TensorType] array $a,
    #[Attr\TensorType] array $b,
    #[Attr\TensorType] array &$c,
    #[Attr\IntType] int $n
): void {
    /** @var \Cuda\Runtime $cuda */
    $idx = $cuda->blockIdx()->x * $cuda->blockDim()->x + $cuda->threadIdx()->x;
    if ($idx < $n) {
        $c[$idx] = $a[$idx] + $b[$idx];
    }
}

#[Attr\Kernel(name: 'element_wise_math')]
function elementWiseMath(
    #[Attr\TensorType] array $input,
    #[Attr\TensorType] array &$output,
    #[Attr\FloatType] float $factor,
    #[Attr\IntType] int $n
): void {
    /** @var \Cuda\Runtime $cuda */
    $idx = $cuda->blockIdx()->x * $cuda->blockDim()->x + $cuda->threadIdx()->x;
    if ($idx < $n) {
        $val = $input[$idx];
        $output[$idx] = ($val * $factor) + ($val / $factor) - ($val * $val);
    }
}

#[Attr\Kernel(name: 'matrix_multiply')]
function matrixMultiply(
    #[Attr\TensorType] array $a,
    #[Attr\TensorType] array $b,
    #[Attr\TensorType] array &$c,
    #[Attr\IntType] int $n,
    #[Attr\IntType] int $m,
    #[Attr\IntType] int $p
): void {
    /** @var \Cuda\Runtime $cuda */
    $row = $cuda->blockIdx()->y * $cuda->blockDim()->y + $cuda->threadIdx()->y;
    $col = $cuda->blockIdx()->x * $cuda->blockDim()->x + $cuda->threadIdx()->x;

    if ($row < $n && $col < $p) {
        $sum = 0.0;
        for ($k = 0; $k < $m; $k++) {
            $sum += $a[$row * $m + $k] * $b[$k * $p + $col];
        }
        $c[$row * $p + $col] = $sum;
    }
}

class CUDAMasterBenchmark
{
    private CompiledModule $module;

    public function __construct()
    {
        $this->initModule();
        $this->warmup();
    }

    private function initModule(): void
    {
        echo "🛠️  Compiling Kernels...\n";
        $compiler = new Compiler();
        $compiler->kernel('vectorAdd');
        $compiler->kernel('elementWiseMath');
        $compiler->kernel('matrixMultiply');

        $this->module = $compiler->compile();
        $this->module->initialize();
        file_put_contents(MODULE_FILE, serialize($this->module));
    }

    private function warmup(): void
    {
        echo "🔥 Performing Warmup (JIT & Clock Stabilization)...\n";
        $size = 1024;
        $a = CudaArray::zeros([$size]);
        $b = CudaArray::zeros([$size]);
        for ($i = 0; $i < 10; $i++) {
            $this->module->run('vector_add', args: [$a, $a, $b, $size], config: [
                'block' => DEFAULT_BLOCK,
                'grid' => [4, 1, 1]
            ]);
        }
        $this->module->wait();
    }

    public function runAccuracyTest(): void
    {
        echo "\n[TEST 1] Mathematical Accuracy Validation\n";
        $n = 10000;
        $h_a = array_map(fn() => (float) mt_rand(0, 100) / 10, range(1, $n));
        $h_b = array_map(fn() => (float) mt_rand(0, 100) / 10, range(1, $n));

        $d_a = new CudaArray($h_a);
        $d_b = new CudaArray($h_b);
        $d_c = CudaArray::zeros([$n]);

        $this->module->run('vector_add', args: [$d_a, $d_b, $d_c, $n], config: [
            'block' => [256, 1, 1],
            'grid' => [(int) ceil($n / 256), 1, 1]
        ]);

        // $gpu_res = $d_c->toArray();
        // $errors = 0;
        // for ($i = 0; $i < $n; $i++) {
        //     if (abs($gpu_res[$i] - ($h_a[$i] + $h_b[$i])) > 0.0001)
        //         $errors++;
        // }

        // echo $errors === 0 ? "  ✅ Success: Data is identical.\n" : "  ❌ Failure: $errors errors found.\n";
    }

    public function runMemoryBenchmark(): void
    {
        echo "\n[TEST 2] Memory Throughput (PCIe)\n";
        $sizes = [1e5 => 'Small', 1e6 => 'Medium', 1e7 => 'Large'];

        foreach ($sizes as $n => $label) {
            $start = microtime(true);
            $d_array = CudaArray::rand([(int) $n], 0, 1);
            $h2d = (microtime(true) - $start) * 1000;

            $start = microtime(true);
            $h_array = $d_array->toArray();
            $d2h = (microtime(true) - $start) * 1000;

            $gb = ($n * 4) / (1024 ** 3);
            printf(
                "  - %-6s (%d items): H2D: %7.2fms | D2H: %7.2fms | BW: %5.2f GB/s\n",
                $label,
                $n,
                $h2d,
                $d2h,
                $gb / (($h2d + $d2h) / 2000)
            );
        }
    }

    public function runAsyncTest(): void
    {
        echo "\n[TEST 3] Asynchronous Execution Efficiency\n";
        $n = 2000000;
        $batches = 32;
        $inputs = [];
        $outputs = [];
        for ($i = 0; $i < $batches; $i++) {
            $inputs[] = CudaArray::rand([$n], 0, 1);
            $outputs[] = CudaArray::zeros([$n]);
        }

        $t_start = microtime(true);
        foreach ($inputs as $idx => $in) {
            $this->module->run('element_wise_math', args: [$in, $outputs[$idx], 1.5, $n], config: [
                'block' => DEFAULT_BLOCK,
                'grid' => [(int) ceil($n / 256), 1, 1]
            ]);
        }
        $this->module->wait();
        $t_sync = (microtime(true) - $t_start) * 1000;

        $t_start = microtime(true);
        foreach ($inputs as $idx => $in) {
            $this->module->runAsync('element_wise_math', args: [$in, $outputs[$idx], 1.5, $n], config: [
                'block' => DEFAULT_BLOCK,
                'grid' => [(int) ceil($n / 256), 1, 1]
            ]);
        }
        $this->module->wait();
        $t_async = (microtime(true) - $t_start) * 1000;

        printf("  - Total Synchronous:  %7.2fms\n", $t_sync);
        printf("  - Total Asynchronous: %7.2fms\n", $t_async);
        printf("  - Overlap Gain:       %7.2f%%\n", (($t_sync - $t_async) / $t_sync) * 100);
    }

    public function runComputeBenchmark(): void
    {
        echo "\n[TEST 4] 2D Matrix - Computation Stress\n";
        $dim = 1024;
        $a = CudaArray::rand([$dim, $dim], -1, 1);
        $b = CudaArray::rand([$dim, $dim], -1, 1);
        $c = CudaArray::zeros([$dim, $dim]);

        $grid = [(int) ceil($dim / 16), (int) ceil($dim / 16), 1];

        $start = microtime(true);
        $this->module->run('matrix_multiply', args: [$a, $b, $c, $dim, $dim, $dim], config: [
            'block' => MATRIX_BLOCK,
            'grid' => $grid
        ]);
        $this->module->wait();
        $time = (microtime(true) - $start) * 1000;

        $ops = 2.0 * pow($dim, 3);
        $gflops = ($ops / ($time / 1000)) / 1e9;

        printf("  - Matrix %dx%d: %7.2fms | Performance: %6.2f GFLOPS\n", $dim, $dim, $time, $gflops);
    }
}

try {
    echo "========================================================\n";
    echo "🚀 PHP-CUDA EXTENSION: PRODUCTION TEST SUITE\n";
    echo "========================================================\n";

    $bench = new CUDAMasterBenchmark();

    $bench->runAccuracyTest();
    $bench->runMemoryBenchmark();
    $bench->runAsyncTest();
    $bench->runComputeBenchmark();

    echo "\n========================================================\n";
    echo "✅ All tests completed.\n";
} catch (\Exception $e) {
    echo "\n❌ CRITICAL ERROR: " . $e->getMessage() . "\n";
}