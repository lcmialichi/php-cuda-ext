<?php

use Cuda\Attr as Attr;
use Cuda\CompiledModule;
use Cuda\Compiler;
use Cuda\CudaArray;

class KernelDefs
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

    #[Attr\Kernel(name: 'v_sub')]
    public function vectorSub(
        #[Attr\TensorType] array $a,
        #[Attr\TensorType] array $b,
        #[Attr\TensorType] array &$c,
        #[Attr\IntType] int $n
    ): void {
        /** @var \Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();
        if ($idx < $n) {
            $c[$idx] = $a[$idx] - $b[$idx];
        }
    }

    #[Attr\Kernel(name: 'v_mul')]
    public function vectorMul(
        #[Attr\TensorType] array $a,
        #[Attr\TensorType] array $b,
        #[Attr\TensorType] array &$c,
        #[Attr\IntType] int $n
    ): void {
        /** @var \Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();
        if ($idx < $n) {
            $c[$idx] = $a[$idx] * $b[$idx];
        }
    }

    #[Attr\Kernel(name: 'v_sigmoid')]
    public function sigmoid(
        #[Attr\TensorType] array $in,
        #[Attr\TensorType] array &$out,
        #[Attr\IntType] int $n
    ): void {
        /** @var \Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();
        if ($idx < $n) {
            $out[$idx] = 1.0 / (1.0 + ($cuda->math->exp(-$in[$idx])));
        }
    }

    #[Attr\Kernel(name: 'v_pow_abs')]
    public function powerAbs(
        #[Attr\TensorType] array $in,
        #[Attr\TensorType] array &$out,
        #[Attr\FloatType] float $p,
        #[Attr\IntType] int $n
    ): void {
        /** @var \Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();
        if ($idx < $n) {
            $out[$idx] = $cuda->math->pow($cuda->math->abs($in[$idx]), $p);
        }
    }

    #[Attr\Kernel(name: 'v_log_sqrt')]
    public function logSqrt(
        #[Attr\TensorType] array $in,
        #[Attr\TensorType] array &$out,
        #[Attr\IntType] int $n
    ): void {
        /** @var \Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();
        if ($idx < $n) {
            $val = $cuda->math->abs($in[$idx]) + 1.0;
            $out[$idx] = $cuda->math->log($cuda->math->sqrt($val));
        }
    }

    #[Attr\Kernel(name: 'v_trig_math')]
    public function trigMath(
        #[Attr\TensorType] array $in,
        #[Attr\TensorType] array &$out,
        #[Attr\IntType] int $n
    ): void {
        /** @var \Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();
        if ($idx < $n) {
            $out[$idx] = $cuda->math->sin($in[$idx]) + $cuda->math->cos($in[$idx]);
        }
    }

    #[Attr\Kernel(name: 'matrix_multiply_tiled')]
    public function matrixMultiplyTiled(
        #[Attr\TensorType] array $a,
        #[Attr\TensorType] array $b,
        #[Attr\TensorType] array &$c,
        #[Attr\IntType] int $n
    ): void {
        /** @var \Cuda\Runtime $cuda */
        $tileDim = 16;
        $cuda->__declare_shared($sA, 'float32', 256);
        $cuda->__declare_shared($sB, 'float32', 256);

        $idx = $cuda->globalIdx();
        if ($idx < $n) {
            $c[$idx] = $cuda->math->sin($a[$idx]) + $cuda->math->cos($a[$idx]);
        }

        $tx = $cuda->threadIdx()->x;
        $ty = $cuda->threadIdx()->y;
        $row = $cuda->blockIdx()->y * $tileDim + $ty;
        $col = $cuda->blockIdx()->x * $tileDim + $tx;
        $p_val = 0.0;
        for ($m = 0; $m < ($n / $tileDim); ++$m) {
            $sA[$ty * $tileDim + $tx] = $a[$row * $n + ($m * $tileDim + $tx)];
            $sB[$ty * $tileDim + $tx] = $b[($m * $tileDim + $ty) * $n + $col];
            $cuda->sync->threads();

            for ($k = 0; $k < $tileDim; ++$k) {
                $p_val += $sA[$ty * $tileDim + $k] * $sB[$k * $tileDim + $tx];
            }
            $cuda->sync->threads();
        }

        if ($row < $n && $col < $n) {
            $c[$row * $n + $col] = $p_val;
        }
    }
}

class Profiler
{
    public static function measure(callable $work, int $iterations = 1, bool $warmup = false): array
    {
        if ($warmup) {
            for ($i = 0; $i < 5; $i++) {
                $work();
            }
        }

        $times = [];
        for ($i = 0; $i < $iterations; $i++) {
            $start = hrtime(true);
            $work();
            $end = hrtime(true);
            $times[] = ($end - $start) / 1e+6;
        }

        return [
            'avg' => array_sum($times) / count($times),
            'min' => min($times),
            'max' => max($times),
            'iterations' => $iterations
        ];
    }
}

class Summary
{
    public static function header(string $title)
    {
        echo "\n\n" . str_pad(" $title ", 120, "=", STR_PAD_BOTH) . "\n\n";
    }
    public static function result(string $key, array $stats, string $unit = "ms")
    {
        printf(
            " - %-55s | Avg: %8.3f%s | Min: %8.3f%s | Max: %8.3f%s | loop: %d\n",
            $key,
            $stats['avg'],
            $unit,
            $stats['min'],
            $unit,
            $stats['max'],
            $unit,
            $stats['iterations']
        );
    }
    public static function simple(string $key, $value)
    {
        printf(" - %-55s | %s\n", $key, $value);
    }
}

Summary::header("PHASE 1: JIT COMPILATION & MAPPING");

$compiler = new Compiler();
$defs = new KernelDefs();

$methods = [
    'vectorAdd',
    'vectorSub',
    'vectorMul',
    'sigmoid',
    'powerAbs',
    'logSqrt',
    'trigMath',
    'matrixMultiplyTiled'
];

$astStats = Profiler::measure(function () use ($compiler, $defs, $methods) {
    foreach ($methods as $m) {
        $compiler->kernel([$defs, $m]);
    }
});

Summary::result("AST Generation (Method Registration)", $astStats);

$compileStats = Profiler::measure(function () use ($compiler, &$module) {
    $module = $compiler->compile();
});

Summary::result("PTX Compilation", $compileStats);


$compileStats = Profiler::measure(function () use ($module, &$serialize) {
    $serialize = serialize($module);
});

echo $serialize;
var_dump($compiler);
exit;

$JITStats = Profiler::measure(function () use (&$module): void {
    $module->initialize();
});

Summary::result("JIT Module Initialization", $JITStats);
Summary::header("PHASE 2: INTENSIVE PERFORMANCE COMPARISONS (21 TESTS)");

$n = 512 * 512;
$shape = [512, 512];
$repeat = 10;
$batchItems = [5, 12, 24];

$inputsA = [];
$inputsB = [];
$outputs = [];
for ($i = 0; $i < 25; $i++) {
    $inputsA[] = CudaArray::rand($shape, -1.0, 1.0);
    $inputsB[] = CudaArray::rand($shape, -1.0, 1.0);
    $outputs[] = CudaArray::zeros($shape);
}

$config = ['block' => [256, 1, 1], 'grid' => [(int) ceil($n / 256), 1, 1]];

$kernelsToTest = [
    'v_add' => 'Basic Addition',
    'v_sub' => 'Basic Subtraction',
    'v_mul' => 'Basic Multiplication',
    'v_sigmoid' => 'Math: exp',
    'v_pow_abs' => 'Math: pow/abs',
    'v_log_sqrt' => 'Math: log/sqrt',
    'v_trig_math' => 'Math: sin/cos'
];

foreach ($kernelsToTest as $attrName => $desc) {
    foreach ($batchItems as $bSize) {
        $testLabel = "[$desc] Load: $bSize items";

        $syncStats = Profiler::measure(function () use ($module, $attrName, $inputsA, $inputsB, $outputs, $bSize, $n, $config) {
            for ($i = 0; $i < $bSize; $i++) {
                $args = match ($attrName) {
                    'v_add', 'v_sub', 'v_mul' => [$inputsA[$i], $inputsB[$i], $outputs[$i], $n],
                    'v_pow_abs' => [$inputsA[$i], $outputs[$i], 3.0, $n],
                    default => [$inputsA[$i], $outputs[$i], $n]
                };
                $module->run($attrName, args: $args, config: $config);
            }
        }, $repeat, warmup: true);

        $asyncStats = Profiler::measure(function () use ($module, $attrName, $inputsA, $inputsB, $outputs, $bSize, $n, $config) {
            for ($i = 0; $i < $bSize; $i++) {
                $args = match ($attrName) {
                    'v_add', 'v_sub', 'v_mul' => [$inputsA[$i], $inputsB[$i], $outputs[$i], $n],
                    'v_pow_abs' => [$inputsA[$i], $outputs[$i], 3.0, $n],
                    default => [$inputsA[$i], $outputs[$i], $n]
                };
                $module->runAsync($attrName, args: $args, config: $config);
            }
            $module->sync();
        }, $repeat, warmup: true);

        $gain = (($syncStats['avg'] - $asyncStats['avg']) / $syncStats['avg']) * 100;
        Summary::result("$testLabel (SYNC)", $syncStats);
        Summary::result("$testLabel (ASYNC)", $asyncStats);
        Summary::simple("$testLabel Efficiency Gain", round($gain, 2) . "%");
        echo str_repeat("-", 120) . "\n";
    }
}

Summary::header("PHASE 3: COMPUTE INTENSITY (TILED MATMUL)");

$dim = 1024;
$mA = CudaArray::rand([$dim, $dim], -1, 1);
$mB = CudaArray::rand([$dim, $dim], -1, 1);
$mC = CudaArray::zeros([$dim, $dim]);

$tiledStats = Profiler::measure(function () use ($module, $mA, $mB, $mC, $dim) {
    $module->run('matrix_multiply_tiled', args: [$mA, $mB, $mC, $dim], config: [
        'block' => [16, 16, 1],
        'grid' => [$dim / 16, $dim / 16, 1]
    ]);
}, 20, true);

$ops = 2.0 * pow($dim, 3);
$gflops = ($ops / ($tiledStats['avg'] / 1000)) / 1e9;

Summary::result("Tiled Matrix $dim x $dim", $tiledStats);
Summary::simple("Throughput", round($gflops, 2) . " GFLOPS");

Summary::header("PHASE 4: NUMERIC ACCURACY CHECK");

$input = CudaArray::rand($shape, -1, 1);
$output = CudaArray::zeros($shape);
$module->run('v_sigmoid', args: [$input, $output, $n], config: $config);

$gpuResult = $output->toArray();
$hInput = $input->toArray();

$expected = 1.0 / (1.0 + exp(-$hInput[0][0]));
$passed = (abs($gpuResult[0][0] - $expected) < 0.00001) ? "PASSED" : "FAILED";
Summary::simple("Sigmoid Mathematical Validation", $passed);
Summary::simple("Sample Comparison", "CPU: $expected | GPU: " . $gpuResult[0][0]);

echo "\n\n" . str_repeat("=", 120) . "\n\n";
echo "BENCHMARK SUITE COMPLETE\n";
