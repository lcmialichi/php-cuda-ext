<?php
if (!extension_loaded('cuda')) {
    die("Extension cuda not load, see README.md to compile the extension .\n");
}

class CudaBenchmark
{
    const ITERATIONS = 3;

    public static function runAllTests()
    {
        echo "   CUDA vs CPU BENCHMARK SUITE\n";
        echo "   Testing running " . self::ITERATIONS . " times per operation.\n";
        echo str_repeat("=", 70) . "\n";

        $size_tests = [
            '1. SMALL TENSORS (16K - 1M)' => [
                '16x16x16' => [16, 16, 16],
                '64x64x64' => [64, 64, 64],
                '128x128x64' => [128, 128, 64],
            ],
            '2. MEDIUM MATRICES (2D/3D: 2M - 4M)' => [
                '1024x1024' => [1024, 1024, 1],
                '128x128x128' => [128, 128, 128],
                '256x256x64' => [256, 256, 64],
            ],
            '3. LARGE MATRICES (4M - 32M)' => [
                '512x512x16' => [512, 512, 16],
                '1024x1024x4' => [1024, 1024, 4],
                '1024x1024x32' => [1024, 1024, 32],
            ],
        ];

        foreach ($size_tests as $suite_name => $tests) {
            echo "\n\n=== SUITE: {$suite_name} ===\n";
            self::runElementWiseBenchmarks($tests);
            self::runReductionBenchmarks($tests);
        }

        echo "\n\n BENCHMARK SUITE COMPLETE!\n";
    }

    public static function runElementWiseBenchmarks($tests)
    {
        echo "[ARITHMETIC (Element-Wise)]\n";

        $ops = [
            'Add/Mul/Sub' => [
                'gpu' => fn($a) => $a + 1.2 * 3.5 - 0.5,
                'cpu' => fn($v) => $v + 1.2 * 3.5 - 0.5
            ],
            'Power' => [
                'gpu' => fn($a) => $a ** 2.0,
                'cpu' => fn($v) => pow($v, 2.0)
            ],
            'Log/Exp' => [
                'gpu' => fn($a) => $a->exp()->log(),
                'cpu' => fn($v) => log(exp($v))
            ],
        ];

        foreach ($ops as $op_name => $functions) {
            foreach ($tests as $name => $dims) {
                self::benchmarkOperation($op_name, $name, $dims, $functions, self::ITERATIONS);
            }
        }
    }

    public static function runReductionBenchmarks($tests)
    {
        echo "\n\t[REDUCTION OPERATIONS]\n";

        $ops = [
            'Sum (Full)' => [
                'gpu' => fn($a) => $a->sum(),
                'cpu' => null
            ],
            'Max (Full)' => [
                'gpu' => fn($a) => $a->max(),
                'cpu' => null
            ],
        ];

        foreach ($ops as $op_name => $functions) {
            foreach ($tests as $name => $dims) {
                self::benchmarkOperation($op_name, $name, $dims, $functions, self::ITERATIONS, 100, 0);
            }
        }
    }

    public static function benchmarkOperation($op_name, $name, $dims, $functions, $runs, $max_val = 10.0, $min_val = 10.0)
    {
        [$rows, $cols, $depth] = $dims;
        $total_elements = $rows * $cols * $depth;
        $test_reduction = is_null($functions['cpu']);
        $element_count_format = number_format($total_elements);

        $test_cpu = ($total_elements < 10000000) || $test_reduction;

        echo "  - {$op_name} ({$name} - {$element_count_format} elems): ";

        $gpu_time_total = 0;
        $gpu_success = true;

        try {
            $gpu_init_time = microtime(true);
            $a = CudaArray::rand([$rows, $cols, $depth], $min_val, $max_val);
            $gpu_init_time_result = (microtime(true) - $gpu_init_time) * 1000;

            $op_gpu = $functions['gpu'];
            $start_gpu = microtime(true);
            for ($i = 0; $i < $runs; $i++) {
                $result = $op_gpu($a);
            }
            $gpu_time_total = (microtime(true) - $start_gpu) * 1000 / $runs;

            $result->toArray();

        } catch (Exception $e) {
            $gpu_time_total = 0;
            $gpu_success = false;
        }

        $cpu_time_total = 0;
        $cpu_success = false;

        if ($test_cpu) {
            $cpu_init_time = microtime(true);
            $cpu_data = self::generateRandomArray($rows, $cols, $depth, $min_val, $max_val);
            $cpu_init_time_result = (microtime(true) - $cpu_init_time) * 1000;

            $op_cpu = $functions['cpu'];
            $start_cpu = microtime(true);
            try {
                for ($i = 0; $i < $runs; $i++) {
                    if ($test_reduction) {
                        $cpu_result = self::performCpuReduction($cpu_data, $op_name);
                    } else {
                        $cpu_result = self::performCpuElementWise($cpu_data, $op_cpu);
                    }
                }
                $cpu_time_total = (microtime(true) - $start_cpu) * 1000 / $runs;
                $cpu_success = true;
            } catch (Exception $e) {
                $cpu_time_total = 0;
                $cpu_success = false;
            }
        }

        self::displayResults($total_elements, $gpu_success, $gpu_init_time_result, $gpu_time_total, $cpu_success, $cpu_init_time_result ?? 0, $cpu_time_total, $test_cpu);
    }

    public static function displayResults($total_elements, $gpu_success, $gpu_init, $gpu_time, $cpu_success, $cpu_init, $cpu_time, $test_cpu)
    {
        $output = "";

        if ($gpu_success) {
            $output .= "GPU: init: " . round($gpu_init, 1) . "ms, exec: " . round($gpu_time, 1) . "ms";
            $throughput = $total_elements / ($gpu_time / 1000);
        } else {
            $output .= "GPU: FAILED";
        }

        if ($test_cpu) {
            if ($cpu_success) {
                $speedup = $cpu_time / $gpu_time;
                $output .= ", CPU: init: " . round($cpu_init, 1) . "ms, exec: " . round($cpu_time, 1) . "ms";
                $output .= " | " . round($speedup, 1) . "x faster (GPU)";
            } else {
                $output .= ", CPU: FAILED";
            }
        } elseif (!$test_cpu && $total_elements > 10000000) {
            $output .= ", CPU: SKIPPED (Memory intensive)";
        }

        if ($gpu_success) {
            $output .= " | " . number_format(round($throughput)) . " ops/sec (GPU)";
        }

        echo $output . "\n";
    }

    public static function performCpuElementWise(array $data, callable $op_cpu)
    {
        $result = [];
        foreach ($data as $i => $matrix) {
            $cpu_matrix = [];
            foreach ($matrix as $j => $row) {
                $cpu_row = [];
                foreach ($row as $k => $val) {
                    $cpu_row[] = $op_cpu($val);
                }
                $cpu_matrix[] = $cpu_row;
            }
            $result[] = $cpu_matrix;
        }
        return $result;
    }

    public static function performCpuReduction(array $data, $op_name)
    {
        $all_rows = [];

        foreach ($data as $matrix) {
            if (!is_array($matrix)) {
                $all_rows[] = [$matrix];
                continue;
            }
            foreach ($matrix as $row) {
                $all_rows[] = is_array($row) ? $row : [$row];
            }
        }

        if (empty($all_rows)) {
            return ($op_name === 'Max') ? -INF : 0;
        }

        $flat = array_merge(...$all_rows);
        if (str_contains($op_name, 'Sum')) {
            return array_sum($flat);
        } elseif (str_contains($op_name, 'Max')) {
            return max($flat);
        }
        
        return null;
    }
    public static function generateRandomArray($rows, $cols, $depth, $min_val, $max_val)
    {
        $data = [];
        $range = $max_val - $min_val;
        for ($r = 0; $r < $rows; $r++) {
            $matrix = [];
            for ($c = 0; $c < $cols; $c++) {
                $row = [];
                for ($d = 0; $d < $depth; $d++) {
                    $row[] = $min_val + ($range * (mt_rand() / mt_getrandmax()));
                }
                $matrix[] = $row;
            }
            $data[] = $matrix;
        }
        return $data;
    }
}

CudaBenchmark::runAllTests();
