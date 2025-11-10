<?php
class CudaBenchmark
{
    public static function runAllTests()
    {
        echo "🎯 CUDA vs CPU BENCHMARK SUITE\n";
        echo str_repeat("=", 50) . "\n";

        self::testSmallTensors();
        self::testMediumTensors();
        self::testLargeTensors();
        self::test2DTensors();
        self::test3DTensors();
        self::testMemoryIntensive();

        echo "\n🎊 BENCHMARK SUITE COMPLETE!\n";
    }

    public static function testSmallTensors()
    {
        echo "\n🔬 1. SMALL TENSORS TEST\n";
        $tests = [
            '16×16×16' => [16, 16, 16],      // 4K
            '32×32×32' => [32, 32, 32],      // 32K
            '64×64×64' => [64, 64, 64],      // 262K
        ];

        foreach ($tests as $name => $dims) {
            self::benchmarkOperation($name, $dims[0], $dims[1], $dims[2]);
        }
    }

    public static function testMediumTensors()
    {
        echo "\n🔬 2. MEDIUM TENSORS TEST\n";
        $tests = [
            '128×128×128' => [128, 128, 128], // 2M
            '256×256×64' => [256, 256, 64],  // 4M
            '512×512×8' => [512, 512, 8],   // 2M
        ];

        foreach ($tests as $name => $dims) {
            self::benchmarkOperation($name, $dims[0], $dims[1], $dims[2]);
        }
    }

    public static function testLargeTensors()
    {
        echo "\n🔬 3. LARGE TENSORS TEST\n";
        $tests = [
            '256×256×256' => [256, 256, 256],
            '512×512×64' => [512, 512, 64],
            '1024×1024×4' => [1024, 1024, 4],
            '1024×1024×32' => [1024, 1024, 32],
        ];

        foreach ($tests as $name => $dims) {
            self::benchmarkOperation($name, $dims[0], $dims[1], $dims[2]);
        }
    }

    public static function test2DTensors()
    {
        echo "\n🔬 4. 2D MATRICES TEST\n";
        $tests = [
            '512×512' => [512, 512, 1],      // 262K
            '1024×1024' => [1024, 1024, 1],  // 1M
            '2048×512' => [2048, 512, 1],    // 1M
            '4096×256' => [4096, 256, 1],    // 1M
        ];

        foreach ($tests as $name => $dims) {
            self::benchmarkOperation($name, $dims[0], $dims[1], $dims[2]);
        }
    }

    public static function test3DTensors()
    {
        echo "\n🔬 5. 3D TENSORS TEST\n";
        $tests = [
            '64×64×64' => [64, 64, 64],      // 262K
            '128×128×32' => [128, 128, 32],  // 524K
            '256×256×16' => [256, 256, 16],  // 1M
            '512×128×16' => [512, 128, 16],  // 1M
        ];

        foreach ($tests as $name => $dims) {
            self::benchmarkOperation($name, $dims[0], $dims[1], $dims[2]);
        }
    }

    public static function testMemoryIntensive()
    {
        echo "\n🔬 6. MEMORY INTENSIVE TEST\n";
        $tests = [
            '512×512×512' => [512, 512, 512], // 134M
            '1024×1024×32' => [1024, 1024, 32], // 33M
            '2048×512×16' => [2048, 512, 16], // 16M
            '2048×512×64' => [2048, 512, 64],
        ];

        foreach ($tests as $name => $dims) {
            self::benchmarkOperation($name, $dims[0], $dims[1], $dims[2], false);
        }
    }

    public static function benchmarkOperation($name, $rows, $cols, $depth, $test_cpu = true)
    {
        $total_elements = $rows * $cols * $depth;

        echo "🧪 {$name} (" . number_format($total_elements) . " elements): ";

        if ($total_elements > 50000000) {
            echo "Creating data...";
            $data = array_fill(0, $rows, array_fill(0, $cols, array_fill(0, $depth, 1.0)));
        } else {
            echo "Creating random data...";
            $data = [];
            for ($i = 0; $i < $rows; $i++) {
                $matrix = [];
                for ($j = 0; $j < $cols; $j++) {
                    $row = [];
                    for ($k = 0; $k < $depth; $k++) {
                        $row[] = (float) rand(1, 100) / 100.0;
                    }
                    $matrix[] = $row;
                }
                $data[] = $matrix;
            }
        }

        try {
            $start_gpu = microtime(true);
            $a = new CudaArray($data);
            $b = new CudaArray($data);

            $gpu_result = $a->multiply($b)->multiply($b)->multiply($a);
            $gpu_time = (microtime(true) - $start_gpu) * 1000;
            $gpu_success = true;
        } catch (Exception $e) {
            $gpu_time = 0;
            $gpu_success = false;
        }

        $cpu_time = 0;
        $cpu_success = false;

        $start_cpu = microtime(true);
        try {
            for ($i = 0; $i < 3; $i++) {
                $cpu_result = [];
                foreach ($data as $i => $matrix) {
                    $cpu_matrix = [];
                    foreach ($matrix as $j => $row) {
                        $cpu_row = [];
                        foreach ($row as $k => $val) {
                            $cpu_row[] = $val * $data[$i][$j][$k];
                        }
                        $cpu_matrix[] = $cpu_row;
                    }
                    $cpu_result[] = $cpu_matrix;
                }
            }

            $cpu_time = (microtime(true) - $start_cpu) * 1000;
            $cpu_success = true;
        } catch (Exception $e) {
            $cpu_time = 0;
            $cpu_success = false;
        }

        echo "GPU: " . ($gpu_success ? round($gpu_time, 1) . "ms" : "FAILED");

        if ($cpu_success) {
            $speedup = $cpu_time / $gpu_time;
            echo ", CPU: " . round($cpu_time, 1) . "ms";
            echo ", " . round($speedup, 1) . "x faster";
        } elseif ($test_cpu && $total_elements > 5000000) {
            echo ", CPU: SKIPPED (too large)";
        } else {
            echo ", CPU: FAILED";
        }

        if ($gpu_success) {
            $throughput = $total_elements / ($gpu_time / 1000);
            echo ", " . number_format(round($throughput)) . " ops/sec";
        }

        echo "\n";

        return [
            'name' => $name,
            'elements' => $total_elements,
            'gpu_time' => $gpu_time,
            'cpu_time' => $cpu_time,
            'gpu_success' => $gpu_success,
            'cpu_success' => $cpu_success
        ];
    }
}

CudaBenchmark::runAllTests();

$cud = new CudaArray([[1, 2], [3, 4]]);
$cud2 = new CudaArray([[5, 6, 5, 6], [7, 8, 9, 10]]);

$newCuda = $cud2->transpose()->matmul($cud)->multiply(1/100);


var_dump($newCuda->getShape());
var_dump($newCuda->toArray());







