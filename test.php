<?php

use Cuda\CudaArray;
use Cuda\CompiledModule;
use Cuda\Compiler;

class CudaAsyncVsSyncBenchmark
{
    private CompiledModule $module;
    private array $testResults = [];
    private float $tolerance = 0.0001;

    public function __construct(bool $compileNew = false)
    {
        $this->initModule($compileNew);
    }

    private function initModule(bool $compileNew): void
    {
        $cacheFile = 'cuda_module.cache';

        if (!$compileNew && file_exists($cacheFile)) {
            echo "📦 Loading cached module...\n";
            $this->module = unserialize(file_get_contents($cacheFile));
            $this->module->initialize();
        } else {
            echo "🛠️  Compiling new module...\n";
            require_once 'kernel_def.php';
            $compiler = new Compiler();
            $compiler->kernel('vectorAdd');
            $compiler->kernel('elementWiseMath');
            $compiler->kernel('matrixMultiply');
            $compiler->kernel('complexMath');
            $compiler->kernel('reduceSum');
            $compiler->kernel('stencil3Pt');
            $compiler->kernel('saxpy');
            $compiler->kernel('conditionalOps');
            $compiler->kernel('memCopy');
            $compiler->kernel('mixedOperations');
            $this->module = $compiler->compile(target: 'sm_86');
            $this->module->initialize();
            file_put_contents($cacheFile, serialize($this->module));
        }

        $this->validateModule();
    }

    private function validateModule(): void
    {
        if (!$this->module->hasKernel('element_wise_math')) {
            throw new RuntimeException("Required kernel not found");
        }
    }

    public function runComprehensiveComparison(): array
    {
        echo "\n" . str_repeat('=', 80) . "\n";
        echo "🏎️  CUDA ASYNC vs SYNC COMPREHENSIVE BENCHMARK\n";
        echo str_repeat('=', 80) . "\n";

        $testCases = [
            ['size' => 100000, 'batch' => 10, 'name' => 'Small'],
            ['size' => 1000000, 'batch' => 10, 'name' => 'Medium'],
            ['size' => 5000000, 'batch' => 5, 'name' => 'Large'],
            ['size' => 1000000, 'batch' => 100, 'name' => 'ManySmallOps'],
        ];

        foreach ($testCases as $case) {
            $this->testCase = $case;
            $this->runSingleTestCase($case);
        }

        $this->printSummary();
        return $this->testResults;
    }

    private function runSingleTestCase(array $case): void
    {
        echo "\n📊 Test Case: {$case['name']} (Size: {$case['size']}, Batch: {$case['batch']})\n";
        echo str_repeat('-', 60) . "\n";

        $inputs = [];
        $outputsSync = [];
        $outputsAsync = [];

        for ($i = 0; $i < $case['batch']; $i++) {
            $inputs[] = CudaArray::rand([$case['size']], 0, 1);
            $outputsSync[] = CudaArray::zeros([$case['size']]);
            $outputsAsync[] = CudaArray::zeros([$case['size']]);
        }

        $this->module->cleanup();

        $syncTimes = [];
        $syncStartTotal = microtime(true);

        for ($i = 0; $i < $case['batch']; $i++) {
            $start = microtime(true);
            $this->module->run('element_wise_math', [
                'block' => [256, 1, 1],
                'grid' => [(int) ceil($case['size'] / 256), 1, 1]
            ], [$inputs[$i], $outputsSync[$i], 1.5, $case['size']]);
            $syncTimes[] = (microtime(true) - $start) * 1000; // ms
        }

        $syncTotal = (microtime(true) - $syncStartTotal) * 1000;

        $asyncTimes = [];
        $operationIds = [];
        $asyncStartTotal = microtime(true);

        for ($i = 0; $i < $case['batch']; $i++) {
            $start = microtime(true);
            $result = $this->module->runAsync('element_wise_math', [
                'block' => [256, 1, 1],
                'grid' => [(int) ceil($case['size'] / 256), 1, 1]
            ], [$inputs[$i], $outputsAsync[$i], 1.5, $case['size']]);

            if ($result) {
                $operationIds[] = $result;
            }
            $asyncTimes[] = (microtime(true) - $start) * 1000;
        }

        $this->module->sync();
        $asyncTotal = (microtime(true) - $asyncStartTotal) * 1000;

        $accuracy = $this->verifyAccuracy($outputsSync, $outputsAsync, $case['size']);

        $stats = [
            'name' => $case['name'],
            'size' => $case['size'],
            'batch' => $case['batch'],
            'sync_total_ms' => $syncTotal,
            'async_total_ms' => $asyncTotal,
            'sync_avg_ms' => array_sum($syncTimes) / count($syncTimes),
            'async_avg_ms' => array_sum($asyncTimes) / count($asyncTimes),
            'speedup' => $syncTotal / max($asyncTotal, 0.001),
            'accuracy_match' => $accuracy,
            'sync_times' => $syncTimes,
            'async_times' => $asyncTimes,
            'operation_ids' => $operationIds,
        ];

        $this->printTestCaseResults($stats);
        $this->testResults[] = $stats;
    }

    private function verifyAccuracy(array $syncOutputs, array $asyncOutputs, int $size): bool
    {
        for ($batch = 0; $batch < count($syncOutputs); $batch++) {
            $syncData = $syncOutputs[$batch]->toArray();
            $asyncData = $asyncOutputs[$batch]->toArray();

            for ($i = 0; $i < min(1000, $size); $i++) {
                if (abs($syncData[$i] - $asyncData[$i]) > $this->tolerance) {
                    echo "⚠️  Accuracy mismatch at batch $batch, index $i: ";
                    echo "Sync={$syncData[$i]}, Async={$asyncData[$i]}\n";
                    return false;
                }
            }
        }
        return true;
    }

    private function printTestCaseResults(array $stats): void
    {
        printf(
            "Sync Total:     %8.2f ms (Avg: %6.2f ms)\n",
            $stats['sync_total_ms'],
            $stats['sync_avg_ms']
        );
        printf(
            "Async Total:    %8.2f ms (Avg: %6.2f ms)\n",
            $stats['async_total_ms'],
            $stats['async_avg_ms']
        );
        printf("Speedup:        %8.2fx\n", $stats['speedup']);
        printf(
            "Time Saved:     %8.2f ms\n",
            $stats['sync_total_ms'] - $stats['async_total_ms']
        );
        printf(
            "Accuracy:       %s\n",
            $stats['accuracy_match'] ? "✅ PASS" : "❌ FAIL"
        );

        if (count($stats['sync_times']) > 1) {
            $syncStd = $this->calculateStdDev($stats['sync_times']);
            $asyncStd = $this->calculateStdDev($stats['async_times']);
            printf("Sync Std Dev:   %8.2f ms\n", $syncStd);
            printf("Async Std Dev:  %8.2f ms\n", $asyncStd);
        }

        if (!empty($stats['operation_ids'])) {
            printf(
                "Async Ops:      %d operations tracked\n",
                count($stats['operation_ids'])
            );

            $status = $this->module->getAsyncStatus();
            if (!empty($status)) {
                printf(
                    "Pending Ops:    %d\n",
                    count($this->module->getPendingOperations())
                );
            }
        }
    }

    private function calculateStdDev(array $values): float
    {
        $avg = array_sum($values) / count($values);
        $sum = 0;
        foreach ($values as $value) {
            $sum += pow($value - $avg, 2);
        }
        return sqrt($sum / count($values));
    }

    private function printSummary(): void
    {
        echo "\n" . str_repeat('=', 80) . "\n";
        echo "📈 BENCHMARK SUMMARY\n";
        echo str_repeat('=', 80) . "\n";

        printf(
            "%-15s %-10s %-10s %-12s %-12s %-10s\n",
            'Test Case',
            'Size',
            'Batch',
            'Sync (ms)',
            'Async (ms)',
            'Speedup'
        );
        echo str_repeat('-', 80) . "\n";

        foreach ($this->testResults as $result) {
            printf(
                "%-15s %-10s %-10d %-12.2f %-12.2f %-10.2fx\n",
                $result['name'],
                $this->formatNumber($result['size']),
                $result['batch'],
                $result['sync_total_ms'],
                $result['async_total_ms'],
                $result['speedup']
            );
        }

        $totalSync = array_sum(array_column($this->testResults, 'sync_total_ms'));
        $totalAsync = array_sum(array_column($this->testResults, 'async_total_ms'));
        $avgSpeedup = $totalSync / max($totalAsync, 0.001);

        echo str_repeat('-', 80) . "\n";
        printf(
            "%-15s %-10s %-10s %-12.2f %-12.2f %-10.2fx\n",
            'TOTAL',
            '',
            '',
            $totalSync,
            $totalAsync,
            $avgSpeedup
        );

        echo "\n" . str_repeat('=', 80) . "\n";
        echo "💡 RECOMMENDATIONS\n";
        echo str_repeat('=', 80) . "\n";

        if ($avgSpeedup > 1.5) {
            echo "✅ Significant improvement with async operations.\n";
            echo "   Consider using async for:\n";
            echo "   - Batch processing\n";
            echo "   - Overlapping I/O with computation\n";
            echo "   - Independent operations\n";
        } elseif ($avgSpeedup > 1.1) {
            echo "⚠️  Moderate improvement with async operations.\n";
            echo "   Use async when:\n";
            echo "   - Processing large batches\n";
            echo "   - Operations are independent\n";
        } else {
            echo "⚠️  Minimal improvement with async operations.\n";
            echo "   Consider:\n";
            echo "   - Increasing batch size\n";
            echo "   - Using larger data sizes\n";
            echo "   - Checking GPU utilization\n";
        }
    }

    private function formatNumber(int $num): string
    {
        if ($num >= 1000000) {
            return sprintf("%.1fM", $num / 1000000);
        }
        if ($num >= 1000) {
            return sprintf("%.1fK", $num / 1000);
        }
        return (string) $num;
    }

    public function runMemoryTransferTest(): void
    {
        echo "\n" . str_repeat('=', 80) . "\n";
        echo "💾 MEMORY TRANSFER IMPACT ANALYSIS\n";
        echo str_repeat('=', 80) . "\n";

        $sizes = [10000, 100000, 1000000];

        foreach ($sizes as $size) {
            $data = CudaArray::rand([$size], 0, 1);
            $output = CudaArray::zeros([$size]);

            $start = microtime(true);
            $values = $data->toArray();
            $d2hTime = (microtime(true) - $start) * 1000;

            $start = microtime(true);
            $copy = new CudaArray($values);
            $h2dTime = (microtime(true) - $start) * 1000;

            $start = microtime(true);
            $this->module->run('element_wise_math', [
                'block' => [256, 1, 1],
                'grid' => [(int) ceil($size / 256), 1, 1]
            ], [$data, $output, 1.5, $size]);
            $opTime = (microtime(true) - $start) * 1000;

            printf(
                "Size: %-10s | D2H: %6.2fms | H2D: %6.2fms | Op: %6.2fms | Overhead: %.1f%%\n",
                $this->formatNumber($size),
                $d2hTime,
                $h2dTime,
                $opTime,
                (($d2hTime + $h2dTime) / $opTime) * 100
            );
        }
    }

    public function runConcurrencyLimitTest(): void
    {
        echo "\n" . str_repeat('=', 80) . "\n";
        echo "🔀 CONCURRENCY LIMIT TEST\n";
        echo str_repeat('=', 80) . "\n";

        $size = 100000;
        $maxConcurrent = 50;

        $data = CudaArray::rand([$size], 0, 1);
        $outputs = [];

        for ($i = 0; $i < $maxConcurrent; $i++) {
            $outputs[] = CudaArray::zeros([$size]);

            $success = $this->module->runAsync('element_wise_math', [
                'block' => [256, 1, 1],
                'grid' => [(int) ceil($size / 256), 1, 1]
            ], [$data, $outputs[$i], 1.5, $size]);

            if (!$success) {
                echo "⚠️  Failed to launch async operation #$i\n";
                break;
            }

            $pending = $this->module->getPendingOperations();
            echo "Launched $i operations, pending: " . count($pending) . "\n";
        }

        $this->module->sync();
        echo "✅ All operations completed\n";
    }
}

try {
    $benchmark = new CudaAsyncVsSyncBenchmark(compileNew: false);

    $results = $benchmark->runComprehensiveComparison();

    $benchmark->runMemoryTransferTest();
    $benchmark->runConcurrencyLimitTest();

    echo "\n" . str_repeat('=', 80) . "\n";
    echo "✅ All tests completed successfully!\n";

    file_put_contents(
        'benchmark_results.json',
        json_encode($results, JSON_PRETTY_PRINT)
    );

} catch (Exception $e) {
    var_dump($e);
    exit(1);
}