<?php

use Cuda\Attr as K;
use Cuda\Compiler;
use Cuda\CudaArray;

class AdvancedKernels
{
    #[Cuda\Attr\Kernel(name: 'sum')]
    public function sum(
        #[K\TensorType] array &$a,
        #[K\TensorType] array &$b,
        #[K\IntType] int $n
    ): void {
        /** @var \Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();
        if ($idx < $n) {
            $a[$idx] += $b[$idx] + 2 * 3.0 * 4 * 5;
        }
    }
}

function benchmarkKernels()
{
    echo "=== BENCHMARK CUDA KERNELS ===\n\n";

    $block_size = 16;

    $x = 512;
    $y = 512;
    $z = 3;

    $grid_x = (int) ceil($x / $block_size);
    $grid_y = (int) ceil($y / $block_size);

    $configDefault = [
        'block' => [$block_size, $block_size, 1],
        'grid' => [$grid_x, $grid_y, 1]
    ];

    echo "Preparing data...\n";

    $test = CudaArray::ones([$x, $y, $z]);
    $test2 = CudaArray::full([$x, $y, $z], 2);

    echo "Compiling kernels...\n";
    $compile_start = hrtime(true);
    $compiler = new Compiler();
    $module = $compiler
        ->kernel([new AdvancedKernels(), 'sum'])
        ->compile();
    $compile_end = hrtime(true);
    $compile_time = ($compile_end - $compile_start) / 1e6;

    echo "Initializing CUDA module...\n";
    $init_start = hrtime(true);
    $module->initialize();
    $init_end = hrtime(true);
    $init_time = ($init_end - $init_start) / 1e6;

    echo "\n=== EXECUTION TIMES ===\n";
    $n_op = 100;
    echo "\n3. ASYNC MULTIPLE ({$n_op}x launchAsync()):\n";
    echo str_repeat("-", 50) . "\n";

    $configSum = $module->autoGrid('sum', $test);
    $async_multi_start = hrtime(true);

    for ($i = 0; $i < $n_op - 1; $i++) {
        $op_start = hrtime(true);
        $op_id = $module->launchAsync(
            'sum',
            config: $configSum,
            args: [$test, $test2, $test->getSize()]
        );
        $op_end = hrtime(true);
        $op_time = ($op_end - $op_start) / 1e6;
        echo "  sum #" . ($op_id) . ": " . number_format($op_time, 3) . " ms (ID: $op_id)\n";
    }

    $async_multi_mid = hrtime(true);
    $launch_total_time = ($async_multi_mid - $async_multi_start) / 1e6;
    echo "  Total launch time ({$n_op} kernels): " . number_format($launch_total_time, 3) . " ms\n";
    echo "  Average per launch: " . number_format($launch_total_time / $n_op, 3) . " ms\n";

    $wait_multi_start = hrtime(true);
    $module->sync();
    $wait_multi_end = hrtime(true);
    $wait_multi_time = ($wait_multi_end - $wait_multi_start) / 1e6;

    $async_multi_total = ($wait_multi_end - $async_multi_start) / 1e6;

    echo "  sync() wait time: " . number_format($wait_multi_time, 3) . " ms\n";
    echo "  Total execution time: " . number_format($async_multi_total, 3) . " ms\n";

    echo "\n4. BATCH OPERATIONS (runAsyncBatch()):\n";
    echo str_repeat("-", 50) . "\n";

    $batch_ops = [];
    for ($i = 0; $i < $n_op; $i++) {
        $batch_ops[] = [
            'kernel' =>  'sum',
            'args' =>  [$test, $test2, $test->getSize()],
            'config' => $configSum
        ];
    }
    
    $batch_start = hrtime(true);
    $results = $module->launchAsyncBatch($batch_ops);
    $batch_end = hrtime(true);
    $batch_time = ($batch_end - $batch_start) / 1e6;

    echo "  {$n_op} operations in batch\n";
    echo "  Batch execution time: " . number_format($batch_time, 3) . " ms\n";
    echo "  Average per operation: " . number_format($batch_time / $n_op, 3) . " ms\n";

    $success_count = 0;
    foreach ($results as $i => $result) {
        if ($result) $success_count++;
    }
    echo "  Successful operations: $success_count/100\n";

    echo "\n=== PERFORMANCE SUMMARY ===\n";
    echo str_repeat("=", 50) . "\n";

    $stats = $module->getStats();

    echo "Compilation time:     " . number_format($compile_time, 3) . " ms\n";
    echo "Initialization time:  " . number_format($init_time, 3) . " ms\n";
    echo "\n";
    echo "Async launch (avg):   " . number_format($launch_total_time / $n_op, 3) . " ms ($n_op ops)\n";
    echo "Batch (avg/op):       " . number_format($batch_time / $n_op, 3) . " ms ($n_op ops)\n";
    echo "\n";
    echo "Total kernels executed: " . ($stats['kernel_execution_count'] ?? 0) . "\n";
    echo "Total exec time:        " . number_format($stats['total_execution_time_ms'] ?? 0, 3) . " ms\n";
    echo "Avg kernel time:        " . number_format($stats['avg_execution_time_ms'] ?? 0, 3) . " ms\n";

    $pending = $module->getPendingOperations();
    echo "Pending operations:    " . count($pending) . "\n";

    if (!empty($pending)) {
        echo "\nWarning: Some operations still pending!\n";
        $module->sync();
    }

    echo "\n" . str_repeat("=", 50) . "\n";
    echo "Benchmark completed!\n";
}

try {
    benchmarkKernels();
} catch (Exception $e) {
    echo "\nERROR: " . $e->getMessage() . "\n";
}
