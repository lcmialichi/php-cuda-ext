<?php

use Cuda\Attr as K;
use Cuda\Compiler;
use Cuda\CudaArray;

class AdvancedKernels
{
    #[Cuda\Attr\Kernel(name: 'matrix_multiply_reduce')]
    public function matrixMultiplyReduce(
        #[K\TensorType] array &$A,
        #[K\TensorType] array &$B,
        #[K\TensorType] array &$C,
        #[K\TensorType] array &$reduction_result,
        #[K\IntType] $rows,
        #[K\IntType] $cols,
        #[K\IntType] $inner_dim,
        #[K\FloatType(bits: 32)] $scale_factor
    ): void {
        /** @var \Cuda\Runtime $cuda */
        $block_size = 16;

        $cuda->__declare_shared($tile_A, 'float32', 272);
        $cuda->__declare_shared($tile_B, 'float32', 272);

        $thread_x = $cuda->threadIdx()->x;
        $thread_y = $cuda->threadIdx()->y;
        $block_x = $cuda->blockIdx()->x;
        $block_y = $cuda->blockIdx()->y;

        $row = $block_y * $block_size + $thread_y;
        $col = $block_x * $block_size + $thread_x;

        $idx = $thread_y * ($block_size + 1) + $thread_x;
        $acc = 0.0;
        for ($tile = 0; $tile < $cuda->math->ceil($inner_dim / $block_size); $tile++) {
            $a_row = $row;
            $a_col = $tile * $block_size + $thread_x;
            $tile_A[$idx] = $a_row < $rows && $a_col < $inner_dim ?  $A[$a_row * $inner_dim + $a_col] : 0.0;

            $b_row = $tile * $block_size + $thread_y;
            $b_col = $col;
            $tile_B[$idx] =
                $b_row < $inner_dim && $b_col < $cols
                ? $B[$b_row * $cols + $b_col]
                : 0.0;

            $cuda->sync->threads();
            for ($k = 0; $k < $block_size; $k++) {
                $acc += $tile_A[$thread_y * ($block_size + 1) + $k] *
                    $tile_B[$k * ($block_size + 1) + $thread_x];
            }

            $cuda->sync->threads();
        }

        if ($row < $rows && $col < $cols) {
            $C[$row * $cols + $col] = $acc * $scale_factor;
        }

        $cuda->sync->threads();
        if ($row < $rows && $col < $cols) {
            $cuda->atomic->add($reduction_result[$row], $C[$row * $cols + $col]);
        }
    }

    #[Cuda\Attr\Kernel(name: 'atomic_sum')]
    public function atomicSum(
        #[K\TensorType] array &$a,
        #[K\TensorType] array &$b,
        #[K\IntType] int $n
    ): void {
        /** @var \Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();
        if ($idx < $n) {
            $cuda->atomic->add($a[$idx], $b[$idx]);
        }
    }
}

function benchmarkKernels()
{
    echo "=== BENCHMARK CUDA KERNELS ===\n\n";

    $rows = 1024;
    $inner_dim = 1024;
    $cols = 1024;
    $scale_factor = 2.0;
    $block_size = 16;

    $grid_x = (int) ceil($cols / $block_size);
    $grid_y = (int) ceil($rows / $block_size);

    $config = [
        'block' => [$block_size, $block_size, 1],
        'grid' => [$grid_x, $grid_y, 1]
    ];

    echo "Preparing data...\n";
    $A_host = [];
    $B_host = [];

    for ($i = 0; $i < $rows; $i++) {
        for ($j = 0; $j < $inner_dim; $j++) {
            $A_host[$i][$j] = ($i + $j * 0.1) * 0.01;
        }
    }

    for ($i = 0; $i < $inner_dim; $i++) {
        for ($j = 0; $j < $cols; $j++) {
            $B_host[$i][$j] = ($i * 0.1 + $j) * 0.01;
        }
    }

    $A = new CudaArray($A_host, dtype: 'float32');
    $B = new CudaArray($B_host, dtype: 'float32');
    $C = CudaArray::zeros([$rows, $cols], dtype: 'float32');

    $reduction_result = CudaArray::zeros([$rows], dtype: 'float32');

    $test = CudaArray::ones([$rows, $cols]);
    $test2 = CudaArray::full([$rows, $cols], 2);

    echo "Compiling kernels...\n";
    $compile_start = hrtime(true);
    $compiler = new Compiler(target: 'sm_75');
    $module = $compiler
        ->kernel([new AdvancedKernels(), 'matrixMultiplyReduce'])
        ->kernel([new AdvancedKernels(), 'atomicSum'])
        ->compile();
    $compile_end = hrtime(true);
    $compile_time = ($compile_end - $compile_start) / 1e6;

    echo "Initializing CUDA module...\n";
    $init_start = hrtime(true);
    $module->initialize();
    $init_end = hrtime(true);
    $init_time = ($init_end - $init_start) / 1e6;

    echo "\n=== EXECUTION TIMES ===\n";
    echo "\n1. SYNC (launch()):\n";
    echo str_repeat("-", 50) . "\n";

    $sync_start = hrtime(true);
    $success_sync = $module->launch(
        'matrix_multiply_reduce',
        config: $config,
        args: [$A, $B, $C, $reduction_result, $rows, $cols, $inner_dim, $scale_factor]
    );
    $sync_end = hrtime(true);
    $sync_time = ($sync_end - $sync_start) / 1e6;

    echo "matrix_multiply_reduce: " . number_format($sync_time, 3) . " ms\n";
    echo "Status: " . ($success_sync ? "SUCCESS" : "FAILED") . "\n";

    echo "\n2. ASYNC SINGLE (launchAsync()):\n";
    echo str_repeat("-", 50) . "\n";

    $async1_start = hrtime(true);
    $op_id1 = $module->launchAsync(
        'matrix_multiply_reduce',
        config: $config,
        args: [$A, $B, $C, $reduction_result, $rows, $cols, $inner_dim, $scale_factor]
    );
    $async1_end = hrtime(true);
    $async1_launch_time = ($async1_end - $async1_start) / 1e6;

    echo "launchAsync() call time: " . number_format($async1_launch_time, 3) . " ms\n";
    echo "Operation ID: " . $op_id1 . "\n";

    $n_op  = 100;

    $wait1_start = hrtime(true);
    $module->sync($op_id1);
    $wait1_end = hrtime(true);
    $wait1_time = ($wait1_end - $wait1_start) / 1e6;

    echo "sync() wait time: " . number_format($wait1_time, 3) . " ms\n";
    echo "Total async time: " . number_format($async1_launch_time + $wait1_time, 3) . " ms\n";

    echo "\n3. ASYNC MULTIPLE ({$n_op}x launchAsync()):\n";
    echo str_repeat("-", 50) . "\n";

    $async_multi_start = hrtime(true);

    for ($i = 0; $i < $n_op - 1; $i++) {
        $op_start = hrtime(true);
        $op_id = $module->launchAsync(
            'atomic_sum',
            config: $config,
            args: [$test, $test2, $test->getSize()]
        );
        $op_end = hrtime(true);
        $op_time = ($op_end - $op_start) / 1e6;
        echo "  atomic_sum #" . ($op_id) . ": " . number_format($op_time, 3) . " ms (ID: $op_id)\n";
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
            'kernel' =>  'atomic_sum',
            'args' =>  [$test, $test2, $test->getSize()],
            'config' => $config
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

    // Resumo
    echo "\n=== PERFORMANCE SUMMARY ===\n";
    echo str_repeat("=", 50) . "\n";

    $stats = $module->getStats();

    echo "Compilation time:     " . number_format($compile_time, 3) . " ms\n";
    echo "Initialization time:  " . number_format($init_time, 3) . " ms\n";
    echo "\n";
    echo "Synchronous launch:   " . number_format($sync_time, 3) . " ms\n";
    echo "Async launch (call):  " . number_format($async1_launch_time, 3) . " ms\n";
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
        $module->sync(); // Limpar tudo
    }

    echo "\n" . str_repeat("=", 50) . "\n";
    echo "Benchmark completed!\n";
}

try {
    benchmarkKernels();
} catch (Exception $e) {
    echo "\nERROR: " . $e->getMessage() . "\n";
}
