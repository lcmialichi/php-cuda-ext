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

        $acc = 0.0;
        for ($tile = 0; $tile < $cuda->math->ceil($inner_dim / $block_size); $tile++) {
            $a_row = $row;
            $a_col = $tile * $block_size + $thread_x;
            if ($a_row < $rows && $a_col < $inner_dim) {
                $tile_A[$thread_y * ($block_size + 1) + $thread_x] = $A[$a_row * $inner_dim + $a_col];
            } else {
                $tile_A[$thread_y * ($block_size + 1) + $thread_x] = 0.0;
            }

            $b_row = $tile * $block_size + $thread_y;
            $b_col = $col;

            $tile_B[$thread_y * ($block_size + 1) + $thread_x] =
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

function testMatrixMultiplyReduce()
{
    echo "=== Testing matrixMultiplyReduce ===\n";

    $rows = 128;
    $inner_dim = 256;
    $cols = 64;
    $scale_factor = 2.0;

    echo "Matrix A: {$rows} x {$inner_dim}\n";
    echo "Matrix B: {$inner_dim} x {$cols}\n";
    echo "Matrix C: {$rows} x {$cols}\n";

    $A_host = [];
    $B_host = [];

    for ($i = 0; $i < $rows; $i++) {
        for ($j = 0; $j < $inner_dim; $j++) {
            $A_host[] = ($i + $j * 0.1) * 0.01;
        }
    }

    for ($i = 0; $i < $inner_dim; $i++) {
        for ($j = 0; $j < $cols; $j++) {
            $B_host[] = ($i * 0.1 + $j) * 0.01;
        }
    }

    $A = new CudaArray($A_host, dtype: 'float32');
    $B = new CudaArray($B_host, dtype: 'float32');
    $C = CudaArray::zeros([$rows, $cols], dtype: 'float32');
    $reduction_result = CudaArray::zeros([$rows], dtype: 'float32');
    echo "CUDA arrays created\n";

    $compiler = new Compiler(target: 'sm_75');
    echo "Compiling kernel...\n";

    $compile_start = hrtime(true);
    $module = $compiler
        ->kernel([new AdvancedKernels(), 'matrixMultiplyReduce'])
        ->kernel([new AdvancedKernels(), 'atomicSum'])
        ->compile();
    $compile_end = hrtime(true);
    echo "Kernel compiled in: " . (($compile_end - $compile_start) / 1e6) . " ms\n";

    $block_size = 16;
    $grid_x = (int) ceil($cols / $block_size);
    $grid_y = (int) ceil($rows / $block_size);

    echo "Initializing CUDA module...\n";
    $init_start = hrtime(true);
    $module->initialize();
    $init_end = hrtime(true);
    echo "Module initialized in: " . (($init_end - $init_start) / 1e6) . " ms\n";

    echo "Executing kernel matrixMultiplyReduce...\n";
    $kernel_start = hrtime(true);

    $success = $module->launchAsync(
        'matrix_multiply_reduce',
        config: [
            'block' => [$block_size, $block_size, 1],
            'grid' => [$grid_x, $grid_y, 1]
        ],
        args: [$A, $B, $C, $reduction_result, $rows, $cols, $inner_dim, $scale_factor]
    );
    $kernel_end = hrtime(true);

    $test = CudaArray::ones([$rows, $cols]);
    $test2 = CudaArray::full([$rows, $cols], 2);

    $kernel_atomic_start = hrtime(true);
    $module->launchAsync(
        'atomic_sum',
        config: [
            'block' => [$block_size, $block_size, 1],
            'grid' => [$grid_x, $grid_y, 1]
        ],
        args: [$test, $test2, $test->getSize()]
    );

    $kernel_atomic_end = hrtime(true);

    $kernel_wait_start = hrtime(true);
    $module->sync();
    $kernel_wait_end = hrtime(true);


    if ($success) {
        echo "Kernel executed successfully!\n";
        echo "Kernel execution time: " . (($kernel_end - $kernel_start) / 1e6) . " ms\n";
        echo "Kernel atomic execution time: " . (($kernel_atomic_end - $kernel_atomic_start) / 1e6) . " ms\n";
        echo "Kernel wait time: " . (($kernel_wait_end - $kernel_wait_start) / 1e6) . " ms\n";

        echo "\n=== Checking results ===\n";

        echo "DEBUG: Checking C type...\n";
        var_dump($C->dtype());

        $C_data = $C->flatten()->toArray();
        $reduction_data = $reduction_result->toArray();

        echo "DEBUG: C data type: " . gettype($C_data) . "\n";
        if (is_array($C_data)) {
            echo "DEBUG: C data is array with " . count($C_data) . " elements\n";
            echo "DEBUG: First 3 elements: " . $C_data[0] . ", " . $C_data[1] . ", " . $C_data[2] . "\n";
        }

        echo "Calculating expected result on CPU...\n";
        $C_expected = array_fill(0, $rows * $cols, 0.0);
        $reduction_expected = array_fill(0, $rows, 0.0);

        $cpu_start = hrtime(true);
        for ($i = 0; $i < $rows; $i++) {
            $row_sum = 0.0;
            for ($j = 0; $j < $cols; $j++) {
                $sum = 0.0;
                for ($k = 0; $k < $inner_dim; $k++) {
                    $a_idx = $i * $inner_dim + $k;
                    $b_idx = $k * $cols + $j;
                    $sum += $A_host[$a_idx] * $B_host[$b_idx];
                }
                $C_expected[$i * $cols + $j] = $sum * $scale_factor;
                $row_sum += $C_expected[$i * $cols + $j];
            }
            $reduction_expected[$i] = $row_sum;
        }
        $cpu_end = hrtime(true);

        echo "CPU calculation completed in: " . (($cpu_end - $cpu_start) / 1e6) . " ms\n";
        if (is_array($C_data)) {
            $errors = [];
            for ($i = 0; $i < min(10, count($C_data)); $i++) {
                $expected = $C_expected[$i];
                $actual = $C_data[$i];
                $error = abs($expected - $actual);
                $errors[] = $error;

                echo "Element [{$i}]: expected={$expected}, actual={$actual}, error={$error}\n";
            }

            $avg_error = array_sum($errors) / count($errors);
            echo "\nAverage error (first 10 elements): " . $avg_error . "\n";
        } else {
            echo "ERROR: C data is not an array\n";
        }

        if (is_array($reduction_data)) {
            echo "\nReduction results (first 5):\n";
            for ($i = 0; $i < min(5, count($reduction_data)); $i++) {
                $expected = $reduction_expected[$i];
                $actual = $reduction_data[$i];
                $error = abs($expected - $actual);
                echo "Row [{$i}]: expected={$expected}, actual={$actual}, error={$error}\n";
            }
        }
    } else {
        echo "Failed to execute kernel!\n";
    }

    echo "\n=== Module statistics ===\n";
    $stats = $module->getStats();
    print_r([
        'kernel_execution_count' => $stats['kernel_execution_count'],
        'total_execution_time_ms' => $stats['total_execution_time_ms'],
        'avg_execution_time_ms' => $stats['avg_execution_time_ms'] ?? 0
    ]);
}

try {
    testMatrixMultiplyReduce();
} catch (Exception $e) {
    echo "\nERROR: " . $e->getMessage() . "\n";
    echo "Trace: " . $e->getTraceAsString() . "\n";
}
