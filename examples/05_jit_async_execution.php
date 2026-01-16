<?php

declare(strict_types=1);

use Cuda\Compiler;
use Cuda\CudaArray;
use Cuda\Attr as K;

/**
 * Asynchronous Kernel Execution
 * * Demonstrates how to launch GPU tasks without blocking the main PHP thread.
 * This allows for massive parallelism between the CPU and GPU.
 */

class HeavyWorkload
{
    /**
     * A computationally expensive kernel simulation
     */
    #[K\Kernel(name: 'heavy_math')]
    public function compute(#[K\TensorType] &$data, #[K\IntType] $rows, #[K\IntType] $cols): void
    {
        /** @var \Cuda\Runtime $cuda */
        $col = $cuda->blockIdx()->x * $cuda->blockDim()->x + $cuda->threadIdx()->x;
        $row = $cuda->blockIdx()->y * $cuda->blockDim()->y + $cuda->threadIdx()->y;

        if ($row >= $rows || $col >= $cols) {
            return;
        }

        $idx = $row * $cols + $col;
        $val = $data[$idx];
        for ($i = 0; $i < 100; $i++) {
            $val = $cuda->math->sin($val) * $cuda->math->cos($val);
        }

        $data[$idx] = $val;
    }
}

// --- 1. JIT Compilation ---

$compiler = new Compiler();
$compiler->kernel([new HeavyWorkload(), 'compute']);
$module = $compiler->compile();
$module->initialize();

// --- 2. Data & Configuration ---

$size = 2_000_000;
$data = CudaArray::rand([32, $size], 0, 1);
[$rows, $cols] = $data->getShape();

$block = [32, 8, 1];
$config = [
    'block' => [256, 1, 1],
    'grid' => [(int) ceil($cols / $block[0]), (int) ceil($rows / $block[1]), 1]
];

// --- 3. Asynchronous Launch ---

/**
 * runAsync() returns an operation ID immediately.
 * The PHP engine does not wait for the GPU to finish.
 */
$opId = $module->runAsync('heavy_math', args: [$data, $rows, $cols], config: $config);

// --- 4. Concurrent CPU Processing ---

/**
 * While the GPU is crunching numbers, PHP is free to perform other tasks.
 * This is perfect for I/O bound operations or preparing other datasets.
 */
echo "GPU is processing heavy math in the background...\n";

while (!$module->isFinished($opId)) {
    // Perform some CPU work here
    usleep(1000); // Simulate other logic
    // echo "PHP is still free to run other code...\n";

    // Optional: Check status
    $status = $module->getAsyncStatus($opId);
}

// --- 5. Final Synchronization ---

/**
 * Ensures all pending GPU operations are complete before moving forward.
 * Required before calling toArray() to ensure data integrity.
 */
$module->sync();
$result = $data->toArray();
echo "Computation finished successfully.\n";