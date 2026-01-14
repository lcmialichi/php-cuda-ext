<?php

declare(strict_types=1);

use Cuda\Compiler;
use Cuda\CudaArray;
use Cuda\Attr as K;

/**
 * Custom CUDA Kernels (JIT)
 * * This class defines the logic that will be compiled into native PTX.
 * Use PHP 8 Attributes to specify types for the JIT compiler.
 */
class KernelDefinitions
{

    #[Cuda\Attr\Kernel(name: 'v_scale')]
    public function scale(
        #[K\TensorType] &$data,
        #[K\IntType] $factor,
        #[K\IntType] $n
    ): void {
        /** @var \Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();

        if ($idx < $n) {
            $data[$idx] *= $factor;
        }
    }
}

// --- Compilation Logic ---

$defs = new KernelDefinitions();
$compiler = new Compiler();
$compiler->kernel([$defs, 'scale']);

// JIT: Translates PHP logic -> PTX -> GPU Module
$module = $compiler->compile();
$module->initialize();

// --- Data & Execution ---

$size = 1_000_000;
$gpuData = CudaArray::ones([$size]);

// Define Parallelism Geometry (Grid/Block)
$launchConfig = [
    'block' => [256, 1, 1],
    'grid' => [(int) ceil($size / 256), 1, 1]
];

// Launch the custom kernel
$module->run('v_scale', args: [$gpuData, 10, $size], config: $launchConfig);

var_dump($gpuData->toArray()[0]); // Expected: 10.0