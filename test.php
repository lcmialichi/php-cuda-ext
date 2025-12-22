<?php

use Cuda\CudaArray;
use Cuda\Attr as Attr;
use Cuda\CompiledModule;

define('THREADS', 256);
define('MODULE_FILE', 'serialized_module.cudas');

function main(): void
{
    if (!file_exists(MODULE_FILE)) {
        file_put_contents(MODULE_FILE, serialize(createCudaModule()));
    }

    $module = unserialize(file_get_contents(MODULE_FILE));

    $input = CudaArray::rand([512, 512, 16], -1, 1);
    $output = CudaArray::zeros([512, 512, 16]);
    $scale = 1;
    $threshold = 0;

    [$xo, $yo, $zo] = $output->getShape();

    $ototal = ($xo * $yo * $zo);

    $module
        ->runAsync(
            name: 'element_wise',
            args: [$input, $output, $threshold, $scale, $ototal],
            config: [
                'block' => [THREADS, 1, 1],
                'grid' => grid($ototal)
            ]
        );

    while (!$module->isFinished()) {
        $module->sync();
        echo "Running...\n";
    }

    echo "Finished...\n";
}

function grid(int $n): array
{
    $blocks_needed = (int) ceil($n / THREADS);
    $max_blocks = 65535;

    if ($blocks_needed <= $max_blocks) {
        return [$blocks_needed, 1, 1];
    }

    $blocks_x = min($max_blocks, $blocks_needed);
    $blocks_y = (int) ceil($blocks_needed / $max_blocks);

    return [$blocks_x, $blocks_y, 1];
}


function createCudaModule(): CompiledModule
{
    $compiler = new Cuda\Compiler();

    #[Attr\Kernel(name: 'element_wise')]
    function element_wise(
        #[Attr\TensorType] array $input,
        #[Attr\TensorType] array &$output,
        #[Attr\FloatType] float $threshold,
        #[Attr\FloatType] float $scale_factor,
        #[Attr\IntType] int $total_elements
    ): void {
        /** @var \Cuda\Runtime $cuda */
        $tid = $cuda->threadIdx()->x;
        $bid = $cuda->blockIdx()->x;
        $bdim = $cuda->blockDim()->x;
        $gid = $bid * $bdim + $tid;

        if ($gid >= $total_elements) {
            return;
        }

        $output[$gid] = $scale_factor * $input[$gid];
    }


    $compiler->kernel('element_wise');
    return $compiler->compile();
}

main();