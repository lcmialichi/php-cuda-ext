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

    /**
     * @var CompiledModule
     */
    $module = unserialize(file_get_contents(MODULE_FILE));

    $input1 = CudaArray::rand([512, 512, 16], -1, 1);
    $input2 = CudaArray::rand([512, 512, 16], -1, 1);
    $input3 = CudaArray::rand([512, 512, 16], -1, 1);
    $input4 = CudaArray::rand([512, 512, 16], -1, 1);
    $input5 = CudaArray::rand([512, 512, 16], -1, 1);
    $input6 = CudaArray::rand([512, 512, 16], -1, 1);
    $input7 = CudaArray::rand([512, 512, 16], -1, 1);
    $output1 = CudaArray::zeros([512, 512, 16]);
    $output2 = CudaArray::zeros([512, 512, 16]);
    $output3 = CudaArray::zeros([512, 512, 16]);
    $output4 = CudaArray::zeros([512, 512, 16]);
    $output5 = CudaArray::zeros([512, 512, 16]);
    $output6 = CudaArray::zeros([512, 512, 16]);
    $output7 = CudaArray::zeros([512, 512, 16]);
    $scale = 100;

    [$x, $y, $z] = $output1->getShape();
    $total = $x * $y * $z;

    $module
        ->runAsync(
            name: 'element_wise',
            args: [$input1, $output1, $scale, $total],
            config: [
                'block' => [THREADS, 1, 1],
                'grid' => grid($total)
            ]
        );

    $module
        ->runAsync(
            name: 'element_wise',
            args: [$input2, $output2, $scale, $total],
            config: [
                'block' => [THREADS, 1, 1],
                'grid' => grid($total)
            ]
        );

    $module
        ->runAsync(
            name: 'element_wise',
            args: [$input3, $output3, $scale, $total],
            config: [
                'block' => [THREADS, 1, 1],
                'grid' => grid($total)
            ]
        );

      $module
        ->runAsync(
            name: 'element_wise',
            args: [$input4, $output4, $scale, $total],
            config: [
                'block' => [THREADS, 1, 1],
                'grid' => grid($total)
            ]
        );

         $module
        ->runAsync(
            name: 'element_wise',
            args: [$input5, $output5, $scale, $total],
            config: [
                'block' => [THREADS, 1, 1],
                'grid' => grid($total)
            ]
        );

         $module
        ->runAsync(
            name: 'element_wise',
            args: [$input6, $output6, $scale, $total],
            config: [
                'block' => [THREADS, 1, 1],
                'grid' => grid($total)
            ]
        );

         $module
        ->runAsync(
            name: 'element_wise',
            args: [$input7, $output7, $scale, $total],
            config: [
                'block' => [THREADS, 1, 1],
                'grid' => grid($total)
            ]
        );

    $count = 0;
    $time = microtime(true);
    while (!$module->isFinished()) {

        $currentTime = round(microtime(true) - $time, 3) * 1000;
        $ops = count($module->getPendingOperations());
        if ($ops != $count) {
            $count = $ops;
            echo "OPs: {$count} at time {$currentTime} ms\n";

        }
    }
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

    #[Attr\Kernel(name: 'elementWise')]
    function elementWise(
        #[Attr\TensorType] array $input,
        #[Attr\TensorType] array &$output,
        #[Attr\FloatType] float $scalar,
        #[Attr\IntType] int $total_elements
    ): void {
        /** @var \Cuda\Runtime $cuda */
        $idx = $cuda->threadIdx()->x * $cuda->blockIdx()->x + $cuda->blockDim()->x;

        if ($idx >= $total_elements) {
            return;
        }

        $output[$idx] = $input[$idx] * $scalar;
    }


    $compiler->kernel('elementWise');
    return $compiler->compile();
}

main();
