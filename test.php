<?php

use Cuda\CudaArray;
use Cuda\Attr as Attr;

define('THREADS', 256);

function grid(int $n)
{
    return ceil($n / THREADS);
}

$compiler = new Cuda\Compiler();

#[Attr\Kernel(name: 'idx_val')]
function idx_val(
    #[Attr\TensorType] array &$output,
    #[Attr\IntType] int $total_elements,
): void {
    /** @var \Cuda\Runtime $cuda */
    $idx = $cuda->blockIdx()->x * $cuda->blockDim()->x + $cuda->threadIdx()->x;
    if ($idx >= $total_elements) {
        return;
    }

    $output[$idx] = $idx;
}


#[Attr\Kernel(name: 'el_wise')]
function el_wise_kernel(
    #[Attr\TensorType] array $input,
    #[Attr\TensorType] array &$output,
    #[Attr\FloatType] float $value,
    #[Attr\IntType] int $total_elements,
): void {
    /** @var \Cuda\Runtime $cuda */
    $idx = $cuda->blockIdx()->x * $cuda->blockDim()->x + $cuda->threadIdx()->x;
    if ($idx >= $total_elements) {
        return;
    }

    $nextIdx = $cuda->math->ceil($idx * 2);
    for ($i = 0; $i <= $nextIdx; $i++) {
        $output[$idx] = $input[$nextIdx];
    }

}

$input = CudaArray::ones([16, 16, 16]);
$output = CudaArray::zeros([8, 8, 8]);
$value = 1.0;

[$xo, $yo, $zo] = $output->getShape();
[$xi, $yi, $zi] = $input->getShape();

$ototal = ($xo * $yo * $zo);
$itotal = ($xi * $yi * $zi);

$compiler->kernel('el_wise_kernel');
$compiler->kernel('idx_val');

$module = $compiler->compile();

$module->run(
    name: 'idx_val',
    args: [$input, $itotal],
    config: [
        'block' => [THREADS, 1, 1],
        'grid' => [grid($itotal), 1, 1]
    ]
);

$module
    ->run(
        name: 'el_wise',
        args: [$input, $output, $value, $ototal],
        config: [
            'block' => [THREADS, 1, 1],
            'grid' => [grid($ototal), 1, 1]
        ]
    );


var_dump($output[0]->toArray());