<?php

use Cuda\CudaArray;
use Cuda\Attr as Attr;

define('THREADS', 256);

function grid(int $n)
{
    return ceil($n / THREADS);
}

$compiler = new Cuda\Compiler();

#[Attr\Kernel(name: 'el_wise')]
function el_wise_kernel(
    #[Attr\Input(dtype: 'float32')] array $input,
    #[Attr\Output(dtype: 'float32')] array &$output,
    #[Attr\Input(dtype: 'float32')] float $value,
    #[Attr\Input(dtype: 'int32')] int $total_elements,
): void {
    /** @var \Cuda\Runtime $cuda */
    $idx = $cuda->blockIdx()->x * $cuda->blockDim()->x + $cuda->threadIdx()->x;
    if ($idx >= $total_elements) {
        return;
    }

    $output[$idx] = (int)($idx * $value);
}

$input = CudaArray::ones([16, 16, 16]);
$output = CudaArray::zeros([16, 16, 16]);
$value = 1.0;

[$x, $y, $z] = $output->getShape();

$total = $x * $y * $z;

$compiler->kernel('el_wise_kernel')
    ->compile()
    ->run(
        name: 'el_wise',
        config: [
            'block' => [THREADS, 1, 1],
            'grid' => [grid($total), 1, 1]
        ],
        args: [$input, $output, $value, $total]
    );

var_dump($output[0]->toArray());

