<?php

use Cuda\Attr as Attr;

#[Attr\Kernel(name: 'vector_add')]
function vectorAdd(
    #[Attr\TensorType] array $a,
    #[Attr\TensorType] array $b,
    #[Attr\TensorType] array &$c,
    #[Attr\IntType] int $n
): void {
    /** @var \Cuda\Runtime $cuda */

    foreach($a as $b => $c){
        
    }
    $idx = $cuda->blockIdx()->x * $cuda->blockDim()->x + $cuda->threadIdx()->x;
    if ($idx < $n) {
        $c[$idx] = $a[$idx] + $b[$idx];
    }
}

#[Attr\Kernel(name: 'element_wise_math')]
function elementWiseMath(
    #[Attr\TensorType] array $input,
    #[Attr\TensorType] array &$output,
    #[Attr\FloatType] float $factor,
    #[Attr\IntType] int $n
): void {
    /** @var \Cuda\Runtime $cuda */
    $idx = $cuda->blockIdx()->x * $cuda->blockDim()->x + $cuda->threadIdx()->x;
    if ($idx < $n) {
        $val = $input[$idx];
        $output[$idx] = ($val * $factor) + ($val / $factor) - ($val * $val);
    }
}

#[Attr\Kernel(name: 'matrix_multiply')]
function matrixMultiply(
    #[Attr\TensorType] array $a,
    #[Attr\TensorType] array $b,
    #[Attr\TensorType] array &$c,
    #[Attr\IntType] int $n,
    #[Attr\IntType] int $m,
    #[Attr\IntType] int $p
): void {
    /** @var \Cuda\Runtime $cuda */
    $row = $cuda->blockIdx()->y * $cuda->blockDim()->y + $cuda->threadIdx()->y;
    $col = $cuda->blockIdx()->x * $cuda->blockDim()->x + $cuda->threadIdx()->x;

    if ($row < $n && $col < $p) {
        $sum = 0.0;
        for ($k = 0; $k < $m; $k++) {
            $sum += $a[$row * $m + $k] * $b[$k * $p + $col];
        }
        $c[$row * $p + $col] = $sum;
    }
}

#[Attr\Kernel(name: 'complex_math')]
function complexMath(
    #[Attr\TensorType] array $input,
    #[Attr\TensorType] array &$output,
    #[Attr\FloatType] float $scale,
    #[Attr\IntType] int $n
): void {
    /** @var \Cuda\Runtime $cuda */
    $idx = $cuda->blockIdx()->x * $cuda->blockDim()->x + $cuda->threadIdx()->x;
    if ($idx < $n) {
        $x = $input[$idx];
        $output[$idx] = $cuda->math->sin($x * $scale) * $cuda->math->cos($x / $scale) +
            $cuda->math->exp(-0.1 * $x) - $cuda->math->log(1.0 + $cuda->math->abs($x));
    }
}


#[Attr\Kernel(name: 'reduce_sum')]
function reduceSum(
    #[Attr\TensorType] array $input,
    #[Attr\TensorType] array &$partial_sums,
    #[Attr\IntType] int $n
): void {
    /** @var \Cuda\Runtime $cuda */
    $global_tid = $cuda->threadIdx()->x + $cuda->blockIdx()->x * $cuda->blockDim()->x;
    $local_tid = $cuda->threadIdx()->x;

    $cuda->__declare_shared($shared, 'float32', 512);

    $shared[$local_tid] = ($global_tid < $n) ? $input[$global_tid] : 0.0;
    $cuda->sync->threads();

    for ($stride = $cuda->blockDim()->x / 2; $stride > 0; $stride >>= 1) {
        if ($local_tid < $stride) {
            $shared[$local_tid] += $shared[$local_tid + $stride];
        }
        $cuda->sync->threads();
    }

    if ($local_tid == 0) {
        $partial_sums[$cuda->blockIdx()->x] = $shared[0];
    }
}

#[Attr\Kernel(name: 'stencil_3pt')]
function stencil3Pt(
    #[Attr\TensorType] array $input,
    #[Attr\TensorType] array &$output,
    #[Attr\IntType] int $n
): void {
    /** @var \Cuda\Runtime $cuda */
    $idx = $cuda->blockIdx()->x * $cuda->blockDim()->x + $cuda->threadIdx()->x;

    if ($idx >= 1 && $idx < $n - 1) {
        $output[$idx] = ($input[$idx - 1] + $input[$idx] + $input[$idx + 1]) / 3.0;
    } elseif ($idx < $n) {
        $output[$idx] = $input[$idx];
    }
}

#[Attr\Kernel(name: 'saxpy')]
function saxpy(
    #[Attr\TensorType] array $x,
    #[Attr\TensorType] array $y,
    #[Attr\TensorType] array &$z,
    #[Attr\FloatType] float $alpha,
    #[Attr\IntType] int $n
): void {
    /** @var \Cuda\Runtime $cuda */
    $idx = $cuda->blockIdx()->x * $cuda->blockDim()->x + $cuda->threadIdx()->x;
    if ($idx < $n) {
        $z[$idx] = $alpha * $x[$idx] + $y[$idx];
    }
}

#[Attr\Kernel(name: 'conditional_ops')]
function conditionalOps(
    #[Attr\TensorType] array $input,
    #[Attr\TensorType] array &$output,
    #[Attr\FloatType] float $threshold,
    #[Attr\IntType] int $n
): void {
    /** @var \Cuda\Runtime $cuda */
    $idx = $cuda->blockIdx()->x * $cuda->blockDim()->x + $cuda->threadIdx()->x;
    if ($idx < $n) {
        $val = $input[$idx];

        if ($val > $threshold) {
            $output[$idx] = $cuda->math->sqrt($val) * 2.0;
        } elseif ($val > $threshold / 2) {
            $output[$idx] = $val * $val;
        } else {
            $output[$idx] = $cuda->math->sin($val) * $cuda->math->cos($val);
        }
    }
}

#[Attr\Kernel(name: 'mem_copy')]
function memCopy(
    #[Attr\TensorType] array $src,
    #[Attr\TensorType] array &$dst,
    #[Attr\IntType] int $n
): void {

    /** @var \Cuda\Runtime $cuda */
    
    $idx = $cuda->blockIdx()->x * $cuda->blockDim()->x + $cuda->threadIdx()->x;
    if ($idx < $n) {
        $dst[$idx] = $src[$idx];
    }
}

#[Attr\Kernel(name: 'mixed_operations')]
function mixedOperations(
    #[Attr\TensorType] array $a,
    #[Attr\TensorType] array $b,
    #[Attr\TensorType] array &$c,
    #[Attr\FloatType] float $param1,
    #[Attr\FloatType] float $param2,
    #[Attr\IntType] int $n
): void {
    /** @var \Cuda\Runtime $cuda */
    $idx = $cuda->blockIdx()->x * $cuda->blockDim()->x + $cuda->threadIdx()->x;
    if ($idx < $n) {
        $x = $a[$idx];
        $y = $b[$idx];

        $temp = $x * $param1 + $y / $param2;
        $c[$idx] = ($temp > 0) ?
            $cuda->math->log(1.0 + $temp * $temp) :
            $cuda->math->exp($temp) - 1.0;
    }
}

define('VECTOR_BLOCK', [256, 1, 1]);
define('MATRIX_BLOCK', [16, 16, 1]);
define('REDUCE_BLOCK', [256, 1, 1]);

function calculateGrid1D(int $n, array $block = VECTOR_BLOCK): array
{
    return [(int) ceil($n / $block[0]), 1, 1];
}

function calculateGrid2D(int $rows, int $cols, array $block = MATRIX_BLOCK): array
{
    return [
        (int) ceil($cols / $block[0]),
        (int) ceil($rows / $block[1]),
        1
    ];
}

function validateKernelConfig(string $kernel, array $block, array $grid): bool
{
    $limits = [
        'block_x' => 1024,
        'block_y' => 1024,
        'block_z' => 64,
        'grid_x' => 2147483647,
        'grid_y' => 65535,
        'grid_z' => 65535,
    ];

    if (
        $block[0] > $limits['block_x'] ||
        $block[1] > $limits['block_y'] ||
        $block[2] > $limits['block_z']
    ) {
        return false;
    }

    return true;
}
