<?php

use Cuda\CudaArray;
use Cuda\Attr as Attr;

class UserKernel extends Cuda\Kernel
{
    #[Cuda\Attr\Kernel(name: 'fused_relu_scale', target: 'sm_60')]
    public function kernel_with_metadata(
        #[Attr\Input(dtype: 'float')] array $a,
        #[Attr\Input(dtype: 'float')] array $b,
        #[Attr\Output(dtype: 'float')] array $c
    ): void {
        $idx = $this->threadIdx();
        $c[$idx] = $this->calculateMax($a[$idx], $b[$idx]) * 2.0;
    }

    #[Cuda\Attr\Device(name: 'calculate_max', target: 'sm_60')]
    private function calculateMax(
        #[Attr\Input(dtype: 'float')] float $a,
        #[Attr\Input(dtype: 'int')] float $b
    ): float {

        if ($a > 10) {
            return max($a * $b, 0.0);
        }

        return $b;
    }
}

$userKernel = new UserKernel(); // aqui compila

