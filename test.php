<?php

use Cuda\CudaArray;
use Cuda\Attr as Attr;

class UserKernel extends Cuda\Kernel
{
    #[Cuda\Attr\Kernel(name: 'fused_relu_scale', target: 'sm_60')]
    public function kernel_with_metadata(
        #[Cuda\Attr\Input(dtype: 'float')] array $a,
        #[Cuda\Attr\Input(dtype: 'float')] array $b,
        #[Cuda\Attr\Output(dtype: 'float')] array $c
    ): void {
        $idx = $this->threadIdx();
        $c[$idx] = $this->calculateMax($a[$idx], $b[$idx]) * 2.0;
    }

    #[Cuda\Attr\Device(name: 'calculate_max', target: 'sm_60')]
    private function calculateMax(
        #[Cuda\Attr\Input(dtype: 'float')] float $a,
        #[Cuda\Attr\Input(dtype: 'int')] float $b
    ): float {
        $idx = $this->threadIdx();
        for($i = 0; $i <= $idx; $i++){
            $a[$i] = $a[$i] * $i;
        }
        if ($a > 10) {
            return $this->max($a * $b, 0.0);
        }

        return $b;
    }
}


$userKernel = new UserKernel();

