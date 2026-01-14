<?php

use Cuda\Attr as K;
use Cuda\Compiler;
use Cuda\CudaArray;

class KernelDefs
{
    #[K\Kernel('multiply')]
    public function multiply(
        #[K\TensorType] &$input,
        #[K\IntType] $value,
        #[K\IntType] $n,
    ): void {
        /** @var \Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();
        if ($idx < $n) {
            $input[$idx] *= $value;
        }
    }
}
