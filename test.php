<?php

use Cuda\CudaArray;
use Cuda\Attr as Attr;

use Cuda\Runtime;
use function Cuda\Builtins\{runtime};

$compiler = new Cuda\Compiler;


$compiler->kernel(

    #[Cuda\Attr\Kernel(name: 'kernel_test')]
    function (
        #[Cuda\Attr\Input(dtype: 'float')]array $a,
        #[Cuda\Attr\Output(dtype: 'float')]array $b
        ): void {
        /** 
         * @var Runtime $cuda 
         */

        $idx = 1;
        $test = $b[$idx] * $a[$idx];

    }
);

var_dump($compiler->getKernels()['kernel_test']['cuda_code']);