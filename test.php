<?php

use Cuda\CudaArray;
use Cuda\Attr as Attr;

use Cuda\Runtime;

$compiler = new Cuda\Compiler();

$compiler->kernel(
    #[Attr\Kernel(name: 'vector_reduction')]
    function (
        #[Attr\Input(dtype: 'float')] array $input,
        #[Attr\Output(dtype: 'float')] array $partial_sums,
        #[Attr\Input(dtype: 'int')] int $size
    ): void {
        /** @var \Cuda\Runtime $cuda */
        $threadId = $cuda->threadIdx();
        $blockId = $cuda->blockIdx();
        $blockDim = $cuda->blockDim();
        
        $globalId = $threadId + $blockId * $blockDim;
        $stride = $blockDim * $cuda->gridDim();
        
        $shared = 0.0;

        $shared[$threadId] = 0.0;
                
        $sum = 0.0;
        for ($i = $globalId; $i < $size; $i += $stride) {
            $sum += $input[$i];
        }
        $shared[$threadId] = $sum;
        
        for ($s = $blockDim / 2; $s > 0; $s >>= 1) {
            if ($threadId < $s) {
                $shared[$threadId] += $shared[$threadId + $s];
            }
        }
        
        if ($threadId == 0) {
            $partial_sums[$blockId] = $shared[0];
        }
    }
);

var_dump($compiler->getKernels()['vector_reduction']['cuda_code']);