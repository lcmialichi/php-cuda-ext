<?php

use Cuda\CudaArray;
use Cuda\Attr as Attr;

use Cuda\Runtime;

$compiler = new Cuda\Compiler();

$compiler->kernel(
    #[Attr\Kernel(name: 'convolution_2d_optimized')]
    function(
        #[Attr\Input(dtype:'float32')]array $voxels, 
        #[Attr\Output(dtype:'float32')] array &$lighting, 
        #[Attr\Input(dtype:'int32')]int $size) {

    /** @var \Cuda\Runtime $cuda */
    $x = $cuda->threadIdx()->x + $cuda->blockIdx()->x * $cuda->blockDim()->x;
    $y = $cuda->threadIdx()->y + $cuda->blockIdx()->y * $cuda->blockDim()->y;
    $z = $cuda->threadIdx()->z + $cuda->blockIdx()->z * $cuda->blockDim()->z;
    
    if ($x < $size && $y < $size && $z < $size) {
        $idx = ($z * $size + $y) * $size + $x;
        
        $height = $cuda->math->sin($x * 0.1) * $cuda->math->cos($z * 0.1) * 10 + $size/2;
        
        if ($y < $height) {
            $voxels[$idx] = 1;
            
            $light = 1.0;
            if ($y > 0 && $voxels[($z * $size + ($y-1)) * $size + $x] == 0) {
                $light = 0.7;
            }
            $lighting[$idx] = $light;
        } else {
            $voxels[$idx] = 0;
            $lighting[$idx] = 1.0;
        }
    }
});

var_dump($compiler->getKernels()['convolution_2d_optimized']['cuda_code']);