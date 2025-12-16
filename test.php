<?php

use Cuda\CudaArray;
use Cuda\Attr as Attr;

use Cuda\Runtime;

$compiler = new Cuda\Compiler();

#[Attr\Kernel(name: 'convolution_3d_volume')]
function conv_3D(
    #[Attr\Input(dtype: 'float32')] array $volume,
    #[Attr\Input(dtype: 'float32')] array $kernel,
    #[Attr\Output(dtype: 'float32')] array &$output,
    #[Attr\Input(dtype: 'int32')] int $width,
    #[Attr\Input(dtype: 'int32')] int $height,
    #[Attr\Input(dtype: 'int32')] int $depth,
    #[Attr\Input(dtype: 'int32')] int $kernelSize
) {
    /** @var \Cuda\Runtime $cuda */
    $x = $cuda->blockIdx()->x * $cuda->blockDim()->x + $cuda->threadIdx()->x;
    $y = $cuda->blockIdx()->y * $cuda->blockDim()->y + $cuda->threadIdx()->y;
    $z = $cuda->blockIdx()->z * $cuda->blockDim()->z + $cuda->threadIdx()->z;

    $halfKernel = ($kernelSize - 1) / 2;

    $a = $x >= $halfKernel && $x < $width - $halfKernel;
    $b = $y >= $halfKernel && $y < $height - $halfKernel;
    $c = $z >= $halfKernel && $z < $depth - $halfKernel;
 
    if ($a && $b && $c) {
        $sum = 0.0;

        for ($kz = 0; $kz < $kernelSize; $kz++) {
            for ($ky = 0; $ky < $kernelSize; $ky++) {
                for ($kx = 0; $kx < $kernelSize; $kx++) {
                    $voxelX = $x + $kx - $halfKernel;
                    $voxelY = $y + $ky - $halfKernel;
                    $voxelZ = $z + $kz - $halfKernel;

                    $volumeIdx = ($voxelZ * $height + $voxelY) * $width + $voxelX;
                    $kernelIdx = ($kz * $kernelSize + $ky) * $kernelSize + $kx;

                    $sum += $volume[$volumeIdx] * $kernel[$kernelIdx];
                }
            }
        }

        $outputIdx = ($z * $height + $y) * $width + $x;
        $output[$outputIdx] = $sum;
    }
}

$compiler->kernel(conv_3D(...));

$compiled = $compiler->compile(debug: true);

var_dump($compiled);