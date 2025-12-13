<?php

use Cuda\CudaArray;
use Cuda\Attr as Attr;

use Cuda\Runtime;

$compiler = new Cuda\Compiler();

$compiler->kernel(
    #[Attr\Kernel(name: 'convolution_2d_optimized')]
    function (#[Attr\Input(dtype: 'float32')] array $input,
        #[Attr\Input(dtype: 'float32')] array $kernel,
        #[Attr\Output(dtype: 'float32')] array &$output,
        #[Attr\Input(dtype: 'int32')] int $width,
        #[Attr\Input(dtype: 'int32')] int $height,
        #[Attr\Input(dtype: 'int32')] int $kernelSize
    ): void {
        /** @var \Cuda\Runtime $cuda */

        // Coordenadas do thread
        $threadX = $cuda->threadIdx()->x;
        $threadY = $cuda->threadIdx()->y;
        $blockX = $cuda->blockIdx()->x;
        $blockY = $cuda->blockIdx()->y;

        // Tamanho do bloco (deve ser configurado como 16x16 ou 32x32)
        $blockWidth = $cuda->blockDim()->x;
        $blockHeight = $cuda->blockDim()->y;

        // Coordenadas globais da imagem
        $x = $blockX * $blockWidth + $threadX;
        $y = $blockY * $blockHeight + $threadY;

        // Tamanho do kernel e padding
        $halfKernel = ($kernelSize - 1) / 2;

        // Verifica se o thread está dentro dos limites válidos
        if (
            $x >= $halfKernel && $x < $width - $halfKernel &&
            $y >= $halfKernel && $y < $height - $halfKernel
        ) {

            // Inicializa acumulador
            $sum = 0.0;

            // Convolução 2D
            for ($ky = 0; $ky < $kernelSize; $ky++) {
                for ($kx = 0; $kx < $kernelSize; $kx++) {
                    // Coordenadas na imagem de entrada
                    $pixelX = $x + $kx - $halfKernel;
                    $pixelY = $y + $ky - $halfKernel;

                    // Índices lineares
                    $inputIdx = $pixelY * $width + $pixelX;
                    $kernelIdx = $ky * $kernelSize + $kx;

                    // Acumula produto
                    $sum += $input[$inputIdx] * $kernel[$kernelIdx];
                }
            }

            // Escreve resultado
            $outputIdx = $y * $width + $x;
            $output[$outputIdx] = $sum;
        }

        // Sincronização opcional (se houver shared memory depois)
        // $cuda->sync->threads();
    }
);

var_dump($compiler->getKernels()['convolution_2d_optimized']['cuda_code']);