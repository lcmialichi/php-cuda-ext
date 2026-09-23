<?php

namespace Benchmarks\Handlers;

use Cuda\CudaArray;
use Benchmarks\Handlers\Benchmark;
use Benchmarks\Support\Attr\InjectArgs;

class CudaArrayLinearAlgebraBenchmark extends Benchmark
{
    public function name(): string
    {
        return "CudaArray Linear Algebra & Multi-Tensor Operations Benchmark";
    }

    public function register(): array
    {
        return [
            [
                "run" => 8,
                "warmup" => true,
                "name" => "CudaArray::matmul()",
                "iterations" => 50,
                "type" => "CUDA",
                "handler" => "cudaArrayMatmul",
                "metadata" => [
                    ["shape" => "16x16 x 16x16", "type" => " 2D"],
                    ["shape" => " 128x128 x 128x128", "type" => " 2D"],
                    ["shape" => " 32x256x256 x 32x256x256", "type" => " 3D"],
                    ["shape" => " 1x512x512 x 64x512x512", "type" => " 3D (broadcast)"],
                    ["shape" => " 1024x768 x 768x512", "type" => " 2D"],
                    ["shape" => " 64x1024x512 x 64x512x256", "type" => " 3D"],
                    ["shape" => " 1x1000000 x 1000000x1", "type" => " 2D (extreme)"],
                    ["shape" => " 8x64x256x256 x 8x64x256x256", "type" => " 4D"],
                ]
            ],

        ];
    }

    public function description(): string
    {
        return "CudaArray linear algebra & multi-tensor operations methods Benchmark";
    }

    public function argsMatmul(int $count): array
    {
        return match ($count) {
            1 => [CudaArray::rand([16, 16]), CudaArray::rand([16, 16])],
            2 => [CudaArray::rand([128, 128]), CudaArray::rand([128, 128])],
            3 => [CudaArray::rand([32, 256, 256]), CudaArray::rand([32, 256, 256])],
            4 => [CudaArray::rand([1, 512, 512]), CudaArray::rand([64, 512, 512])],
            5 => [CudaArray::rand([1024, 768]), CudaArray::rand([768, 512])],
            6 => [CudaArray::rand([64, 1024, 512]), CudaArray::rand([64, 512, 256])],
            7 => [CudaArray::rand([1, 1000000]), CudaArray::rand([1000000, 1])],
            8 => [CudaArray::rand([8, 64, 256, 256]), CudaArray::rand([8, 64, 256, 256])],
        };
    }

    #[InjectArgs("argsMatmul")]
    public function cudaArrayMatmul(CudaArray $first, CudaArray $second): void
    {
        $first->matmul($second);
    }
}
