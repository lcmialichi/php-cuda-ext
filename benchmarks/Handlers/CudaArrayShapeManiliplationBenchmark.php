<?php

namespace Benchmarks\Handlers;

use Cuda\CudaArray;
use Benchmarks\Handlers\Benchmark;
use Benchmarks\Support\Attr\InjectArgs;

class CudaArrayShapeManiliplationBenchmark extends Benchmark
{
    public function name(): string
    {
        return "CudaArray Shape & Manipulation Benchmark";
    }

    public function register(): array
    {
        return [
            [
                "run" => 7,
                "warmup" => true,
                "name" => "CudaArray::flatten()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayFlatten",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 12,
                "warmup" => true,
                "name" => "CudaArray::transpose()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayTranspose",
                "metadata" => [
                    ["shape" => "16x16", "idxs" => "[1, 0]"],
                    ["shape" => "32x64", "idxs" => "[1, 0]"],
                    ["shape" => "64x32", "idxs" => "[1, 0]"],
                    ["shape" => "128x128", "idxs" => "[1, 0]"],
                    ["shape" => "8x16x32", "idxs" => "[0, 2, 1]"],
                    ["shape" => "8x16x32", "idxs" => "[2, 1, 0]"],
                    ["shape" => "8x16x32", "idxs" => "[1, 0, 2]"],
                    ["shape" => "16x8x2", "idxs" => "[0, 2, 1]"],
                    ["shape" => "4x8x16x32", "idxs" => "[0, 1, 3, 2]"],
                    ["shape" => "4x8x16x32", "idxs" => "[3, 2, 1, 0]"],
                    ["shape" => "4x8x16x32", "idxs" => "[1, 0, 2, 3]"],
                    ["shape" => "4x8x16x32", "idxs" => "[0, 2, 1, 3]"],
                ]
            ],
            [
                "run" => 12,
                "warmup" => true,
                "name" => "CudaArray::reshape()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayReshape",
                "metadata" => [
                    ["shape" => "4x4", "output-shape" => "16x1"],
                    ["shape" => "4x4", "output-shape" => "1x16"],
                    ["shape" => "8x8", "output-shape" => "4x16"],
                    ["shape" => "8x8", "output-shape" => "16x4"],
                    ["shape" => "12x12", "output-shape" => "3x4x12"],
                    ["shape" => "12x12", "output-shape" => "4x3x12"],
                    ["shape" => "24x6", "output-shape" => "8x3x6"],
                    ["shape" => "6x24", "output-shape" => "2x3x24"],
                    ["shape" => "3x4x5", "output-shape" => "12x5"],
                    ["shape" => "3x4x5", "output-shape" => "3x20"],
                    ["shape" => "2x6x8", "output-shape" => "12x8"],
                    ["shape" => "2x6x8", "output-shape" => "2x48"],
                ]
            ],
        ];
    }

    private function unaryMetadata(): array
    {
        return [
            ["shape" => "16x16x16", "type" => "3D"],
            ["shape" => "64x64x64", "type" => "3D"],
            ["shape" => "512x512x64", "type" => "3D"],
            ["shape" => "512x512x512", "type" => "3D"],
            ["shape" => "512x512", "type" => "2D"],
            ["shape" => "1024x512", "type" => "2D"],
            ["shape" => "1x180000", "type" => "2D"],
        ];
    }

    public function description(): string
    {
        return "CudaArray shape & manipulation methods Benchmark";
    }
    public function argsUnary(int $count): array
    {
        return match ($count) {
            1 => [CudaArray::rand([16, 16, 16])],
            2 => [CudaArray::rand([64, 64, 64])],
            3 => [CudaArray::rand([512, 512, 64])],
            4 => [CudaArray::rand([512, 512, 512])],
            5 => [CudaArray::rand([512, 512])],
            6 => [CudaArray::rand([1024, 512])],
            7 => [CudaArray::rand([1, 180000])],
        };
    }

    public function argsTranspose(int $count): array
    {
        return match ($count) {
            1 => [CudaArray::rand([16, 16]), [1, 0]],
            2 => [CudaArray::rand([32, 64]), [1, 0]],
            3 => [CudaArray::rand([64, 32]), [1, 0]],
            4 => [CudaArray::rand([128, 128]), [1, 0]],
            5 => [CudaArray::rand([8, 16, 32]), [0, 2, 1]],
            6 => [CudaArray::rand([8, 16, 32]), [2, 1, 0]],
            7 => [CudaArray::rand([8, 16, 32]), [1, 0, 2]],
            8 => [CudaArray::rand([16, 8, 4]), [0, 2, 1]],
            9 => [CudaArray::rand([4, 8, 16, 32]), [0, 1, 3, 2]],
            10 => [CudaArray::rand([4, 8, 16, 32]), [3, 2, 1, 0]],
            11 => [CudaArray::rand([4, 8, 16, 32]), [1, 0, 2, 3]],
            12 => [CudaArray::rand([4, 8, 16, 32]), [0, 2, 1, 3]]
        };
    }

    public function argsReshape(int $count): array
    {
        return match ($count) {
            1 => [CudaArray::rand([4, 4]), [16, 1]],
            2 => [CudaArray::rand([4, 4]), [1, 16]],
            3 => [CudaArray::rand([8, 8]), [4, 16]],
            4 => [CudaArray::rand([8, 8]), [16, 4]],
            5 => [CudaArray::rand([12, 12]), [3, 4, 12]],
            6 => [CudaArray::rand([12, 12]), [4, 3, 12]],
            7 => [CudaArray::rand([24, 6]), [8, 3, 6]],
            8 => [CudaArray::rand([6, 24]), [2, 3, 24]],
            9 => [CudaArray::rand([3, 4, 5]), [12, 5]],
            10 => [CudaArray::rand([3, 4, 5]), [3, 20]],
            11 => [CudaArray::rand([2, 6, 8]), [12, 8]],
            12 => [CudaArray::rand([2, 6, 8]), [2, 48]],
        };
    }

    #[InjectArgs("argsUnary")]
    public function cudaArrayFlatten(CudaArray $tensor): void
    {
        $tensor->flatten();
    }

    #[InjectArgs("argsReshape")]
    public function cudaArrayReshape(CudaArray $tensor, array $shape): void
    {
        $tensor->reshape($shape);
    }

    #[InjectArgs("argsTranspose")]
    public function cudaArrayTranspose(CudaArray $tensor, array $idxs): void
    {
        $tensor->transpose($idxs);
    }
  
}
