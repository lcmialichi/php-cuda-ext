<?php

namespace Benchmarks\Handlers;

use Cuda\CudaArray;
use Benchmarks\Handlers\Benchmark;
use Benchmarks\Support\Attr\InjectArgs;

class CudaArrayMemoryCopyOperationsBenchmark extends Benchmark
{
    public function name(): string
    {
        return "CudaArray Memory Copy Benchmark";
    }

    public function register(): array
    {
        return [
            [
                "run" => 6,
                "warmup" => true,
                "name" => "CudaArray::zeros()",
                "iterations" => 50,
                "type" => "CUDA",
                "handler" => "cudaArrayZeros",
                "metadata" => [
                    ["shape" => "16x16x16"],
                    ["shape" => "64x64x64"],
                    ["shape" => "128x128x128"],
                    ["shape" => "256x256x256"],
                    ["shape" => "1024x128x64"],
                    ["shape" => "512x256x128"],
                ]
            ],
            [
                "run" => 6,
                "warmup" => true,
                "name" => "CudaArray::ones()",
                "iterations" => 50,
                "type" => "CUDA",
                "handler" => "cudaArrayOnes",
                "metadata" => [
                    ["shape" => "16x16x16"],
                    ["shape" => "64x64x64"],
                    ["shape" => "128x128x128"],
                    ["shape" => "256x256x256"],
                    ["shape" => "1024x128x64"],
                    ["shape" => "512x256x128"],
                ]
            ],
            [
                "run" => 6,
                "warmup" => true,
                "name" => "CudaArray::full()",
                "iterations" => 50,
                "type" => "CUDA",
                "handler" => "cudaArrayFull",
                "metadata" => [
                    ["shape" => "16x16x16"],
                    ["shape" => "64x64x64"],
                    ["shape" => "128x128x128"],
                    ["shape" => "256x256x256"],
                    ["shape" => "1024x128x64"],
                    ["shape" => "512x256x128"],
                ]

            ],
            [
                "run" => 3,
                "warmup" => true,
                "name" => "CudaArray::rand()",
                "iterations" => 50,
                "type" => "CUDA",
                "handler" => "cudaArrayRand",
                "metadata" => [
                    ["shape" => "16x16x16"],
                    ["shape" => "64x64x64"],
                    ["shape" => "128x128x128"],
                    ["shape" => "256x256x256"],
                    ["shape" => "1024x128x64"],
                    ["shape" => "512x256x128"],
                ]
            ],
            [
                "run" => 3,
                "warmup" => true,
                "name" => "CudaArray::concat()",
                "iterations" => 50,
                "type" => "CUDA",
                "handler" => "cudaArrayConcatAxisZero",
                "metadata" => [
                    ["shape" => "16x16", "axis" => "0"],
                    ["shape" => "64x64", "axis" => "0"],
                    ["shape" => "512x512", "axis" => "0"],
                ]
            ],
            [
                "run" => 3,
                "warmup" => true,
                "name" => "CudaArray::concat()",
                "iterations" => 50,
                "type" => "CUDA",
                "handler" => "cudaArrayConcatAxisOne",
                "metadata" => [
                    ["shape" => "16x16", "axis" => "1"],
                    ["shape" => "64x64", "axis" => "1"],
                    ["shape" => "512x512", "axis" => "1"],
                ]
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "CudaArray::toHost() [GPU -> ContiguousArray]",
                "iterations" => 50,
                "type" => "CUDA",
                "handler" => "cudaArraytoHost",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "CudaArray::toArray() [GPU -> PHP Array]",
                "iterations" => 50,
                "type" => "CUDA",
                "handler" => "cudaArraytoArray",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "CudaArray::__construct() [PHP Array -> GPU]",
                "iterations" => 50,
                "type" => "CUDA",
                "handler" => "cudaArrayConstructor",
                "metadata" => $this->unaryMetadata()
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
        return "CudaArray memory copy methods Benchmark";
    }

    public function args3DShape(int $count): array
    {
        return  match ($count) {
            1 => [[16, 16, 16]],
            2 => [[64, 64, 64]],
            3 => [[128, 128, 128]],
            4 => [[256, 256, 256]],
            5 => [[1024, 128, 64]],
            6 => [[512, 256, 128]],
        };
    }

    public function args3DAndValue(int $count): array
    {
        return  match ($count) {
            1 => [[16, 16, 16], 10],
            2 => [[64, 64, 64], 10],
            3 => [[128, 128, 128], 10],
            4 => [[256, 256, 256], 10],
            5 => [[1024, 128, 64], 10],
            6 => [[512, 256, 128], 10],
        };
    }

    public function args3DAndRange(int $count): array
    {
        return  match ($count) {
            1 => [[16, 16, 16], -1, 1],
            2 => [[64, 64, 64],  -1, 1],
            3 => [[128, 128, 128],  -1, 1],
            4 => [[256, 256, 256],  -1, 1],
            5 => [[1024, 128, 64],  -1, 1],
            6 => [[512, 256, 128],  -1, 1],
        };
    }

    public function argsCudaArrayConcatAxisZero(int $count): array
    {
        $size = pow(8, $count);
        $shape = [$size, $size];
        return [CudaArray::rand($shape), CudaArray::rand($shape), 0];
    }

    public function argsCudaArrayConcatAxisOne(int $count): array
    {
        $size = pow(8, $count);
        $shape = [$size, $size];
        return [CudaArray::rand($shape), CudaArray::rand($shape), 1];
    }

    public function argsArithmetic(int $count): array
    {
        return match ($count) {
            1 => [CudaArray::rand([16, 16, 16]), CudaArray::rand([16, 16, 16])],
            2 => [CudaArray::rand([64, 64, 64]), CudaArray::rand([64, 64, 64])],
            3 => [CudaArray::rand([512, 512, 64]), CudaArray::rand([512, 512, 64])],
            4 => [CudaArray::rand([16, 16, 16]), 2],
            5 => [CudaArray::rand([64, 64, 64]), 2],
            6 => [CudaArray::rand([512, 512, 512]), 2],
            7 => [CudaArray::rand([512, 64]), CudaArray::rand([512, 64])],
            8 => [CudaArray::rand([1024, 512]), CudaArray::rand([1024, 512])],
            9 => [CudaArray::rand([1024, 512]), CudaArray::rand([1, 512])],
            10 => [CudaArray::rand([1024, 3, 512]), CudaArray::rand([1, 3, 1])],
        };
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

    public function argsReduction(int $count): array
    {
        return match ($count) {
            1 => [CudaArray::rand([16, 16, 16]), 2],
            2 => [CudaArray::rand([64, 64, 64]), 2],
            3 => [CudaArray::rand([512, 512, 64]), 2],
            4 => [CudaArray::rand([512, 512, 512]), 2],
            5 => [CudaArray::rand([16, 16, 16]), 1],
            6 => [CudaArray::rand([64, 64, 64]), 1],
            7 => [CudaArray::rand([512, 512, 64]), 1],
            8 => [CudaArray::rand([512, 512, 512]), 1],
            9 => [CudaArray::rand([16, 16, 16]), 0],
            10 => [CudaArray::rand([64, 64, 64]), 0],
            11 => [CudaArray::rand([512, 512, 64]), 0],
            12 => [CudaArray::rand([512, 512, 512]), 0],
            13 => [CudaArray::rand([16, 16, 16]), null],
            14 => [CudaArray::rand([64, 64, 64]), null],
            15 => [CudaArray::rand([512, 512, 64]), null],
            16 => [CudaArray::rand([512, 512, 512]), null],
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

    public function argsTransfer(int $count): array
    {
        return match ($count) {
            1 => [CudaArray::rand([16, 16, 16])],
            2 => [CudaArray::rand([64, 64, 64])],
            3 => [CudaArray::rand([1024, 512, 64])],
            4 => [CudaArray::rand([512, 512])],
            5 => [CudaArray::rand([1024, 512])],
            6 => [CudaArray::rand([1, 180000])],
            7 => [CudaArray::rand([180000])],
        };
    }

    public function argsConstructor(int $count): array
    {
        return match ($count) {
            1 => [array_fill(0, 15, array_fill(0, 15, array_fill(0, 15, $count)))],
            2 => [array_fill(0, 63, array_fill(0, 63, array_fill(0, 63, $count)))],
            3 => [array_fill(0, 1023, array_fill(0, 511, array_fill(0, 63, $count)))],
            4 => [array_fill(0, 511, array_fill(0, 511, $count))],
            5 => [array_fill(0, 1023, array_fill(0, 511, $count))],
            6 => [array_fill(0, 1, array_fill(0, 180000, $count))],
            7 => [array_fill(0, 180000, $count)],
        };
    }

    #[InjectArgs("args3DShape")]
    public function cudaArrayOnes(array $shape): void
    {
        CudaArray::ones($shape);
    }

    #[InjectArgs("args3DShape")]
    public function cudaArrayZeros(array $shape): void
    {
        CudaArray::zeros($shape);
    }

    #[InjectArgs("args3DAndValue")]
    public function cudaArrayFull(array $shape, int $value): void
    {
        CudaArray::full($shape, $value);
    }

    #[InjectArgs("args3DAndRange")]
    public function cudaArrayRand(array $shape, float $min, float $max): void
    {
        CudaArray::rand($shape, $min, $max);
    }

    #[InjectArgs("argsCudaArrayConcatAxisZero")]
    public function cudaArrayConcatAxisZero(CudaArray $first, CudaArray $second, int $axis): void
    {
        $first->concat([$second], $axis);
    }

    #[InjectArgs("argsCudaArrayConcatAxisOne")]
    public function cudaArrayConcatAxisOne(CudaArray $first, CudaArray $second, int $axis): void
    {
        $first->concat([$second], $axis);
    }

    #[InjectArgs("argsTransfer")]
    public function cudaArrayToHost(CudaArray $tensor): void
    {
        $tensor->toHost();
    }

    #[InjectArgs("argsTransfer")]
    public function cudaArrayToArray(CudaArray $tensor): void
    {
        $tensor->toArray();
    }

    #[InjectArgs("argsConstructor")]
    public function cudaArrayConstructor(array $phpArray): void
    {
        new CudaArray($phpArray);
    }
}
