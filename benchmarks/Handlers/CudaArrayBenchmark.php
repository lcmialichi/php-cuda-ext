<?php

namespace Benchmarks\Handlers;

use Cuda\CudaArray;
use Benchmarks\Handlers\Benchmark;
use Benchmarks\Support\Attr\InjectArgs;

class CudaArrayBenchmark extends Benchmark
{
    public function name(): string
    {
        return "CudaArray Benchmark";
    }

    public function register(): array
    {
        return [
            [
                "run" => 3,
                "warmup" => true,
                "name" => "CudaArray::zeros()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayZeros",
                "metadata" => [
                    ["shape" => "16x16x16"],
                    ["shape" => "64x64x64"],
                    ["shape" => "512x512x512"],
                ]
            ],
            [
                "run" => 3,
                "warmup" => true,
                "name" => "CudaArray::ones()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayOnes",
                "metadata" => [
                    ["shape" => "16x16x16"],
                    ["shape" => "64x64x64"],
                    ["shape" => "512x512x512"],
                ]
            ],
            [
                "run" => 3,
                "warmup" => true,
                "name" => "CudaArray::full()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayFull",
                "metadata" => [
                    ["shape" => "16x16x16"],
                    ["shape" => "64x64x64"],
                    ["shape" => "512x512x512"],
                ]

            ],
            [
                "run" => 3,
                "warmup" => true,
                "name" => "CudaArray::rand()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayRand",
                "metadata" => [
                    ["shape" => "16x16x16"],
                    ["shape" => "64x64x64"],
                    ["shape" => "512x512x512"],
                ]
            ],
            [
                "run" => 3,
                "warmup" => true,
                "name" => "CudaArray::concat()",
                "iterations" => 10,
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
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayConcatAxisOne",
                "metadata" => [
                    ["shape" => "16x16", "axis" => "1"],
                    ["shape" => "64x64", "axis" => "1"],
                    ["shape" => "512x512", "axis" => "1"],
                ]
            ],
            [
                "run" => 10,
                "warmup" => true,
                "name" => "CudaArray::add()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayAdd",
                "metadata" => [
                    ["shape" => "16x16x16 + 16x16x16", "type" => "3D Tensor + 3D Tesor"],
                    ["shape" => "64x64x64 + 64x64x64", "type" => "3D Tensor + 3D Tesor"],
                    ["shape" => "512x512x64 + 512x512x64", "type" => "3D Tensor + 3D Tesor"],
                    ["shape" => "16x16x16 + float", "type" => "3D Tensor + Scalar"],
                    ["shape" => "64x64x64 + float", "type" => "3D Tensor + Scalar"],
                    ["shape" => "512x512x512 + float", "type" => "3D Tensor + Scalar"],
                    ["shape" => "512x64 + 512x64", "type" => "2D Tensor + 2D Tesor"],
                    ["shape" => "1024x512 + 1024x512", "type" => "2D Tensor + 2D Tesor"],
                    ["shape" => "1024x512 + 1x512", "type" => "2D Broadcast"],
                    ["shape" => "1024x3x512 + 1x3x512", "type" => "3D Broadcast"],
                ]
            ],
            [
                "run" => 10,
                "warmup" => true,
                "name" => "CudaArray::subtract()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArraySubtract",
                "metadata" => [
                    ["shape" => "16x16x16 - 16x16x16", "type" => "3D Tensor - 3D Tesor"],
                    ["shape" => "64x64x64 - 64x64x64", "type" => "3D Tensor - 3D Tesor"],
                    ["shape" => "512x512x64 - 512x512x64", "type" => "3D Tensor - 3D Tesor"],
                    ["shape" => "16x16x16 - float", "type" => "3D Tensor - Scalar"],
                    ["shape" => "64x64x64 - float", "type" => "3D Tensor - Scalar"],
                    ["shape" => "512x512x512 - float", "type" => "3D Tensor - Scalar"],
                    ["shape" => "512x64 - 512x64", "type" => "2D Tensor - 2D Tesor"],
                    ["shape" => "1024x512 - 1024x512", "type" => "2D Tensor - 2D Tesor"],
                    ["shape" => "1024x512 - 1x512", "type" => "2D Broadcast"],
                    ["shape" => "1024x3x512 - 1x3x512", "type" => "3D Broadcast"],
                ]
            ],
            [
                "run" => 10,
                "warmup" => true,
                "name" => "CudaArray::multiply()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayMultiply",
                "metadata" => [
                    ["shape" => "16x16x16 * 16x16x16", "type" => "3D Tensor * 3D Tesor"],
                    ["shape" => "64x64x64 * 64x64x64", "type" => "3D Tensor * 3D Tesor"],
                    ["shape" => "512x512x64 * 512x512x64", "type" => "3D Tensor * 3D Tesor"],
                    ["shape" => "16x16x16 * float", "type" => "3D Tensor * Scalar"],
                    ["shape" => "64x64x64 * float", "type" => "3D Tensor * Scalar"],
                    ["shape" => "512x512x512 * float", "type" => "3D Tensor * Scalar"],
                    ["shape" => "512x64 * 512x64", "type" => "2D Tensor * 2D Tesor"],
                    ["shape" => "1024x512 * 1024x512", "type" => "2D Tensor * 2D Tesor"],
                    ["shape" => "1024x512 * 1x512", "type" => "2D Broadcast"],
                    ["shape" => "1024x3x512 * 1x3x512", "type" => "3D Broadcast"],
                ]
            ],
            [
                "run" => 10,
                "warmup" => true,
                "name" => "CudaArray::divide()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayDivide",
                "metadata" => [
                    ["shape" => "16x16x16 / 16x16x16", "type" => "3D Tensor / 3D Tesor"],
                    ["shape" => "64x64x64 / 64x64x64", "type" => "3D Tensor / 3D Tesor"],
                    ["shape" => "512x512x64 / 512x512x64", "type" => "3D Tensor / 3D Tesor"],
                    ["shape" => "16x16x16 / float", "type" => "3D Tensor / Scalar"],
                    ["shape" => "64x64x64 / float", "type" => "3D Tensor / Scalar"],
                    ["shape" => "512x512x512 / float", "type" => "3D Tensor / Scalar"],
                    ["shape" => "512x64 / 512x64", "type" => "2D Tensor / 2D Tesor"],
                    ["shape" => "1024x512 / 1024x512", "type" => "2D Tensor / 2D Tesor"],
                    ["shape" => "1024x512 / 1x512", "type" => "2D Broadcast"],
                    ["shape" => "1024x3x512 / 1x3x512", "type" => "3D Broadcast"],
                ]
            ],
            [
                "run" => 10,
                "warmup" => true,
                "name" => "CudaArray::power()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayPower",
                "metadata" => [
                    ["shape" => "16x16x16 ** 16x16x16", "type" => "3D Tensor ** 3D Tesor"],
                    ["shape" => "64x64x64 ** 64x64x64", "type" => "3D Tensor ** 3D Tesor"],
                    ["shape" => "512x512x64 ** 512x512x64", "type" => "3D Tensor ** 3D Tesor"],
                    ["shape" => "16x16x16 ** float", "type" => "3D Tensor ** Scalar"],
                    ["shape" => "64x64x64 ** float", "type" => "3D Tensor ** Scalar"],
                    ["shape" => "512x512x512 ** float", "type" => "3D Tensor ** Scalar"],
                    ["shape" => "512x64 ** 512x64", "type" => "2D Tensor ** 2D Tesor"],
                    ["shape" => "1024x512 ** 1024x512", "type" => "2D Tensor ** 2D Tesor"],
                    ["shape" => "1024x512 ** 1x512", "type" => "2D Broadcast"],
                    ["shape" => "1024x3x512 ** 1x3x512", "type" => "3D Broadcast"],
                ]
            ],
            [
                "run" => 10,
                "warmup" => true,
                "name" => "CudaArray::gt()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayGt",
                "metadata" => [
                    ["shape" => "16x16x16 > 16x16x16", "type" => "3D Tensor > 3D Tesor"],
                    ["shape" => "64x64x64 > 64x64x64", "type" => "3D Tensor > 3D Tesor"],
                    ["shape" => "512x512x64 > 512x512x64", "type" => "3D Tensor > 3D Tesor"],
                    ["shape" => "16x16x16 > float", "type" => "3D Tensor > Scalar"],
                    ["shape" => "64x64x64 > float", "type" => "3D Tensor > Scalar"],
                    ["shape" => "512x512x512 > float", "type" => "3D Tensor > Scalar"],
                    ["shape" => "512x64 > 512x64", "type" => "2D Tensor > 2D Tesor"],
                    ["shape" => "1024x512 > 1024x512", "type" => "2D Tensor > 2D Tesor"],
                    ["shape" => "1024x512 > 1x512", "type" => "2D Broadcast"],
                    ["shape" => "1024x3x512 > 1x3x512", "type" => "3D Broadcast"],
                ]
            ],
            [
                "run" => 10,
                "warmup" => true,
                "name" => "CudaArray::lt()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayLt",
                "metadata" => [
                    ["shape" => "16x16x16 < 16x16x16", "type" => "3D Tensor < 3D Tesor"],
                    ["shape" => "64x64x64 < 64x64x64", "type" => "3D Tensor < 3D Tesor"],
                    ["shape" => "512x512x64 < 512x512x64", "type" => "3D Tensor < 3D Tesor"],
                    ["shape" => "16x16x16 < float", "type" => "3D Tensor < Scalar"],
                    ["shape" => "64x64x64 < float", "type" => "3D Tensor < Scalar"],
                    ["shape" => "512x512x512 < float", "type" => "3D Tensor < Scalar"],
                    ["shape" => "512x64 < 512x64", "type" => "2D Tensor < 2D Tesor"],
                    ["shape" => "1024x512 < 1024x512", "type" => "2D Tensor < 2D Tesor"],
                    ["shape" => "1024x512 < 1x512", "type" => "2D Broadcast"],
                    ["shape" => "1024x3x512 < 1x3x512", "type" => "3D Broadcast"],
                ]
            ],
            [
                "run" => 10,
                "warmup" => true,
                "name" => "CudaArray::eq()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayEq",
                "metadata" => [
                    ["shape" => "16x16x16 == 16x16x16", "type" => "3D Tensor == 3D Tesor"],
                    ["shape" => "64x64x64 == 64x64x64", "type" => "3D Tensor == 3D Tesor"],
                    ["shape" => "512x512x64 == 512x512x64", "type" => "3D Tensor == 3D Tesor"],
                    ["shape" => "16x16x16 == float", "type" => "3D Tensor == Scalar"],
                    ["shape" => "64x64x64 == float", "type" => "3D Tensor == Scalar"],
                    ["shape" => "512x512x512 == float", "type" => "3D Tensor == Scalar"],
                    ["shape" => "512x64 == 512x64", "type" => "2D Tensor == 2D Tesor"],
                    ["shape" => "1024x512 == 1024x512", "type" => "2D Tensor == 2D Tesor"],
                    ["shape" => "1024x512 == 1x512", "type" => "2D Broadcast"],
                    ["shape" => "1024x3x512 == 1x3x512", "type" => "3D Broadcast"],
                ]
            ],
            [
                "run" => 10,
                "warmup" => true,
                "name" => "CudaArray::ne()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayNe",
                "metadata" => [
                    ["shape" => "16x16x16 != 16x16x16", "type" => "3D Tensor != 3D Tesor"],
                    ["shape" => "64x64x64 != 64x64x64", "type" => "3D Tensor != 3D Tesor"],
                    ["shape" => "512x512x64 != 512x512x64", "type" => "3D Tensor != 3D Tesor"],
                    ["shape" => "16x16x16 != float", "type" => "3D Tensor != Scalar"],
                    ["shape" => "64x64x64 != float", "type" => "3D Tensor != Scalar"],
                    ["shape" => "512x512x512 != float", "type" => "3D Tensor != Scalar"],
                    ["shape" => "512x64 != 512x64", "type" => "2D Tensor != 2D Tesor"],
                    ["shape" => "1024x512 != 1024x512", "type" => "2D Tensor != 2D Tesor"],
                    ["shape" => "1024x512 != 1x512", "type" => "2D Broadcast"],
                    ["shape" => "1024x3x512 != 1x3x512", "type" => "3D Broadcast"],
                ]
            ],
            [
                "run" => 10,
                "warmup" => true,
                "name" => "CudaArray::le()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayLe",
                "metadata" => [
                    ["shape" => "16x16x16 <= 16x16x16", "type" => "3D Tensor <= 3D Tesor"],
                    ["shape" => "64x64x64 <= 64x64x64", "type" => "3D Tensor <= 3D Tesor"],
                    ["shape" => "512x512x64 <= 512x512x64", "type" => "3D Tensor <= 3D Tesor"],
                    ["shape" => "16x16x16 <= float", "type" => "3D Tensor <= Scalar"],
                    ["shape" => "64x64x64 <= float", "type" => "3D Tensor <= Scalar"],
                    ["shape" => "512x512x512 <= float", "type" => "3D Tensor <= Scalar"],
                    ["shape" => "512x64 <= 512x64", "type" => "2D Tensor <= 2D Tesor"],
                    ["shape" => "1024x512 <= 1024x512", "type" => "2D Tensor <= 2D Tesor"],
                    ["shape" => "1024x512 <= 1x512", "type" => "2D Broadcast"],
                    ["shape" => "1024x3x512 <= 1x3x512", "type" => "3D Broadcast"],
                ]
            ],
            [
                "run" => 10,
                "warmup" => true,
                "name" => "CudaArray::ge()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayGe",
                "metadata" => [
                    ["shape" => "16x16x16 >= 16x16x16", "type" => "3D Tensor >= 3D Tesor"],
                    ["shape" => "64x64x64 >= 64x64x64", "type" => "3D Tensor >= 3D Tesor"],
                    ["shape" => "512x512x64 >= 512x512x64", "type" => "3D Tensor >= 3D Tesor"],
                    ["shape" => "16x16x16 >= float", "type" => "3D Tensor >= Scalar"],
                    ["shape" => "64x64x64 >= float", "type" => "3D Tensor >= Scalar"],
                    ["shape" => "512x512x512 >= float", "type" => "3D Tensor >= Scalar"],
                    ["shape" => "512x64 >= 512x64", "type" => "2D Tensor >= 2D Tesor"],
                    ["shape" => "1024x512 >= 1024x512", "type" => "2D Tensor >= 2D Tesor"],
                    ["shape" => "1024x512 >= 1x512", "type" => "2D Broadcast"],
                    ["shape" => "1024x3x512 >= 1x3x512", "type" => "3D Broadcast"],
                ]
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "CudaArray::neg()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayNeg",
                "metadata" => [
                    ["shape" => " - 16x16x16", "type" => " - 3D Tensor"],
                    ["shape" => " - 64x64x64", "type" => " - 3D Tesor"],
                    ["shape" => " - 512x512x64", "type" => " - 3D Tesor"],
                    ["shape" => " - 512x512x512", "type" => " - 3D Tesor"],
                    ["shape" => " - 512x512", "type" => " - 2D Tesor"],
                    ["shape" => " - 1024x512", "type" => " - 2D Tesor"],
                    ["shape" => " - 1x180000", "type" => " - 2D Tesor"],
                ]
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "CudaArray::floor()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayFloor",
                "metadata" => [
                    ["shape" => "16x16x16", "type" => "3D Tensor"],
                    ["shape" => "64x64x64", "type" => "3D Tesor"],
                    ["shape" => "512x512x64", "type" => "3D Tesor"],
                    ["shape" => "512x512x512", "type" => "3D Tesor"],
                    ["shape" => "512x512", "type" => "2D Tesor"],
                    ["shape" => "1024x512", "type" => "2D Tesor"],
                    ["shape" => "1x180000", "type" => " - 2D Tesor"],
                ]
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "CudaArray::ceil()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayCeil",
                "metadata" => [
                    ["shape" => "16x16x16", "type" => "3D Tensor"],
                    ["shape" => "64x64x64", "type" => "3D Tesor"],
                    ["shape" => "512x512x64", "type" => "3D Tesor"],
                    ["shape" => "512x512x512", "type" => "3D Tesor"],
                    ["shape" => "512x512", "type" => "2D Tesor"],
                    ["shape" => "1024x512", "type" => "2D Tesor"],
                    ["shape" => "1x180000", "type" => " - 2D Tesor"],
                ]
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "CudaArray::round()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayRound",
                "metadata" => [
                    ["shape" => "16x16x16", "type" => "3D Tensor"],
                    ["shape" => "64x64x64", "type" => "3D Tesor"],
                    ["shape" => "512x512x64", "type" => "3D Tesor"],
                    ["shape" => "512x512x512", "type" => "3D Tesor"],
                    ["shape" => "512x512", "type" => "2D Tesor"],
                    ["shape" => "1024x512", "type" => "2D Tesor"],
                    ["shape" => "1x180000", "type" => " - 2D Tesor"],
                ]
            ],
            [
                "run" => 8,
                "warmup" => true,
                "name" => "CudaArray::matmul()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayMatmul",
                "metadata" => [
                    ["shape" => "16x16 x 16x16", "type" => " 2D Tensor"],
                    ["shape" => " 128x128 x 128x128", "type" => " 2D Tensor"],
                    ["shape" => " 32x256x256 x 32x256x256", "type" => " 3D Tensor"],
                    ["shape" => " 1x512x512 x 64x512x512", "type" => " 3D Tensor (broadcast)"],
                    ["shape" => " 1024x768 x 768x512", "type" => " 2D Tensor"],
                    ["shape" => " 64x1024x512 x 64x512x256", "type" => " 3D Tensor"],
                    ["shape" => " 1x1000000 x 1000000x1", "type" => " 2D Tensor (extreme)"],
                    ["shape" => " 8x64x256x256 x 8x64x256x256", "type" => " 4D Tensor"],
                ]
            ]
        ];
    }

    public function description(): string
    {
        return "CudaArray methods Benchmark";
    }

    public function args3DShape(int $count): array
    {
        $size = pow(8, $count);
        return [[$size, $size, $size]];
    }

    public function args3DAndValue(int $count): array
    {
        $size = pow(8, $count);
        return [[$size, $size, $size], $count * 2];
    }

    public function args3DAndRange(int $count): array
    {
        $size = pow(8, $count);
        return [[$size, $size, $size], -1, 1];
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

    #[InjectArgs("argsArithmetic")]
    public function cudaArrayAdd(CudaArray $first, CudaArray|float $second): void
    {
        $first + $second;
    }

    #[InjectArgs("argsArithmetic")]
    public function cudaArraySubtract(CudaArray $first, CudaArray|float $second): void
    {
        $first - $second;
    }

    #[InjectArgs("argsArithmetic")]
    public function cudaArrayMultiply(CudaArray $first, CudaArray|float $second): void
    {
        $first * $second;
    }

    #[InjectArgs("argsArithmetic")]
    public function cudaArrayDivide(CudaArray $first, CudaArray|float $second): void
    {
        $first / $second;
    }

    #[InjectArgs("argsArithmetic")]
    public function cudaArrayGt(CudaArray $first, CudaArray|float $second): void
    {
        $first->gt($second);
    }

    #[InjectArgs("argsArithmetic")]
    public function cudaArrayLt(CudaArray $first, CudaArray|float $second): void
    {
        $first->lt($second);
    }

    #[InjectArgs("argsArithmetic")]
    public function cudaArrayEq(CudaArray $first, CudaArray|float $second): void
    {
        $first->eq($second);
    }

    #[InjectArgs("argsArithmetic")]
    public function cudaArrayGe(CudaArray $first, CudaArray|float $second): void
    {
        $first->ge($second);
    }

    #[InjectArgs("argsArithmetic")]
    public function cudaArrayNe(CudaArray $first, CudaArray|float $second): void
    {
        $first->ne($second);
    }

    #[InjectArgs("argsArithmetic")]
    public function cudaArrayle(CudaArray $first, CudaArray|float $second): void
    {
        $first->le($second);
    }

    #[InjectArgs("argsArithmetic")]
    public function cudaArrayPower(CudaArray $first, CudaArray|float $second): void
    {
        $first ** $second;
    }

    #[InjectArgs("argsUnary")]
    public function cudaArrayNeg(CudaArray $tensor): void
    {
        -$tensor;
    }

    #[InjectArgs("argsUnary")]
    public function cudaArrayFloor(CudaArray $tensor): void
    {
        $tensor->floor();
    }

    #[InjectArgs("argsUnary")]
    public function cudaArrayCeil(CudaArray $tensor): void
    {
        $tensor->ceil();
    }

    #[InjectArgs("argsUnary")]
    public function cudaArrayRound(CudaArray $tensor): void
    {
        $tensor->round();
    }

    #[InjectArgs("argsMatmul")]
    public function cudaArrayMatmul(CudaArray $first, CudaArray $second): void
    {
        $first->matmul($second);
    }
}
