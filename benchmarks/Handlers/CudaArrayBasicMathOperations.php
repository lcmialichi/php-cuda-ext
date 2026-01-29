<?php

namespace Benchmarks\Handlers;

use Cuda\CudaArray;
use Benchmarks\Handlers\Benchmark;
use Benchmarks\Support\Attr\InjectArgs;

class CudaArrayBasicMathOperations extends Benchmark
{
    public function name(): string
    {
        return "CudaArray Basic Operations Benchmark";
    }

    public function register(): array
    {
        return [
            [
                "run" => 10,
                "warmup" => true,
                "name" => "CudaArray::add()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayAdd",
                "metadata" => [
                    ["shape" => "16x16x16 + 16x16x16", "type" => "3D + 3D"],
                    ["shape" => "64x64x64 + 64x64x64", "type" => "3D + 3D"],
                    ["shape" => "512x512x64 + 512x512x64", "type" => "3D + 3D"],
                    ["shape" => "16x16x16 + float", "type" => "3D + Scalar"],
                    ["shape" => "64x64x64 + float", "type" => "3D + Scalar"],
                    ["shape" => "512x512x512 + float", "type" => "3D + Scalar"],
                    ["shape" => "512x64 + 512x64", "type" => "2D + 2D"],
                    ["shape" => "1024x512 + 1024x512", "type" => "2D + 2D"],
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
                    ["shape" => "16x16x16 - 16x16x16", "type" => "3D - 3D"],
                    ["shape" => "64x64x64 - 64x64x64", "type" => "3D - 3D"],
                    ["shape" => "512x512x64 - 512x512x64", "type" => "3D - 3D"],
                    ["shape" => "16x16x16 - float", "type" => "3D - Scalar"],
                    ["shape" => "64x64x64 - float", "type" => "3D - Scalar"],
                    ["shape" => "512x512x512 - float", "type" => "3D - Scalar"],
                    ["shape" => "512x64 - 512x64", "type" => "2D - 2D"],
                    ["shape" => "1024x512 - 1024x512", "type" => "2D - 2D"],
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
                    ["shape" => "16x16x16 * 16x16x16", "type" => "3D * 3D"],
                    ["shape" => "64x64x64 * 64x64x64", "type" => "3D * 3D"],
                    ["shape" => "512x512x64 * 512x512x64", "type" => "3D * 3D"],
                    ["shape" => "16x16x16 * float", "type" => "3D * Scalar"],
                    ["shape" => "64x64x64 * float", "type" => "3D * Scalar"],
                    ["shape" => "512x512x512 * float", "type" => "3D * Scalar"],
                    ["shape" => "512x64 * 512x64", "type" => "2D * 2D"],
                    ["shape" => "1024x512 * 1024x512", "type" => "2D * 2D"],
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
                    ["shape" => "16x16x16 / 16x16x16", "type" => "3D / 3D"],
                    ["shape" => "64x64x64 / 64x64x64", "type" => "3D / 3D"],
                    ["shape" => "512x512x64 / 512x512x64", "type" => "3D / 3D"],
                    ["shape" => "16x16x16 / float", "type" => "3D / Scalar"],
                    ["shape" => "64x64x64 / float", "type" => "3D / Scalar"],
                    ["shape" => "512x512x512 / float", "type" => "3D / Scalar"],
                    ["shape" => "512x64 / 512x64", "type" => "2D / 2D"],
                    ["shape" => "1024x512 / 1024x512", "type" => "2D / 2D"],
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
                    ["shape" => "16x16x16 ** 16x16x16", "type" => "3D ** 3D"],
                    ["shape" => "64x64x64 ** 64x64x64", "type" => "3D ** 3D"],
                    ["shape" => "512x512x64 ** 512x512x64", "type" => "3D ** 3D"],
                    ["shape" => "16x16x16 ** float", "type" => "3D ** Scalar"],
                    ["shape" => "64x64x64 ** float", "type" => "3D ** Scalar"],
                    ["shape" => "512x512x512 ** float", "type" => "3D ** Scalar"],
                    ["shape" => "512x64 ** 512x64", "type" => "2D ** 2D"],
                    ["shape" => "1024x512 ** 1024x512", "type" => "2D ** 2D"],
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
                    ["shape" => "16x16x16 > 16x16x16", "type" => "3D > 3D"],
                    ["shape" => "64x64x64 > 64x64x64", "type" => "3D > 3D"],
                    ["shape" => "512x512x64 > 512x512x64", "type" => "3D > 3D"],
                    ["shape" => "16x16x16 > float", "type" => "3D > Scalar"],
                    ["shape" => "64x64x64 > float", "type" => "3D > Scalar"],
                    ["shape" => "512x512x512 > float", "type" => "3D > Scalar"],
                    ["shape" => "512x64 > 512x64", "type" => "2D > 2D"],
                    ["shape" => "1024x512 > 1024x512", "type" => "2D > 2D"],
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
                    ["shape" => "16x16x16 < 16x16x16", "type" => "3D < 3D"],
                    ["shape" => "64x64x64 < 64x64x64", "type" => "3D < 3D"],
                    ["shape" => "512x512x64 < 512x512x64", "type" => "3D < 3D"],
                    ["shape" => "16x16x16 < float", "type" => "3D < Scalar"],
                    ["shape" => "64x64x64 < float", "type" => "3D < Scalar"],
                    ["shape" => "512x512x512 < float", "type" => "3D < Scalar"],
                    ["shape" => "512x64 < 512x64", "type" => "2D < 2D"],
                    ["shape" => "1024x512 < 1024x512", "type" => "2D < 2D"],
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
                    ["shape" => "16x16x16 == 16x16x16", "type" => "3D == 3D"],
                    ["shape" => "64x64x64 == 64x64x64", "type" => "3D == 3D"],
                    ["shape" => "512x512x64 == 512x512x64", "type" => "3D == 3D"],
                    ["shape" => "16x16x16 == float", "type" => "3D == Scalar"],
                    ["shape" => "64x64x64 == float", "type" => "3D == Scalar"],
                    ["shape" => "512x512x512 == float", "type" => "3D == Scalar"],
                    ["shape" => "512x64 == 512x64", "type" => "2D == 2D"],
                    ["shape" => "1024x512 == 1024x512", "type" => "2D == 2D"],
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
                    ["shape" => "16x16x16 != 16x16x16", "type" => "3D != 3D"],
                    ["shape" => "64x64x64 != 64x64x64", "type" => "3D != 3D"],
                    ["shape" => "512x512x64 != 512x512x64", "type" => "3D != 3D"],
                    ["shape" => "16x16x16 != float", "type" => "3D != Scalar"],
                    ["shape" => "64x64x64 != float", "type" => "3D != Scalar"],
                    ["shape" => "512x512x512 != float", "type" => "3D != Scalar"],
                    ["shape" => "512x64 != 512x64", "type" => "2D != 2D"],
                    ["shape" => "1024x512 != 1024x512", "type" => "2D != 2D"],
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
                    ["shape" => "16x16x16 <= 16x16x16", "type" => "3D <= 3D"],
                    ["shape" => "64x64x64 <= 64x64x64", "type" => "3D <= 3D"],
                    ["shape" => "512x512x64 <= 512x512x64", "type" => "3D <= 3D"],
                    ["shape" => "16x16x16 <= float", "type" => "3D <= Scalar"],
                    ["shape" => "64x64x64 <= float", "type" => "3D <= Scalar"],
                    ["shape" => "512x512x512 <= float", "type" => "3D <= Scalar"],
                    ["shape" => "512x64 <= 512x64", "type" => "2D <= 2D"],
                    ["shape" => "1024x512 <= 1024x512", "type" => "2D <= 2D"],
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
                    ["shape" => "16x16x16 >= 16x16x16", "type" => "3D >= 3D"],
                    ["shape" => "64x64x64 >= 64x64x64", "type" => "3D >= 3D"],
                    ["shape" => "512x512x64 >= 512x512x64", "type" => "3D >= 3D"],
                    ["shape" => "16x16x16 >= float", "type" => "3D >= Scalar"],
                    ["shape" => "64x64x64 >= float", "type" => "3D >= Scalar"],
                    ["shape" => "512x512x512 >= float", "type" => "3D >= Scalar"],
                    ["shape" => "512x64 >= 512x64", "type" => "2D >= 2D"],
                    ["shape" => "1024x512 >= 1024x512", "type" => "2D >= 2D"],
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
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "CudaArray::floor()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayFloor",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "CudaArray::ceil()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayCeil",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "CudaArray::round()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayRound",
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
        return "CudaArray basic operation methods Benchmark";
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
}
