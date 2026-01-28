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
                "run" => 6,
                "warmup" => true,
                "name" => "CudaArray::zeros()",
                "iterations" => 10,
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
                "iterations" => 10,
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
                "iterations" => 10,
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
                "iterations" => 10,
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
            [
                "run" => 8,
                "warmup" => true,
                "name" => "CudaArray::matmul()",
                "iterations" => 10,
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
            [
                "run" => 16,
                "warmup" => true,
                "name" => "CudaArray::sum()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArraySum",
                "metadata" => $this->reductionMetadata()
            ],
            [
                "run" => 16,
                "warmup" => true,
                "name" => "CudaArray::min()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayMin",
                "metadata" => $this->reductionMetadata()
            ],
            [
                "run" => 16,
                "warmup" => true,
                "name" => "CudaArray::prod()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayProd",
                "metadata" => $this->reductionMetadata()
            ],
            [
                "run" => 16,
                "warmup" => true,
                "name" => "CudaArray::argMax()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayArgMax",
                "metadata" => $this->reductionMetadata()
            ],
            [
                "run" => 16,
                "warmup" => true,
                "name" => "CudaArray::argMin()",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayArgMin",
                "metadata" => $this->reductionMetadata()
            ],
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
            [
                "run" => 7,
                "warmup" => true,
                "name" => "CudaArray::toHost() [GPU -> ContiguousArray]",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArraytoHost",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "CudaArray::toArray() [GPU -> PHP Array]",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArraytoArray",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "CudaArray::__construct() [PHP Array -> GPU]",
                "iterations" => 10,
                "type" => "CUDA",
                "handler" => "cudaArrayConstructor",
                "metadata" => $this->unaryMetadata()
            ],
        ];
    }

    private function reductionMetadata(): array
    {
        return [
            ["shape" => "16x16x16", "type" => "3D", "axis" => "2"],
            ["shape" => "64x64x64", "type" => "3D", "axis" => "2"],
            ["shape" => "512x512x64", "type" => "3D", "axis" => "2"],
            ["shape" => "1024x512x512", "type" => "3D", "axis" => "2"],
            ["shape" => "16x16x16", "type" => "3D", "axis" => "1"],
            ["shape" => "64x64x64", "type" => "3D", "axis" => "1"],
            ["shape" => "512x512x64", "type" => "3D", "axis" => "1"],
            ["shape" => "1024x512x512", "type" => "3D", "axis" => "1"],
            ["shape" => "16x16x16", "type" => "3D", "axis" => "0"],
            ["shape" => "64x64x64", "type" => "3D", "axis" => "0"],
            ["shape" => "512x512x64", "type" => "3D", "axis" => "0"],
            ["shape" => "1024x512x512", "type" => "3D", "axis" => "0"],
            ["shape" => "16x16x16", "type" => "3D", "axis" => "-1"],
            ["shape" => "64x64x64", "type" => "3D", "axis" => "-1"],
            ["shape" => "512x512x64", "type" => "3D", "axis" => "-1"],
            ["shape" => "1024x512x512", "type" => "3D", "axis" => "-1"],

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
        return "CudaArray methods Benchmark";
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

    #[InjectArgs("argsReduction")]
    public function cudaArraySum(CudaArray $tensor, ?int $axis): void
    {
        $tensor->sum($axis);
    }

    #[InjectArgs("argsReduction")]
    public function cudaArrayMax(CudaArray $tensor, ?int $axis): void
    {
        $tensor->max($axis);
    }

    #[InjectArgs("argsReduction")]
    public function cudaArrayMin(CudaArray $tensor, ?int $axis): void
    {
        $tensor->min($axis);
    }

    #[InjectArgs("argsReduction")]
    public function cudaArrayProd(CudaArray $tensor, ?int $axis): void
    {
        $tensor->prod($axis);
    }

    #[InjectArgs("argsReduction")]
    public function cudaArrayArgMax(CudaArray $tensor, ?int $axis): void
    {
        $tensor->argMax($axis);
    }

    #[InjectArgs("argsReduction")]
    public function cudaArrayArgMin(CudaArray $tensor, ?int $axis): void
    {
        $tensor->argMin($axis);
    }
}
