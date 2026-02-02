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
                "run" => 20,
                "warmup" => true,
                "name" => "CudaArray::add()",
                "iterations" => 100,
                "type" => "CUDA",
                "handler" => "cudaArrayAdd",
                "metadata" => [
                    ["shape" => "256", "type" => "1D + 1D (tiny)", "category" => "elementwise"],
                    ["shape" => "65536", "type" => "1D + 1D (small)", "category" => "elementwise"],
                    ["shape" => "16777216", "type" => "1D + 1D (medium)", "category" => "elementwise"],
                    ["shape" => "67108864", "type" => "1D + 1D (large)", "category" => "elementwise"],
                    ["shape" => "16x16", "type" => "2D + 2D (tiny square)", "category" => "elementwise"],
                    ["shape" => "256x256", "type" => "2D + 2D (small square)", "category" => "elementwise"],
                    ["shape" => "1024x1024", "type" => "2D + 2D (medium square)", "category" => "elementwise"],
                    ["shape" => "4096x4096", "type" => "2D + 2D (large square)", "category" => "elementwise"],
                    ["shape" => "32x32768", "type" => "2D + 2D (wide)", "category" => "elementwise"],
                    ["shape" => "32768x32", "type" => "2D + 2D (tall)", "category" => "elementwise"],
                    ["shape" => "32x32x32", "type" => "3D + 3D (small cube)", "category" => "elementwise"],
                    ["shape" => "128x128x128", "type" => "3D + 3D (medium cube)", "category" => "elementwise"],
                    ["shape" => "512x512x3", "type" => "3D + 3D (HWC image)", "category" => "elementwise"],
                    ["shape" => "3x512x512", "type" => "3D + 3D (CHW image)", "category" => "elementwise"],
                    ["shape" => "8x64x64x3", "type" => "4D + 4D (small batch)", "category" => "elementwise"],
                    ["shape" => "32x256x256x3", "type" => "4D + 4D (medium batch)", "category" => "elementwise"],
                    ["shape" => "1024x1024 + scalar", "type" => "2D + scalar", "category" => "broadcast"],
                    ["shape" => "1024x1024 + 1x1024", "type" => "2D + 1D (broadcast dim0)", "category" => "broadcast"],
                    ["shape" => "1024x1024 + 1024x1", "type" => "2D + 1D (broadcast dim1)", "category" => "broadcast"],
                    ["shape" => "1024x3x512 + 1x3x512", "type" => "3D + 3D (broadcast)", "category" => "broadcast"],
                ]
            ],
            [
                "run" => 20,
                "warmup" => true,
                "name" => "CudaArray::subtract()",
                "iterations" => 100,
                "type" => "CUDA",
                "handler" => "cudaArraySubtract",
                "metadata" => [
                    ["shape" => "256", "type" => "1D - 1D (tiny)", "category" => "elementwise"],
                    ["shape" => "65536", "type" => "1D - 1D (small)", "category" => "elementwise"],
                    ["shape" => "16777216", "type" => "1D - 1D (medium)", "category" => "elementwise"],
                    ["shape" => "67108864", "type" => "1D - 1D (large)", "category" => "elementwise"],
                    ["shape" => "16x16", "type" => "2D - 2D (tiny square)", "category" => "elementwise"],
                    ["shape" => "256x256", "type" => "2D - 2D (small square)", "category" => "elementwise"],
                    ["shape" => "1024x1024", "type" => "2D - 2D (medium square)", "category" => "elementwise"],
                    ["shape" => "4096x4096", "type" => "2D - 2D (large square)", "category" => "elementwise"],
                    ["shape" => "32x32768", "type" => "2D - 2D (wide)", "category" => "elementwise"],
                    ["shape" => "32768x32", "type" => "2D - 2D (tall)", "category" => "elementwise"],
                    ["shape" => "32x32x32", "type" => "3D - 3D (small cube)", "category" => "elementwise"],
                    ["shape" => "128x128x128", "type" => "3D - 3D (medium cube)", "category" => "elementwise"],
                    ["shape" => "512x512x3", "type" => "3D - 3D (HWC image)", "category" => "elementwise"],
                    ["shape" => "3x512x512", "type" => "3D - 3D (CHW image)", "category" => "elementwise"],
                    ["shape" => "8x64x64x3", "type" => "4D - 4D (small batch)", "category" => "elementwise"],
                    ["shape" => "32x256x256x3", "type" => "4D - 4D (medium batch)", "category" => "elementwise"],
                    ["shape" => "1024x1024 - scalar", "type" => "2D - scalar", "category" => "broadcast"],
                    ["shape" => "1024x1024 - 1x1024", "type" => "2D - 1D (broadcast dim0)", "category" => "broadcast"],
                    ["shape" => "1024x1024 - 1024x1", "type" => "2D - 1D (broadcast dim1)", "category" => "broadcast"],
                    ["shape" => "1024x3x512 - 1x3x512", "type" => "3D - 3D (broadcast)", "category" => "broadcast"],
                ]
            ],
            [
                "run" => 20,
                "warmup" => true,
                "name" => "CudaArray::multiply()",
                "iterations" => 100,
                "type" => "CUDA",
                "handler" => "cudaArrayMultiply",
                "metadata" => [
                    ["shape" => "256", "type" => "1D * 1D (tiny)", "category" => "elementwise"],
                    ["shape" => "65536", "type" => "1D * 1D (small)", "category" => "elementwise"],
                    ["shape" => "16777216", "type" => "1D * 1D (medium)", "category" => "elementwise"],
                    ["shape" => "67108864", "type" => "1D * 1D (large)", "category" => "elementwise"],
                    ["shape" => "16x16", "type" => "2D * 2D (tiny square)", "category" => "elementwise"],
                    ["shape" => "256x256", "type" => "2D * 2D (small square)", "category" => "elementwise"],
                    ["shape" => "1024x1024", "type" => "2D * 2D (medium square)", "category" => "elementwise"],
                    ["shape" => "4096x4096", "type" => "2D * 2D (large square)", "category" => "elementwise"],
                    ["shape" => "32x32768", "type" => "2D * 2D (wide)", "category" => "elementwise"],
                    ["shape" => "32768x32", "type" => "2D * 2D (tall)", "category" => "elementwise"],
                    ["shape" => "32x32x32", "type" => "3D * 3D (small cube)", "category" => "elementwise"],
                    ["shape" => "128x128x128", "type" => "3D * 3D (medium cube)", "category" => "elementwise"],
                    ["shape" => "512x512x3", "type" => "3D * 3D (HWC image)", "category" => "elementwise"],
                    ["shape" => "3x512x512", "type" => "3D * 3D (CHW image)", "category" => "elementwise"],
                    ["shape" => "8x64x64x3", "type" => "4D * 4D (small batch)", "category" => "elementwise"],
                    ["shape" => "32x256x256x3", "type" => "4D * 4D (medium batch)", "category" => "elementwise"],
                    ["shape" => "1024x1024 * scalar", "type" => "2D * scalar", "category" => "broadcast"],
                    ["shape" => "1024x1024 * 1x1024", "type" => "2D * 1D (broadcast dim0)", "category" => "broadcast"],
                    ["shape" => "1024x1024 * 1024x1", "type" => "2D * 1D (broadcast dim1)", "category" => "broadcast"],
                    ["shape" => "1024x3x512 * 1x3x512", "type" => "3D * 3D (broadcast)", "category" => "broadcast"],
                ]
            ],
            [
                "run" => 20,
                "warmup" => true,
                "name" => "CudaArray::divide()",
                "iterations" => 100,
                "type" => "CUDA",
                "handler" => "cudaArrayDivide",
                "metadata" => [
                    ["shape" => "256", "type" => "1D / 1D (tiny)", "category" => "elementwise"],
                    ["shape" => "65536", "type" => "1D / 1D (small)", "category" => "elementwise"],
                    ["shape" => "16777216", "type" => "1D / 1D (medium)", "category" => "elementwise"],
                    ["shape" => "67108864", "type" => "1D / 1D (large)", "category" => "elementwise"],
                    ["shape" => "16x16", "type" => "2D / 2D (tiny square)", "category" => "elementwise"],
                    ["shape" => "256x256", "type" => "2D / 2D (small square)", "category" => "elementwise"],
                    ["shape" => "1024x1024", "type" => "2D / 2D (medium square)", "category" => "elementwise"],
                    ["shape" => "4096x4096", "type" => "2D / 2D (large square)", "category" => "elementwise"],
                    ["shape" => "32x32768", "type" => "2D / 2D (wide)", "category" => "elementwise"],
                    ["shape" => "32768x32", "type" => "2D / 2D (tall)", "category" => "elementwise"],
                    ["shape" => "32x32x32", "type" => "3D / 3D (small cube)", "category" => "elementwise"],
                    ["shape" => "128x128x128", "type" => "3D / 3D (medium cube)", "category" => "elementwise"],
                    ["shape" => "512x512x3", "type" => "3D / 3D (HWC image)", "category" => "elementwise"],
                    ["shape" => "3x512x512", "type" => "3D / 3D (CHW image)", "category" => "elementwise"],
                    ["shape" => "8x64x64x3", "type" => "4D / 4D (small batch)", "category" => "elementwise"],
                    ["shape" => "32x256x256x3", "type" => "4D / 4D (medium batch)", "category" => "elementwise"],
                    ["shape" => "1024x1024 / scalar", "type" => "2D / scalar", "category" => "broadcast"],
                    ["shape" => "1024x1024 / 1x1024", "type" => "2D / 1D (broadcast dim0)", "category" => "broadcast"],
                    ["shape" => "1024x1024 / 1024x1", "type" => "2D / 1D (broadcast dim1)", "category" => "broadcast"],
                    ["shape" => "1024x3x512 / 1x3x512", "type" => "3D / 3D (broadcast)", "category" => "broadcast"],
                ]
            ],
            [
                "run" => 20,
                "warmup" => true,
                "name" => "CudaArray::power()",
                "iterations" => 100,
                "type" => "CUDA",
                "handler" => "cudaArrayPower",
                "metadata" => [
                    ["shape" => "256", "type" => "1D ** 1D (tiny)", "category" => "elementwise"],
                    ["shape" => "65536", "type" => "1D ** 1D (small)", "category" => "elementwise"],
                    ["shape" => "16777216", "type" => "1D ** 1D (medium)", "category" => "elementwise"],
                    ["shape" => "67108864", "type" => "1D ** 1D (large)", "category" => "elementwise"],
                    ["shape" => "16x16", "type" => "2D ** 2D (tiny square)", "category" => "elementwise"],
                    ["shape" => "256x256", "type" => "2D ** 2D (small square)", "category" => "elementwise"],
                    ["shape" => "1024x1024", "type" => "2D ** 2D (medium square)", "category" => "elementwise"],
                    ["shape" => "4096x4096", "type" => "2D ** 2D (large square)", "category" => "elementwise"],
                    ["shape" => "32x32768", "type" => "2D ** 2D (wide)", "category" => "elementwise"],
                    ["shape" => "32768x32", "type" => "2D ** 2D (tall)", "category" => "elementwise"],
                    ["shape" => "32x32x32", "type" => "3D ** 3D (small cube)", "category" => "elementwise"],
                    ["shape" => "128x128x128", "type" => "3D ** 3D (medium cube)", "category" => "elementwise"],
                    ["shape" => "512x512x3", "type" => "3D ** 3D (HWC image)", "category" => "elementwise"],
                    ["shape" => "3x512x512", "type" => "3D ** 3D (CHW image)", "category" => "elementwise"],
                    ["shape" => "8x64x64x3", "type" => "4D ** 4D (small batch)", "category" => "elementwise"],
                    ["shape" => "32x256x256x3", "type" => "4D ** 4D (medium batch)", "category" => "elementwise"],
                    ["shape" => "1024x1024 ** scalar", "type" => "2D ** scalar", "category" => "broadcast"],
                    ["shape" => "1024x1024 ** 1x1024", "type" => "2D ** 1D (broadcast dim0)", "category" => "broadcast"],
                    ["shape" => "1024x1024 ** 1024x1", "type" => "2D ** 1D (broadcast dim1)", "category" => "broadcast"],
                    ["shape" => "1024x3x512 ** 1x3x512", "type" => "3D ** 3D (broadcast)", "category" => "broadcast"],
                ]
            ],
            [
                "run" => 10,
                "warmup" => true,
                "name" => "CudaArray::gt()",
                "iterations" => 100,
                "type" => "CUDA",
                "handler" => "cudaArrayGt",
                "metadata" => [
                    ["shape" => "1024x1024 > 1024x1024", "type" => "2D > 2D", "category" => "comparison"],
                    ["shape" => "1024x1024 > scalar", "type" => "2D > scalar", "category" => "comparison"],
                    ["shape" => "4096x4096 > 4096x4096", "type" => "2D > 2D (large)", "category" => "comparison"],
                    ["shape" => "32x32x32 > 32x32x32", "type" => "3D > 3D", "category" => "comparison"],
                    ["shape" => "512x512x3 > 512x512x3", "type" => "3D > 3D (HWC)", "category" => "comparison"],
                    ["shape" => "3x512x512 > 3x512x512", "type" => "3D > 3D (CHW)", "category" => "comparison"],
                    ["shape" => "1024x1024 > 1x1024", "type" => "2D > 1D (broadcast)", "category" => "comparison"],
                    ["shape" => "1024x1024 > 1024x1", "type" => "2D > 1D (broadcast)", "category" => "comparison"],
                    ["shape" => "65536 > 65536", "type" => "1D > 1D", "category" => "comparison"],
                    ["shape" => "16777216 > 16777216", "type" => "1D > 1D (medium)", "category" => "comparison"],
                ]
            ],
            [
                "run" => 10,
                "warmup" => true,
                "name" => "CudaArray::lt()",
                "iterations" => 100,
                "type" => "CUDA",
                "handler" => "cudaArrayLt",
                "metadata" => [
                    ["shape" => "1024x1024 < 1024x1024", "type" => "2D < 2D", "category" => "comparison"],
                    ["shape" => "1024x1024 < scalar", "type" => "2D < scalar", "category" => "comparison"],
                    ["shape" => "4096x4096 < 4096x4096", "type" => "2D < 2D (large)", "category" => "comparison"],
                    ["shape" => "32x32x32 < 32x32x32", "type" => "3D < 3D", "category" => "comparison"],
                    ["shape" => "512x512x3 < 512x512x3", "type" => "3D < 3D (HWC)", "category" => "comparison"],
                    ["shape" => "3x512x512 < 3x512x512", "type" => "3D < 3D (CHW)", "category" => "comparison"],
                    ["shape" => "1024x1024 < 1x1024", "type" => "2D < 1D (broadcast)", "category" => "comparison"],
                    ["shape" => "1024x1024 < 1024x1", "type" => "2D < 1D (broadcast)", "category" => "comparison"],
                    ["shape" => "65536 < 65536", "type" => "1D < 1D", "category" => "comparison"],
                    ["shape" => "16777216 < 16777216", "type" => "1D < 1D (medium)", "category" => "comparison"],
                ]
            ],
            [
                "run" => 10,
                "warmup" => true,
                "name" => "CudaArray::eq()",
                "iterations" => 100,
                "type" => "CUDA",
                "handler" => "cudaArrayEq",
                "metadata" => [
                    ["shape" => "1024x1024 == 1024x1024", "type" => "2D == 2D", "category" => "comparison"],
                    ["shape" => "1024x1024 == scalar", "type" => "2D == scalar", "category" => "comparison"],
                    ["shape" => "4096x4096 == 4096x4096", "type" => "2D == 2D (large)", "category" => "comparison"],
                    ["shape" => "32x32x32 == 32x32x32", "type" => "3D == 3D", "category" => "comparison"],
                    ["shape" => "512x512x3 == 512x512x3", "type" => "3D == 3D (HWC)", "category" => "comparison"],
                    ["shape" => "3x512x512 == 3x512x512", "type" => "3D == 3D (CHW)", "category" => "comparison"],
                    ["shape" => "1024x1024 == 1x1024", "type" => "2D == 1D (broadcast)", "category" => "comparison"],
                    ["shape" => "1024x1024 == 1024x1", "type" => "2D == 1D (broadcast)", "category" => "comparison"],
                    ["shape" => "65536 == 65536", "type" => "1D == 1D", "category" => "comparison"],
                    ["shape" => "16777216 == 16777216", "type" => "1D == 1D (medium)", "category" => "comparison"],
                ]
            ],
            [
                "run" => 10,
                "warmup" => true,
                "name" => "CudaArray::ne()",
                "iterations" => 100,
                "type" => "CUDA",
                "handler" => "cudaArrayNe",
                "metadata" => [
                    ["shape" => "1024x1024 != 1024x1024", "type" => "2D != 2D", "category" => "comparison"],
                    ["shape" => "1024x1024 != scalar", "type" => "2D != scalar", "category" => "comparison"],
                    ["shape" => "4096x4096 != 4096x4096", "type" => "2D != 2D (large)", "category" => "comparison"],
                    ["shape" => "32x32x32 != 32x32x32", "type" => "3D != 3D", "category" => "comparison"],
                    ["shape" => "512x512x3 != 512x512x3", "type" => "3D != 3D (HWC)", "category" => "comparison"],
                    ["shape" => "3x512x512 != 3x512x512", "type" => "3D != 3D (CHW)", "category" => "comparison"],
                    ["shape" => "1024x1024 != 1x1024", "type" => "2D != 1D (broadcast)", "category" => "comparison"],
                    ["shape" => "1024x1024 != 1024x1", "type" => "2D != 1D (broadcast)", "category" => "comparison"],
                    ["shape" => "65536 != 65536", "type" => "1D != 1D", "category" => "comparison"],
                    ["shape" => "16777216 != 16777216", "type" => "1D != 1D (medium)", "category" => "comparison"],
                ]
            ],
            [
                "run" => 10,
                "warmup" => true,
                "name" => "CudaArray::le()",
                "iterations" => 100,
                "type" => "CUDA",
                "handler" => "cudaArrayLe",
                "metadata" => [
                    ["shape" => "1024x1024 <= 1024x1024", "type" => "2D <= 2D", "category" => "comparison"],
                    ["shape" => "1024x1024 <= scalar", "type" => "2D <= scalar", "category" => "comparison"],
                    ["shape" => "4096x4096 <= 4096x4096", "type" => "2D <= 2D (large)", "category" => "comparison"],
                    ["shape" => "32x32x32 <= 32x32x32", "type" => "3D <= 3D", "category" => "comparison"],
                    ["shape" => "512x512x3 <= 512x512x3", "type" => "3D <= 3D (HWC)", "category" => "comparison"],
                    ["shape" => "3x512x512 <= 3x512x512", "type" => "3D <= 3D (CHW)", "category" => "comparison"],
                    ["shape" => "1024x1024 <= 1x1024", "type" => "2D <= 1D (broadcast)", "category" => "comparison"],
                    ["shape" => "1024x1024 <= 1024x1", "type" => "2D <= 1D (broadcast)", "category" => "comparison"],
                    ["shape" => "65536 <= 65536", "type" => "1D <= 1D", "category" => "comparison"],
                    ["shape" => "16777216 <= 16777216", "type" => "1D <= 1D (medium)", "category" => "comparison"],
                ]
            ],
            [
                "run" => 10,
                "warmup" => true,
                "name" => "CudaArray::ge()",
                "iterations" => 100,
                "type" => "CUDA",
                "handler" => "cudaArrayGe",
                "metadata" => [
                    ["shape" => "1024x1024 >= 1024x1024", "type" => "2D >= 2D", "category" => "comparison"],
                    ["shape" => "1024x1024 >= scalar", "type" => "2D >= scalar", "category" => "comparison"],
                    ["shape" => "4096x4096 >= 4096x4096", "type" => "2D >= 2D (large)", "category" => "comparison"],
                    ["shape" => "32x32x32 >= 32x32x32", "type" => "3D >= 3D", "category" => "comparison"],
                    ["shape" => "512x512x3 >= 512x512x3", "type" => "3D >= 3D (HWC)", "category" => "comparison"],
                    ["shape" => "3x512x512 >= 3x512x512", "type" => "3D >= 3D (CHW)", "category" => "comparison"],
                    ["shape" => "1024x1024 >= 1x1024", "type" => "2D >= 1D (broadcast)", "category" => "comparison"],
                    ["shape" => "1024x1024 >= 1024x1", "type" => "2D >= 1D (broadcast)", "category" => "comparison"],
                    ["shape" => "65536 >= 65536", "type" => "1D >= 1D", "category" => "comparison"],
                    ["shape" => "16777216 >= 16777216", "type" => "1D >= 1D (medium)", "category" => "comparison"],
                ]
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "CudaArray::neg()",
                "iterations" => 100,
                "type" => "CUDA",
                "handler" => "cudaArrayNeg",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "CudaArray::floor()",
                "iterations" => 100,
                "type" => "CUDA",
                "handler" => "cudaArrayFloor",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "CudaArray::ceil()",
                "iterations" => 100,
                "type" => "CUDA",
                "handler" => "cudaArrayCeil",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "CudaArray::round()",
                "iterations" => 100,
                "type" => "CUDA",
                "handler" => "cudaArrayRound",
                "metadata" => $this->unaryMetadata()
            ],
        ];
    }

    private function unaryMetadata(): array
    {
        return [
            ["shape" => "256", "type" => "1D (tiny)", "category" => "unary"],
            ["shape" => "65536", "type" => "1D (small)", "category" => "unary"],
            ["shape" => "16777216", "type" => "1D (medium)", "category" => "unary"],
            ["shape" => "1024x1024", "type" => "2D (medium)", "category" => "unary"],
            ["shape" => "4096x4096", "type" => "2D (large)", "category" => "unary"],
            ["shape" => "512x512x3", "type" => "3D (HWC)", "category" => "unary"],
            ["shape" => "3x512x512", "type" => "3D (CHW)", "category" => "unary"],
        ];
    }

    public function description(): string
    {
        return "CudaArray basic operation methods Benchmark";
    }

    public function argsArithmetic(int $count): array
    {
        return match ($count) {
            1 => [CudaArray::rand([256]), CudaArray::rand([256])],
            2 => [CudaArray::rand([65536]), CudaArray::rand([65536])],
            3 => [CudaArray::rand([16777216]), CudaArray::rand([16777216])],
            4 => [CudaArray::rand([67108864]), CudaArray::rand([67108864])],
            
            5 => [CudaArray::rand([16, 16]), CudaArray::rand([16, 16])],
            6 => [CudaArray::rand([256, 256]), CudaArray::rand([256, 256])],
            7 => [CudaArray::rand([1024, 1024]), CudaArray::rand([1024, 1024])],
            8 => [CudaArray::rand([4096, 4096]), CudaArray::rand([4096, 4096])],
            
            9 => [CudaArray::rand([32, 32768]), CudaArray::rand([32, 32768])],
            10 => [CudaArray::rand([32768, 32]), CudaArray::rand([32768, 32])],
            
            11 => [CudaArray::rand([32, 32, 32]), CudaArray::rand([32, 32, 32])],
            12 => [CudaArray::rand([128, 128, 128]), CudaArray::rand([128, 128, 128])],
            
            13 => [CudaArray::rand([512, 512, 3]), CudaArray::rand([512, 512, 3])],
            14 => [CudaArray::rand([3, 512, 512]), CudaArray::rand([3, 512, 512])],
            
            15 => [CudaArray::rand([8, 64, 64, 3]), CudaArray::rand([8, 64, 64, 3])],
            16 => [CudaArray::rand([32, 256, 256, 3]), CudaArray::rand([32, 256, 256, 3])],
            
            17 => [CudaArray::rand([1024, 1024]), 2.0], 
            18 => [CudaArray::rand([1024, 1024]), CudaArray::rand([1, 1024])],
            19 => [CudaArray::rand([1024, 1024]), CudaArray::rand([1024, 1])],
            20 => [CudaArray::rand([1024, 3, 512]), CudaArray::rand([1, 3, 512])],
            
            default => throw new \InvalidArgumentException("Invalid count: $count")
        };
    }

    public function argsComparison(int $count): array
    {
        return match ($count) {
            1 => [CudaArray::rand([1024, 1024]), CudaArray::rand([1024, 1024])],
            2 => [CudaArray::rand([1024, 1024]), 0.5],
            3 => [CudaArray::rand([4096, 4096]), CudaArray::rand([4096, 4096])],
            4 => [CudaArray::rand([32, 32, 32]), CudaArray::rand([32, 32, 32])],
            5 => [CudaArray::rand([512, 512, 3]), CudaArray::rand([512, 512, 3])],
            6 => [CudaArray::rand([3, 512, 512]), CudaArray::rand([3, 512, 512])],
            7 => [CudaArray::rand([1024, 1024]), CudaArray::rand([1, 1024])],
            8 => [CudaArray::rand([1024, 1024]), CudaArray::rand([1024, 1])],
            9 => [CudaArray::rand([65536]), CudaArray::rand([65536])],
            10 => [CudaArray::rand([16777216]), CudaArray::rand([16777216])],
            
            default => throw new \InvalidArgumentException("Invalid count: $count")
        };
    }

    public function argsUnary(int $count): array
    {
        return match ($count) {
            1 => [CudaArray::rand([256])],
            2 => [CudaArray::rand([65536])],
            3 => [CudaArray::rand([16777216])],
            4 => [CudaArray::rand([1024, 1024])],
            5 => [CudaArray::rand([4096, 4096])],
            6 => [CudaArray::rand([512, 512, 3])],
            7 => [CudaArray::rand([3, 512, 512])],
            
            default => throw new \InvalidArgumentException("Invalid count: $count")
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
    public function cudaArrayPower(CudaArray $first, CudaArray|float $second): void
    {
        $first ** $second;
    }

    #[InjectArgs("argsComparison")]
    public function cudaArrayGt(CudaArray $first, CudaArray|float $second): void
    {
        $first->gt($second);
    }

    #[InjectArgs("argsComparison")]
    public function cudaArrayLt(CudaArray $first, CudaArray|float $second): void
    {
        $first->lt($second);
    }

    #[InjectArgs("argsComparison")]
    public function cudaArrayEq(CudaArray $first, CudaArray|float $second): void
    {
        $first->eq($second);
    }

    #[InjectArgs("argsComparison")]
    public function cudaArrayNe(CudaArray $first, CudaArray|float $second): void
    {
        $first->ne($second);
    }

    #[InjectArgs("argsComparison")]
    public function cudaArrayLe(CudaArray $first, CudaArray|float $second): void
    {
        $first->le($second);
    }

    #[InjectArgs("argsComparison")]
    public function cudaArrayGe(CudaArray $first, CudaArray|float $second): void
    {
        $first->ge($second);
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