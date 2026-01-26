<?php

namespace Benchmarks\Handlers;

use Cuda\CudaArray;
use Benchmarks\Handlers\Benchmark;

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
            ]
        ];
    }

    public function description(): string
    {
        return "testing description";
    }

    public function argsCudaArrayOnes(int $count): array
    {
        $size = pow(8, $count);
        return [[$size, $size, $size]];
    }

    public function argsCudaArrayFull(int $count): array
    {
        $size = pow(8, $count);
        return [[$size, $size, $size], $count * 2];
    }

    public function argsCudaArrayRand(int $count): array
    {
        $size = pow(8, $count);
        return [[$size, $size, $size], -1, $count * 10];
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

    public function cudaArrayOnes(array $shape): void
    {
        CudaArray::ones($shape);
    }

    public function cudaArrayFull(array $shape, int $value): void
    {
        CudaArray::full($shape, $value);
    }

    public function cudaArrayRand(array $shape, int $value): void
    {
        CudaArray::full($shape, $value);
    }

    public function cudaArrayConcatAxisZero(CudaArray $first, CudaArray $second, int $axis): void
    {
        $first->concat([$second], $axis);
    }

    public function cudaArrayConcatAxisOne(CudaArray $first, CudaArray $second, int $axis): void
    {
        $first->concat([$second], $axis);
    }
}
