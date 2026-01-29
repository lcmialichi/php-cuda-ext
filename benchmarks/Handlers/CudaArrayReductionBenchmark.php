<?php

namespace Benchmarks\Handlers;

use Cuda\CudaArray;
use Benchmarks\Handlers\Benchmark;
use Benchmarks\Support\Attr\InjectArgs;

class CudaArrayReductionBenchmark extends Benchmark
{
    public function name(): string
    {
        return "CudaArray Reduction Benchmark";
    }

    public function register(): array
    {
        return [
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

    public function description(): string
    {
        return "CudaArray reduction methods Benchmark";
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
