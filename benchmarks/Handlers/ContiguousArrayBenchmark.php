<?php

namespace Benchmarks\Handlers;

use Cuda\CudaArray;
use Benchmarks\Handlers\Benchmark;
use Benchmarks\Support\Attr\InjectArgs;
use Cuda\ContiguousArray;

class ContiguousArrayBenchmark extends Benchmark
{
    public function name(): string
    {
        return "ContiguousArray Benchmark";
    }

    public function register(): array
    {
        return [
            [
                "run" => 7,
                "warmup" => true,
                "name" => "ContiguousArray::toArray() [ContiguousArray -> PHP Array]",
                "iterations" => 10,
                "type" => "HOST",
                "handler" => "contiguousArrayToArray",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "ContiguousArray::toGpu() [Host -> GPU]",
                "iterations" => 10,
                "type" => "TRANSFER",
                "handler" => "contiguousArrayToGpu",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "ContiguousArray::get() [Element Access via array]",
                "iterations" => 10,
                "type" => "HOST",
                "handler" => "contiguousArrayGet",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "ContiguousArray::at() [Element Access via variadic]",
                "iterations" => 10,
                "type" => "HOST",
                "handler" => "contiguousArrayAt",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "ContiguousArray::getSize()",
                "iterations" => 10,
                "type" => "HOST",
                "handler" => "contiguousArrayGetSize",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "ContiguousArray::getNdims()",
                "iterations" => 10,
                "type" => "HOST",
                "handler" => "contiguousArrayGetNdims",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "ContiguousArray::getDtype()",
                "iterations" => 10,
                "type" => "HOST",
                "handler" => "contiguousArrayGetDtype",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "ContiguousArray::getElementSize()",
                "iterations" => 10,
                "type" => "HOST",
                "handler" => "contiguousArrayGetElementSize",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "ContiguousArray::count()",
                "iterations" => 10,
                "type" => "HOST",
                "handler" => "contiguousArrayCount",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "ContiguousArray::getShape()",
                "iterations" => 10,
                "type" => "HOST",
                "handler" => "contiguousArrayGetShape",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "ContiguousArray Iteration (foreach axis 0)",
                "iterations" => 10,
                "type" => "HOST",
                "handler" => "contiguousArrayIteration",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "ContiguousArray ArrayAccess [Single Dimension]",
                "iterations" => 10,
                "type" => "HOST",
                "handler" => "contiguousArrayArrayAccessSingle",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "ContiguousArray ArrayAccess [Multi Dimension]",
                "iterations" => 10,
                "type" => "HOST",
                "handler" => "contiguousArrayArrayAccessMulti",
                "metadata" => $this->arrayAccessMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "ContiguousArray::__serialize()",
                "iterations" => 10,
                "type" => "HOST",
                "handler" => "contiguousArraySerialize",
                "metadata" => $this->unaryMetadata()
            ],
            [
                "run" => 7,
                "warmup" => true,
                "name" => "ContiguousArray::__unserialize()",
                "iterations" => 10,
                "type" => "HOST",
                "handler" => "contiguousArrayUnserialize",
                "metadata" => $this->unaryMetadata()
            ],
        ];
    }

    public function description(): string
    {
        return "ContiguousArray methods Benchmark";
    }

    private function unaryMetadata(): array
    {
        return [
            ["shape" => "16x16x16", "type" => "3D"],
            ["shape" => "64x64x64", "type" => "3D"],
            ["shape" => "512x512x64", "type" => "3D"],
            ["shape" => "512x512", "type" => "3D"],
            ["shape" => "1024x512", "type" => "2D"],
            ["shape" => "1024x512", "type" => "2D"],
            ["shape" => "1x180000", "type" => "2D"],
            ["shape" => "180000", "type" => "1D"],
        ];
    }

    private function arrayAccessMetadata(): array
    {
        return [
            ["shape" => "16x16x16", "indices" => "[0,0,0]", "type" => "3D"],
            ["shape" => "64x64x64", "indices" => "[32,32,32]", "type" => "3D"],
            ["shape" => "1024x512x32", "indices" => "[512,256,16]", "type" => "3D"],
            ["shape" => "512x512x512", "indices" => "[256,256,256]", "type" => "3D"],
            ["shape" => "512x512", "indices" => "[256,256]", "type" => "2D"],
            ["shape" => "1024x512", "indices" => "[512,256]", "type" => "2D"],
            ["shape" => "1x180000", "indices" => "[0,90000]", "type" => "2D"],
        ];
    }

    public function argsTransfer(int $count): array
    {
        return match ($count) {
            1 => [CudaArray::rand([16, 16, 16])->toHost()],
            2 => [CudaArray::rand([64, 64, 64])->toHost()],
            3 => [CudaArray::rand([1024, 512, 32])->toHost()],
            4 => [CudaArray::rand([512, 512])->toHost()],
            5 => [CudaArray::rand([1024, 512])->toHost()],
            6 => [CudaArray::rand([1, 180000])->toHost()],
            7 => [CudaArray::rand([180000])->toHost()],
        };
    }

    public function argsTransferAt(int $count): array
    {
        return match ($count) {
            1 => [CudaArray::rand([16, 16, 16])->toHost(), [8, 8, 8]],
            2 => [CudaArray::rand([64, 64, 64])->toHost(), [32, 32, 32]],
            3 => [CudaArray::rand([1024, 512, 32])->toHost(), [512, 256, 16]],
            4 => [CudaArray::rand([512, 512])->toHost(), [256, 256]],
            5 => [CudaArray::rand([1024, 512])->toHost(), [512, 256]],
            6 => [CudaArray::rand([1, 180000])->toHost(), [0, 90000]],
            7 => [CudaArray::rand([180000])->toHost(), [90000]],
        };
    }

    public function argsArrayAccessMulti(int $count): array
    {
        return match ($count) {
            1 => [CudaArray::rand([16, 16, 16])->toHost(), 0, 0, 0],
            2 => [CudaArray::rand([64, 64, 64])->toHost(), 32, 32, 32],
            3 => [CudaArray::rand([1024, 512, 32])->toHost(), 512, 256, 16],
            4 => [CudaArray::rand([512, 512, 512])->toHost(), 256, 256, 256],
            5 => [CudaArray::rand([512, 512])->toHost(), 256, 256],
            6 => [CudaArray::rand([1024, 512])->toHost(), 512, 256],
            7 => [CudaArray::rand([1, 180000])->toHost(), 0, 90000],
        };
    }

    public function argsSerialize(int $count): array
    {
        $array = match ($count) {
            1 => CudaArray::rand([16, 16, 16])->toHost(),
            2 => CudaArray::rand([64, 64, 64])->toHost(),
            3 => CudaArray::rand([512, 512, 64])->toHost(),
            4 => CudaArray::rand([512, 512])->toHost(),
            5 => CudaArray::rand([1024, 512])->toHost(),
            6 => CudaArray::rand([1, 180000])->toHost(),
            7 => CudaArray::rand([180000])->toHost(),
        };

        return [serialize($array)];
    }

    #[InjectArgs("argsTransfer")]
    public function contiguousArrayToArray(ContiguousArray $host): void
    {
        $host->toArray();
    }

    #[InjectArgs("argsTransfer")]
    public function contiguousArrayToGpu(ContiguousArray $host): void
    {
        $host->toGpu();
    }

    #[InjectArgs("argsTransferAt")]
    public function contiguousArrayGet(ContiguousArray $host, array $idxs): void
    {
        $host->get($idxs);
    }

    #[InjectArgs("argsTransferAt")]
    public function contiguousArrayAt(ContiguousArray $host, array $idxs): void
    {
        call_user_func_array([$host, 'at'], $idxs);
    }

    #[InjectArgs("argsTransfer")]
    public function contiguousArrayGetSize(ContiguousArray $host): void
    {
        $host->getSize();
    }

    #[InjectArgs("argsTransfer")]
    public function contiguousArrayGetNdims(ContiguousArray $host): void
    {
        $host->getNdims();
    }

    #[InjectArgs("argsTransfer")]
    public function contiguousArrayGetDtype(ContiguousArray $host): void
    {
        $host->getDtype();
    }

    #[InjectArgs("argsTransfer")]
    public function contiguousArrayGetElementSize(ContiguousArray $host): void
    {
        $host->getElementSize();
    }

    #[InjectArgs("argsTransfer")]
    public function contiguousArrayCount(ContiguousArray $host): void
    {
        count($host);
    }

    #[InjectArgs("argsTransfer")]
    public function contiguousArrayGetShape(ContiguousArray $host): void
    {
        $host->getShape();
    }

    #[InjectArgs("argsTransfer")]
    public function contiguousArrayIteration(ContiguousArray $host): void
    {
        foreach ($host as $item) {
            // Just iterate, no operation needed
        }
    }

    #[InjectArgs("argsTransfer")]
    public function contiguousArrayArrayAccessSingle(ContiguousArray $host): void
    {
        $dummy = $host[0];
    }

    #[InjectArgs("argsArrayAccessMulti")]
    public function contiguousArrayArrayAccessMulti(ContiguousArray $host, ...$indices): void
    {
        $current = $host;
        foreach ($indices as $index) {
            $current = $current[$index];
        }
    }

    #[InjectArgs("argsTransfer")]
    public function contiguousArraySerialize(ContiguousArray $host): void
    {
        serialize($host);
    }

    #[InjectArgs("argsSerialize")]
    public function contiguousArrayUnserialize(string $data): void
    {
        unserialize($data);
    }
}
