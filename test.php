<?php

use Cuda\CudaArray;
use Cuda\Attr as Attr;

use function Cuda\Builtins\{threadIdx};


$compiler = new Cuda\Compiler;

$device = Cuda\Device::fn(
    #[Cuda\Attr\Device(name: 'device_test')]
    function (float $a, int $b): float|int {
        return $a + $b;
    }
);

$compiler->kernel(
    #[Cuda\Attr\Kernel(name: 'kernel_test')]
    function ($a, $b) use ($device): void {
        $b = $device($b);
    }
);

var_dump($compiler->compile()->getKernels());

