<?php

use Cuda\CudaArray;
use Cuda\Attr as Attr;

$device = Cuda\Kernel::fn(function () {
    return 1;
});

$compiler = new Cuda\Compiler;
// var_dump($compiler);

$stauts = $compiler->kernel(
    #[Cuda\Attr\Kernel(name: 'kernel1')]
    function ($a, $b) {
        return $a + $b;
    }
);

$status2 = $compiler->kernel(
    #[Cuda\Attr\Kernel(name: 'kernel2')]
    fn($a) => $a
);




var_dump($compiler->getKernels());
