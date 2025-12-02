<?php

$test = Cuda\Kernel::fusion(function () {
    $rand0 = Cuda\CudaArray::rand([4, 4], -1, 1);
    $rand1 = Cuda\CudaArray::rand([4, 4], -1, 1);
    $rand2 = Cuda\CudaArray::rand([4, 4], -1, 1);
    $rand3 = Cuda\CudaArray::rand([1, 4], -1, 1);
    $rand4 = Cuda\CudaArray::rand([1, 4], -1, 1);

    $tensor = ($rand0 * $rand1) / ($rand2 ** $rand3);
    return $tensor * 3 /15  * $rand4;
});

var_dump($test);
exit;
