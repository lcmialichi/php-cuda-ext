<?php

$test = Cuda\Kernel::fusion(function () {
    $ca = Cuda\CudaArray::rand([4, 4], -1, 1);
    $ca2 = Cuda\CudaArray::rand([4, 4], -1, 1);

    $tensor = (($ca * ($ca2 + $ca2)) / $ca) ** Cuda\CudaArray::ones([4]);

    return $tensor * 3 / 15;
});

var_dump($test);
exit;
