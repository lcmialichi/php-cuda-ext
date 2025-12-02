<?php

$test = Cuda\Kernel::fusion(function () {
    $ca = Cuda\CudaArray::rand([4, 4], -1, 1);
    $ca2 = Cuda\CudaArray::rand([4, 4], -1, 1);

    return ($ca * $ca2 + $ca2) / $ca;
});

var_dump($test);
exit;
