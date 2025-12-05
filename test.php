<?php

$tensor = Cuda\Kernel::fusion(function () {
    $rand0 = Cuda\CudaArray::rand([4, 4], -1, 1);
    $rand1 = Cuda\CudaArray::rand([4, 4], -1, 1);
    $t0 = $rand0 + 10 * 5 / 11;

    $t2 = $rand0 + 100;
    $t3 = ($t0 * 4);
    $t4 = $t2 / ($rand1 * 2);

    return $t4->sqrt();
});

// $tensor->getShape();
exit;
