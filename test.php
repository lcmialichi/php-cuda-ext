<?php

$start2 = microtime(true);

$ones = Cuda\CudaArray::ones([1024, 516, 32]);
$fours = Cuda\CudaArray::rand([1024, 516, 32]);

$temp1 = ($ones * $fours) / 11;
$temp2 = ($temp1 * $ones) ** 3;
$temp3 = ($temp2 * 2) / 1;
$temp4 = ($temp3 * $fours) / 1;
$temp5 = ($temp4 * 2) / 1;
$temp6 = ($temp5 * $fours) / 1;
$temp7 = ($temp6 * 2) / 1;
$temp8 = ($temp7 * $fours) / 1;
$temp8->sqrt() ** 2;

$time2 = round(microtime(true) - $start2, 3);

$start1 = microtime(true);
$tensor = Cuda\Kernel::fusion(function () {
    $ones = Cuda\CudaArray::ones([1024, 516, 32]);
    $fours = Cuda\CudaArray::rand([1024, 516, 32]);

    $temp1 = ($ones * $fours) / 11;
    $temp2 = ($temp1 * $ones) ** 3;
    $temp3 = ($temp2 * 2) / 1;
    $temp4 = ($temp3 * $fours) / 1;
    $temp5 = ($temp4 * 2) / 1;
    $temp6 = ($temp5 * $fours) / 1;
    $temp7 = ($temp6 * 2) / 1;
    $temp8 = ($temp7 * $fours) / 1;
    return $temp8->sqrt() ** 2;
});

$time1 = round(microtime(true) - $start1, 3);

var_dump([
    'NOT_FUSED' => $time2,
    'FUSED' => $time1
]);

exit;



