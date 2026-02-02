<?php

use Cuda\CudaArray;

$ca = CudaArray::ones([3, 6, 2]);
$ca2 = CudaArray::ones([6, 2]);

$test = $ca + $ca2;

var_dump($test->toArray());
