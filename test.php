<?php

use Cuda\CudaArray;

// $ca = new  CudaArray([1, 0, 1, 0], dtype: "bool");
// $ca2 = new CudaArray([0, 1, 0, 1], dtype: "int");

$test = CudaArray::rand([10, 10]);

var_dump($test->toArray());