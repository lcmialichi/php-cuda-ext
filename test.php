<?php

use Cuda\CudaArray;

$ca = new  CudaArray([1000, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0], dtype: "bool");
$ca2 = new CudaArray([1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0], dtype: "float");

// $ca = $ca->astype($ca2->dtype());

$test = $ca * 10 ;
var_dump($ca, $ca->toArray());
