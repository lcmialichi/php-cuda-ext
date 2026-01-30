<?php

use Cuda\CudaArray;

$ca = new  CudaArray([1, 0, 1, 0], dtype: "bool");
$ca2 = new CudaArray([0, 1, 0, 1], dtype: "int");

// $result = $ca + $ca2;

// var_dump($result, $ca->toArray());