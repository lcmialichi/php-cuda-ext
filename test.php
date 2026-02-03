<?php

use Cuda\CudaArray;

$ca = CudaArray::rand([32, 10]);
$ca2 = CudaArray::rand([32, 32768]);

var_dump($ca[0]->toArray());