<?php

use Cuda\CudaArray;

$ca = CudaArray::rand([3, 6, 2], min: 1, max: 2, dtype: 'float32');
$cb = CudaArray::rand([2]);

$ca * $cb;

$ca2 = CudaArray::ones([6, 2]);