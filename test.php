<?php

use Cuda\CudaArray;

$ca = CudaArray::ones([3, 6, 2], dtype: 'float');
$ca2 = CudaArray::ones([6, 2]);

var_dump($ca[0][0]->toHost()->toArray());