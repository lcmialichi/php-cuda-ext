<?php

use Cuda\CudaArray;

$cuda = new CudaArray([[[1, 2, 3], [3, 4, 5]], [[5, 6, 7], [7, 8, 9]], [[10, 11, 12], [13, 14, 15]]]);
$host = $cuda->toHost();
$arr = [];

foreach ($host as $i => $value) {
    foreach ($value as $j => $subvalue) {
       var_dump($subvalue->getShape());
    }
}