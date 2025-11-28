--TEST--
Cuda Array Sum reduce
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip');
?>
--FILE--
<?php
$ca = new CudaArray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
var_dump($ca->sum(1)->toArray());
var_dump($ca->sum(0)->toArray());
var_dump($ca->sum(2)->toArray());
var_dump($ca->sum()->toArray());
?>
--EXPECT--
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(4)
    [1]=>
    float(6)
  }
  [1]=>
  array(2) {
    [0]=>
    float(12)
    [1]=>
    float(14)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(6)
    [1]=>
    float(8)
  }
  [1]=>
  array(2) {
    [0]=>
    float(10)
    [1]=>
    float(12)
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(3)
    [1]=>
    float(7)
  }
  [1]=>
  array(2) {
    [0]=>
    float(11)
    [1]=>
    float(15)
  }
}
array(1) {
  [0]=>
  float(36)
}
