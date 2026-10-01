--TEST--
CudaArray reports GPU memory budget exhaustion
--INI--
cuda.memory_size=1M
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip cuda extension unavailable');
if (cuda_get_device_count() < 1) die('skip CUDA device unavailable');
?>
--FILE--
<?php
try {
    Cuda\CudaArray::ones([300000]);
    echo "unexpected success\n";
} catch (Throwable $error) {
    echo "allocation rejected\n";
}
?>
--EXPECT--
allocation rejected