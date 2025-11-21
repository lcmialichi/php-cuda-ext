### PHP CUDA Extension

A native PHP extension that provides direct access to NVIDIA CUDA functionality, enabling high-performance GPU computing inside PHP applications.

# ⚠️ NOTICE — Experimental Project

This extension is **actively under development**.  
It is **not production-ready**, may contain bugs, and its API may change at any time.  
Use **only in testing or experimental environments**.

---

### Requirements
To build and run this extension, you need:

- NVIDIA CUDA Toolkit (12.x recommended)
- Compatible NVIDIA GPU driver
- PHP 8.0+ with support for C extensions
- gcc / g++
- make / autoconf
- Linux (Ubuntu, Debian, CentOS, Arch, etc.)


### How to Compile
```bash
git clone https://github.com/lcmialichi/php-cuda-ext.git/
cd php-cuda-ext
```
Compile and install the extension by running:
```bash
./compile.sh
```

The compile script automatically performs:
1. phpize
2. ./configure
3. make
4. make install

Verify that it loaded correctly:
```bash
php -m | grep cuda
```


### Quick start

```php
/**
 * creates a CudaArray with shape of 4x4x4 full of ones
 */
$ca = CudaArray::ones([4, 4, 4]);

/**
 *  idx 1 * 2 (4x4), and sum with idx 2 (4x4)
 * @var CudaArray
 */
$result = ($ca[1] * 2) + $ca[2];

/**
 * set at idx 0 the result (the shape remains 4x4x4)
 * @var CudaArray
 */
$ca[0] = $result;

/**
 * get shape from the matrix
 */
[$x, $y, $z] = $ca->getShape();

/**
 * reshape as 1x64
 */
$newCa = $ca->reshape([$x * $y * $z]);

/**
 * creates a window of indices 0 to 4 (does not create a new tensor in memory)
 */
$newCa = clone $newCa([0, 4]);

/**
 * return to CPU as an Array
 * array(5) {
  * [0]=>
  * float(3)
  * [1]=>
  * float(3)
  * [2]=>
  * float(3)
  * [3]=>
  * float(3)
  * [4]=>
  * float(3)
*}

 */
var_dump($newCa->toArray());
```


