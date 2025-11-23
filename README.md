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
 * Creates a CudaArray with a 4×4×4 shape filled with ones.
 */
$ca = CudaArray::ones([4, 4, 4]);

/**
 * Performs:  (ca[1] * 2) + ca[2]
 * Both slices have shape 4×4.
 */
$result = ($ca[1] * 2) + $ca[2];

/**
 * Assigns the result to index 0.
 * The overall tensor shape remains 4×4×4.
 */
$ca[0] = $result;

/**
 * Get tensor shape.
 */
[$x, $y, $z] = $ca->getShape();

/**
 * Reshape into a flat 1D tensor of size 64.
 */
$newCa = $ca->reshape([$x * $y * $z]);

/**
 * - Creates a view/window from indices 0 to 4 (no new GPU memory allocated)
 * - clone() then forces materialization (new GPU tensor)
 */
$newCa = clone $newCa([0, 4]);

/**
 * Transfer the result back to CPU as a PHP array.
 *
 * Output example:
 * array(5) {
 *   [0] => float(3)
 *   [1] => float(3)
 *   [2] => float(3)
 *   [3] => float(3)
 *   [4] => float(3)
 * }
 */
var_dump($newCa->toArray());

```

## Methods
### Basic math
All methods list bellow accept an scalar value or a CudaArray instance, the shape is broadcasted automatically

```php
// Multiplication
$ca->multiply($x);
$ca * $x;

// Addition
$ca->add($x);
$ca + $x;

// Division
$ca->divide($x);
$ca / $x;

// Subtraction
$ca->subtract($x);
$ca - $x;

// Power
$ca->power($x);
$ca ** $x;

// Exponential / Square Root / Logarithm
$ca->exp();
$ca->sqrt();
$ca->log();

// Trigonometry
$ca->cos();
$ca->sin();
$ca->tan();

// others
$ca->neg(); 
```

### Getters

```php
$ca->toArray();     // Transfer tensor to CPU as nested PHP array
$ca->getShape();    // Returns shape (array of ints)
$ca->getStrides();  // Returns memory strides (array of ints)
```

### New Instance

```php
# Notice: when using the constructor, the PHP array is transferred from CPU → GPU
$ca = new CudaArray([[1, 2], [3, 4]]);

# Creates a tensor directly on the GPU, without transferring data from PHP
$ca = CudaArray::ones($shape);
$ca = CudaArray::zeros($shape);
$ca = CudaArray::full($shape, 1.5);
```

### Shape Manipulation

```php
$ca->reshape([4, 4, 4]);
$ca->flatten(); // Same as reshape([n])
```

### Comparing
All comparison methods return a new CudaArray stored on the GPU, containing 1.0 for true and 0.0 for false.
They accept either:
 - a scalar, or another CudaArray (broadcasting is automatically applied)
```php
$x->gt($y);   // greater than      (x > y)
$y->lt($x);   // less than         (x < y)
$x->eq($y);   // equal             (x == y)
$x->ne($y);   // not equal         (x != y)
$x->ge($y);   // greater or equal  (x >= y)
$x->le($y);   // less or equal     (x <= y)
```

