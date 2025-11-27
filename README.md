### PHP CUDA Extension

A native PHP extension that provides direct access to NVIDIA CUDA functionality, enabling high-performance GPU computing inside PHP applications.

<div align="center">
  <img src="https://i.imgur.com/EqIDkTp.gif" width="300px" height="170px" alt="GIF demonstrativo">
</div>

<div align="center">
  <img src="https://img.shields.io/badge/PHP-8.0+-purple?logo=php">
  <img src="https://img.shields.io/badge/CUDA-11.0%2B-76B900?logo=nvidia">  
  <img src="https://img.shields.io/badge/License-MIT-blue">
  <img src="https://img.shields.io/badge/Platform-Linux-red">
</div>

# ⚠️ NOTICE — Under Development

This extension is **actively under development**.  
It is **not production-ready**, may contain bugs, and its API may change at any time.  
Use **only in testing or experimental environments**.

---

## Requirements
To build and run this extension, you need:

- NVIDIA CUDA Toolkit (12.x recommended)
- Compatible NVIDIA GPU driver
- PHP 8.0+ with support for C extensions
- gcc / g++
- make / autoconf
- Linux (Ubuntu, Debian, CentOS, Arch, etc.)


## How to Compile
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

## Quick start
### Operator Overloading
CudaArray supports native PHP operator overloading, providing an intuitive syntax for GPU-accelerated tensor operations. This allows you to write mathematical expressions that look like standard PHP code but execute entirely on the NVIDIA GPU.
```php
$a = CudaArray::ones([3, 3]);
$b = CudaArray::full([3, 3], 2.0);
$scalar = 5.0;

// Addition
$result = $a + $b;      // Element-wise addition
$result = $a + $scalar; // Broadcasting: adds 5.0 to every element
$result = $a++;        // same as $a + 1 

// Subtraction  
$result = $a - $b;      // Element-wise subtraction
$result = $a - 2.0;     // Broadcasting: subtracts 2.0 from every element
$result = $a--;         // same as $a - 1 

// Multiplication
$result = $a * $b;      // Element-wise multiplication (Hadamard product)
$result = $a * 3.0;     // Broadcasting: multiplies every element by 3.0

// Division
$result = $a / $b;      // Element-wise division
$result = $a / 2.0;     // Broadcasting: divides every element by 2.0

// Exponentiation
$result = $a ** $b;     // Element-wise power: aᵢⱼ ^ bᵢⱼ
$result = $a ** 2;      // Broadcasting: squares every element
```
Operator overloading enables complex mathematical expressions that execute efficiently on the ``GPU``:

```php
// Complex GPU-accelerated expression
$result = ($a * 2.0 + $b) ** ($c / 3.0) - $d;

// Equivalent to:
$temp1 = $a->multiply(2.0);
$temp2 = $temp1->add($b);
$temp3 = $c->divide(3.0);
$temp4 = $temp2->power($temp3);
$result = $temp4->subtract($d);
```

### Basic Tensor Operations
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
All operations support automatic shape broadcasting and accept both CudaArray instances and scalar values.

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
$ca = CudaArray::rand($shape, 0, 10);
```

### Shape Manipulation

```php
$ca->reshape([4, 4, 4]);
$ca->flatten(); // Same as reshape([n])
$ca->concat([$a, $b, $c], axis: null);
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

### Reduction
Reduction operations collapse one or more axes of a tensor into a smaller shape.
All reduction methods accept:
 - axis (optional)
 - Positive or negative axis indices
 - If axis is not specified, the reduction is applied to the entire tensor, returning a tensor with shape [1].
```php
$x->argMax(axis: null); // Returns the index of the maximum value along the specified axis.
$x->argMin(axis: null); // Same behavior as argMax, but finds the index of the minimum value. flatten
$x->sum(axis: null); // Computes the sum along the given axis.
$x->min(axis: null); // Computes the minimum value along the axis.
$x->max(axis: null); // Computes the maximum value along the axis.
$x->prod(axis: null); // Computes the product of all elements along the axis.
```

## Run Benchmark
You can run benchmark script to see real execution time
```bash
php benchmark.php
```

## Use Cases
 - Machine Learning & AI: GPU-accelerated model inference and preprocessing
 - Data Science & Analytics: Large-scale numerical computations
 - Image & Video Processing: Real-time filtering and transformations
 - Scientific Computing: Complex mathematical simulations
 - Game Development: Physics engines and procedural generation
 - Financial Modeling: Risk analysis and quantitative finance

 ## Contributing
 We welcome contributions from the community!
 
 ##  License
 This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

 ## Acknowledgments
  - NVIDIA for the CUDA parallel computing platform
  -  PHP internals developers and community
  - Contributors and early testers

## Support
Star this repository if you find it interesting!

Follow development progress and report issues on GitHub

Keywords: PHP CUDA extension, GPU computing PHP, NVIDIA PHP, tensor operations, machine learning PHP, high-performance computing, GPU acceleration, scientific computing PHP, CUDA tensor, PHP extension development
</div>
