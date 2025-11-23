# TODO — PHP CUDA Extension

This file lists upcoming features and tasks planned for the PHP CUDA extension.  
Items are prioritized but not yet scheduled.


## Matrix Operations
- [ ] Implement `matmul` (GPU-accelerated matrix multiplication)
  - Detect shapes and validate dimensional compatibility
  - Add broadcasting support if possible
  - Optimize for contiguous and non-contiguous tensors


## Tensor Indexing & Assignment

- [ ] Add support for ranged write assignment using syntax:
```php
$x[[0, 5]] = $x[[1, 6]];
```

## Tasks
- [ ] Parse slice ranges during assignment
- [ ] Ensure view semantics (no copy unless necessary)
- [ ] Handle overlapping ranges safely

## GPU Device Management
- [ ] Implement effective GPU device selection (function exists but does not apply the device yet)
    - Apply device via cudaSetDevice()
    - Validate device availability
    - Update internal global state
    - Ensure tensors respect the selected device

## Data Types

- [ ] Add support for float32 and float64
    - Introduce dtype field inside tensor_t
    - Implement type-aware kernels
    - Add casting and type promotion rules
    - Update memory allocation to match dtype size

## Conditional Operations
- [] Implement where operation:
```php
$z = CudaArray::where($cond, $x, $y);
```
 - Condition tensor must be treated as boolean mask
 - Broadcasting support for all arguments
 - Kernel must pick element-wise from ``$x`` or ``$y``

## Additional Notes
Future improvements may include:

- Kernel fusion / operation batching
- Stream support for async execution
- Memory pool improvements