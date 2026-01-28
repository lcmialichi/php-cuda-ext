<?php

namespace Cuda;

use IteratorAggregate;

#[\Attribute(Attribute::TARGET_PARAMETER)]
abstract class ParamAttribute
{
    abstract public function getDtype(): string;
    abstract public function isList(): bool;
    abstract public function isNullable(): bool;
}

class Compiler
{
    public function kernel(callable $fn): static
    {
        return $this;
    }

    public function compile(?string $target = '', bool $optimize = true, bool $debug = false)
    {
        return new CompiledModule();
    }
}

class Device
{
    public function __invoke() {}

    public static function fn(callable $callable): static
    {
        return new self();
    }
}

/**
 * Compiled CUDA module containing PTX code and kernel functions.
 * 
 * @method bool run(string $name) Execute kernel with default configuration and no arguments
 * @method bool run(string $name, array $config) Execute kernel with configuration but no arguments
 * @method bool run(string $name, array $config, array $args) Execute kernel with configuration and arguments
 * @method bool run(string $name, array $config = [], array $args = []) Execute kernel with optional configuration and arguments
 * 
 * @package Cuda
 */
class CompiledModule
{
    public function initialize(): bool
    {
        return true;
    }

    public function run(string $name, array $config = [], array $args): bool
    {
        return true;
    }

    public function runAsync(string $name, array $config = [], array $args): int
    {
        return true;
    }

    public function isFinished(?int $id = null): bool
    {
        return true;
    }

    public function sync(): bool
    {
        return true;
    }

    public function getAsyncStatus(?int $id = null): array
    {
        return [];
    }

    public function getStats(): array
    {
        return [];
    }

    public function wait(?int $id = null, int $timeout = -1) {}

    public function getPendingOperations(): array
    {
        return [];
    }

    public function cancelOperation(int $id): bool
    {
        return false;
    }

    public function cleanup(): int
    {
        return 0;
    }


    public function hasKernel(string $kernel): bool
    {
        return false;
    }

    public function getKernels(): array
    {
        return [];
    }

    public function getPtx() {}

    public function save(string $path) {}
}

class Kernel {}

class ContiguousArray implements \ArrayAccess
{
    public function toGpu(): CudaArray {}

    function offsetExists(mixed $offset): bool {}

    function offsetGet(mixed $offset): mixed {}

    function offsetSet(mixed $offset, mixed $value): void {}

    function offsetUnset(mixed $offset): void {}

    public function get(array $idxs): mixed {}

    public function at(int ...$idxs): mixed {}

    public function getNdims(): int {}

    public function getShape(): array {}

    public function toArray(): array {}

    public function getSize(): int {}

    public function count(): int {}

    public function getDtype(): string {}
}

/**
 * <psalm
 * disallowLiteralKeysOnUnshapedArrays="[bool]"
 *>
 */
class CudaArray implements \ArrayAccess
{
    /**
     *
     * @param array<int, float> $data
     */
    public function __construct(array $data) {}

    public function toArray(): array {}

    public function toHost(): ContiguousArray {}

    public function getShape(): array {}

    public function getNdims(): int {}

    public function getSize(): int {}

    public function __invoke(int|null|array ...$slices): CudaArray {}

    /**
     * @param CudaArray|float|int $other
     * @return CudaArray
     */
    public function multiply(CudaArray|float $other): CudaArray {}

    /**
     * @param CudaArray|float|int $other
     * @return CudaArray
     */
    public function add(CudaArray|float $other): CudaArray {}

    /**
     * @param CudaArray|float|int $other
     * @return CudaArray
     */
    public function subtract(CudaArray|float $other): CudaArray {}

    /**
     * @param CudaArray|float|int $other
     * @return CudaArray
     */
    public function divide(CudaArray|float $other): CudaArray {}

    /**
     * @param CudaArray|float|int $other
     * @return CudaArray
     */
    public function power(float|CudaArray $exponent): CudaArray {}

    public function cos(): CudaArray {}
    public function sin(): CudaArray {}
    public function exp(): CudaArray {}
    public function log(): CudaArray {}
    public function sqrt(): CudaArray {}

    public static function ones(array $shape): CudaArray {}
    public static function zeros(array $shape): CudaArray {}
    public static function full(array $shape, float $value): CudaArray {}

    public static function rand(array $shape, ?float $min = null, ?float $max = null): CudaArray {}

    public function transpose(?array $axis = null): CudaArray {}
    public function reshape(array $newShape): CudaArray {}
    public function flatten(): CudaArray {}

    public function gt(float|CudaArray $other): CudaArray {}

    public function ge(float|CudaArray $other): CudaArray {}

    public function lt(float|CudaArray $other): CudaArray {}

    public function eq(float|CudaArray $other): CudaArray {}
    public function ne(float|CudaArray $other): CudaArray {}

    public function le(float|CudaArray $other): CudaArray {}

    public function neg(): CudaArray {}

    public function floor(): CudaArray {}

    public function ceil(): CudaArray {}

    public function round(): CudaArray {}

    public function sum(?int $axis = null): CudaArray {}
    public function mean(?int $axis = null): CudaArray {}
    public function max(?int $axis = null): CudaArray {}
    public function min(?int $axis = null): CudaArray {}
    public function prod(?int $axis = null): CudaArray {}
    public function argMax(?int $axis = null): CudaArray {}
    public function argMin(?int $axis = null): CudaArray {}

    public function matmul(CudaArray $other): CudaArray {}

    /**
     * @param array[CudaArray] $tensors
     * @param mixed $axis
     * @return void
     */
    public function concat(array $tensors, ?int $axis = null): CudaArray {}

    function offsetExists(mixed $offset): bool {}

    function offsetGet(mixed $offset): mixed {}

    function offsetSet(mixed $offset, mixed $value): void {}

    function offsetUnset(mixed $offset): void {}
}
