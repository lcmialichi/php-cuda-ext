<?php

namespace Cuda;

use Countable;
use IteratorAggregate;

#[\Attribute(\Attribute::TARGET_PARAMETER)]
abstract class ParamAttribute
{
    abstract public function getDtype(): string;
    abstract public function isList(): bool;
    abstract public function isNullable(): bool;
}

class Compiler
{
    public function __construct(private ?string $target = null) {}

    public function kernel(callable $fn): static
    {
        return $this;
    }

    public function compile(bool $optimize = true, bool $debug = false): CompiledModule
    {
        return new CompiledModule();
    }

    public function getCacheStats(): array
    {
        return [];
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

    public function autoGrid(string $kernel, int|CudaArray $elements): array
    {
        return [];
    }

    public function launch(string $name, array $config = [], array $args): bool
    {
        return false;
    }

    public function launchAsync(string $name, array $config = [], array $args): int
    {
        return false;
    }

    public function launchAsyncBatch(array $operations): bool|array
    {
        return false;
    }


    public function isFinished(?int $id = null): bool
    {
        return true;
    }

    public function sync(?int $id = null): bool
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

    public function __serialize(): array
    {
        return [];
    }

    public function __unserialize(array $data): void {}
}

class Kernel {}

class ContiguousArray implements \ArrayAccess, Countable
{
    public function toGpu(): CudaArray {}

    function offsetExists(mixed $offset): bool {}

    function offsetGet(mixed $offset): mixed {}

    function offsetSet(mixed $offset, mixed $value): void {}

    function offsetUnset(mixed $offset): void {}

    function getElementSize(): int {}

    public function get(array $idxs): mixed {}

    public function at(int ...$idxs): mixed {}

    public function getNdims(): int {}

    public function getShape(): array {}

    public function toArray(): array {}

    public function getSize(): int {}

    public function count(): int {}

    public function getDtype(): string {}

    public function count(): int {}

    public function __serialize(): array {}
    public function __unserialize(array $data): void {}
}

/**
 * @property int|float|bool|null|string|CudaArray $cdata
 */
class CudaArray implements \ArrayAccess
{
    /**
     *
     * @param array<int, float> $data
     */
    public function __construct(array $data, ?string $dtype = 'float32') {}

    public function toArray(): array {}

    public function toHost(): ContiguousArray {}

    public function dtype(): string {}

    public function astype(string $type): CudaArray {}

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

    public static function ones(array $shape, ?string $dtype = 'float32'): CudaArray {}
    public static function zeros(array $shape, ?string $dtype = 'float32'): CudaArray {}
    public static function full(array $shape, float|int $value, ?string $dtype = 'float32'): CudaArray {}

    public static function rand(array $shape, ?float $min = null, ?float $max = null, ?string $dtype = 'float32'): CudaArray {}

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

/**
 * Abstract base class to enable mathematical operator overloading.
 * * When inherited, any arithmetic operations performed on the object 
 * will be dispatched to the corresponding magic methods.
 */
abstract class Number
{
    /**
     * Handles the Addition (+) operation.
     * * @param mixed $left The left-hand operand of the expression.
     * @param mixed $right The right-hand operand of the expression.
     * @return mixed Usually returns a new instance of the inheriting class.
     */
    abstract public function __add(mixed $left, mixed $right): mixed;

    /**
     * Handles the Subtraction (-) operation.
     * * @param mixed $left The left-hand operand of the expression.
     * @param mixed $right The right-hand operand of the expression.
     * @return mixed
     */
    abstract public function __sub(mixed $left, mixed $right): mixed;

    /**
     * Handles the Multiplication (*) operation.
     * * @param mixed $left The left-hand operand of the expression.
     * @param mixed $right The right-hand operand of the expression.
     * @return mixed
     */
    abstract public function __mul(mixed $left, mixed $right): mixed;

    /**
     * Handles the Division (/) operation.
     * * @param mixed $left The left-hand operand of the expression.
     * @param mixed $right The right-hand operand of the expression.
     * @return mixed
     */
    abstract public function __div(mixed $left, mixed $right): mixed;

    /**
     * Handles the Modulo (%) operation.
     * * @param mixed $left The left-hand operand of the expression.
     * @param mixed $right The right-hand operand of the expression.
     * @return mixed
     */
    abstract public function __mod(mixed $left, mixed $right): mixed;

    /**
     * Handles the Exponentiation (**) operation.
     * * @param mixed $left The left-hand operand of the expression.
     * @param mixed $right The right-hand operand of the expression.
     * @return mixed
     */
    abstract public function __pow(mixed $left, mixed $right): mixed;


    /**
     * Handles increment operations (++$a or $a++)
     * @return void
     */
    abstract public function __inc(): void;

    /**
     * Handles decrement operations (--$a or $a--)
     * @return void
     */
    abstract public function __dec(): void;
}
