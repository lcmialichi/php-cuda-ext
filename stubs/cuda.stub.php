<?php

/**
 * <psalm
 * disallowLiteralKeysOnUnshapedArrays="[bool]"
 *>
 */
class CudaArray implements ArrayAccess
{
    public function __construct(array $data)
    {
    }

    public function toArray(): array
    {
    }
    public function getShape(): array
    {
    }

    public function __invoke(int|null|array ...$slices): CudaArray
    {
    }


    /**
     * @param CudaArray|float|int $other
     * @return CudaArray
     */
    public function multiply(CudaArray|float $other): CudaArray
    {
    }

    /**
     * @param CudaArray|float|int $other
     * @return CudaArray
     */
    public function add(CudaArray|float $other): CudaArray
    {
    }

    /**
     * @param CudaArray|float|int $other
     * @return CudaArray
     */
    public function subtract(CudaArray|float $other): CudaArray
    {
    }

    /**
     * @param CudaArray|float|int $other
     * @return CudaArray
     */
    public function divide(CudaArray|float $other): CudaArray
    {
    }

    /**
     * @param CudaArray|float|int $other
     * @return CudaArray
     */
    public function power(float|CudaArray $exponent): CudaArray
    {
    }

    public function cos(): CudaArray
    {
    }
    public function sin(): CudaArray
    {
    }
    public function exp(): CudaArray
    {
    }
    public function log(): CudaArray
    {
    }
    public function sqrt(): CudaArray
    {
    }


    public static function ones(array $shape): CudaArray
    {
    }
    public static function zeros(array $shape): CudaArray
    {
    }
    public static function full(array $shape, float $value): CudaArray
    {
    }

    public function transpose(?int $axis = null): CudaArray
    {
    }
    public function reshape(array $newShape): CudaArray
    {
    }
    public function flatten(): CudaArray
    {
    }


    function offsetExists(mixed $offset): bool
    {
    }

    function offsetGet(mixed $offset): mixed
    {
    }

    function offsetSet(mixed $offset, mixed $value): void
    {
    }

    function offsetUnset(mixed $offset): void
    {
    }

}

