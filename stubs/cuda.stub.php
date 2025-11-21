<?php

/**
 * @method CudaArray __add(CudaArray|float|int $other)
 * @method CudaArray __sub(CudaArray|float|int $other)
 * @method CudaArray __mul(CudaArray|float|int $other)
 * @method CudaArray __div(CudaArray|float|int $other)
 * @method CudaArray __pow(CudaArray|float|int $other)
 * @method bool __gt(CudaArray|float|int $other)
 * @method bool __lt(CudaArray|float|int $other)
 * @method bool __eq(CudaArray|float|int $other)
 */
class CudaArray
{
    public function __construct(array $data){}

    public function toArray(): array {}
    public function getShape(): array{}

    public function __invoke(int|null|array ...$slices ): CudaArray {}

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

    public function transpose(?int $axis = null): CudaArray {}
    public function reshape(array $newShape): CudaArray {}
    public function flatten(): CudaArray {}


}

