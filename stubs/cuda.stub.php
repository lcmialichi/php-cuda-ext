<?php

class CudaArray
{
    public function __construct(array $data){}

    public function toArray(): array {}
    public function getShape(): array{}

    public function __invoke(int|null|array ...$slices ): CudaArray {}
    public function multiply(CudaArray|float $other): CudaArray {}
    public function add(CudaArray|float $other): CudaArray {}
    public function subtract(CudaArray|float $other): CudaArray {}
    public function divide(CudaArray|float $other): CudaArray {}
    public function power(float|CudaArray $exponent): CudaArray {}

    public function cos(): CudaArray {}
    public function sin(): CudaArray {}
    public function exp(): CudaArray {}
    public function log(): CudaArray {}
    public function sqrt(): CudaArray {}

    
    public static function ones(array $shape): CudaArray {}
    public static function zeros(array $shape): CudaArray {}
    public static function fill(array $shape, float $value): CudaArray {}

    public function transpose(?int $axis = null): CudaArray {}
    public function reshape(array $newShape): CudaArray {}


}

