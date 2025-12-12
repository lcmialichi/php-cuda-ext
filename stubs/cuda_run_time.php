<?php 

namespace Cuda;

class Runtime
{   
    public Math $math;
    public Atomic $atomic;
    public Memory $memory;
    public Sync $sync;


    public function threadIdx(): int {}
    
    public function blockIdx(): int {}
    
    public function blockDim(): int {}
    
    public function gridDim(): int {}
    
}

class Math
{
    public function sin(float $x): float {}
    public function cos(float $x): float {}
    public function exp(float $x): float {}
    public function log(float $x): float {}
    public function sqrt(float $x): float {}
    public function pow(float $x, float $y): float {}
    public function max(float $a, float $b): float {}
    public function min(float $a, float $b): float {}
    public function abs(float $x): float {}
    public function ceil(float $x): float {}
    public function floor(float $x): float {}
    public function round(float $x): float {}
}

class Atomic
{
    public function add(float &$address, float $val): float {}
    public function sub(float &$address, float $val): float {}
    public function max(int &$address, int $val): int {}
    public function min(int &$address, int $val): int {}
}

class Memory
{
    public function shared(int $size): array {}
    public function global(): void {}
}

class Sync
{
    public function threads(): void {}
    public function warp(): void {}
}