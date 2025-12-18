<?php

namespace Cuda\Attr;

use Attribute;

#[Attribute(Attribute::TARGET_METHOD)]
class Kernel
{
    public function __construct(
        public string $name,
    ) {
    }
}

#[Attribute(Attribute::TARGET_METHOD)]
class Device
{
    public function __construct(
        public string $name,
    ) {
    }
}

#[Attribute(Attribute::TARGET_PARAMETER)]
abstract class ParamAttribute
{
    abstract public function getDtype(): string;
    abstract public function isList(): bool;
    abstract public function isNullable(): bool;
}

#[Attribute(Attribute::TARGET_PARAMETER)]
class TensorType extends ParamAttribute
{
    public function __construct(
        public string $dtype = 'float32',
    ) {
    }

    public function getDtype(): string
    {
        return $this->dtype;
    }

    public function isList(): bool
    {
        return true;
    }

    public function isNullable(): bool
    {
        return false;
    }

}

#[Attribute(Attribute::TARGET_PARAMETER)]
class BoolType extends ParamAttribute
{

    public function getDtype(): string
    {
        return 'bool';
    }

    public function isList(): bool
    {
        return false;
    }

    public function isNullable(): bool
    {
        return false;
    }

}

#[Attribute(Attribute::TARGET_PARAMETER)]
class IntType extends ParamAttribute
{
    public function __construct(private int $bits)
    {
    }

    public function getDtype(): string
    {
        return "int{$this->bits}";
    }

    public function isList(): bool
    {
        return false;
    }

    public function isNullable(): bool
    {
        return false;
    }

}

#[Attribute(Attribute::TARGET_PARAMETER)]
class FloatType extends ParamAttribute
{

    public function __construct(private int $bits = 32)
    {

    }

    public function getDtype(): string
    {
        return "float{$this->bits}";
    }

    public function isList(): bool
    {
        return false;
    }

    public function isNullable(): bool
    {
        return false;
    }

}


