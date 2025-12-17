<?php

namespace Cuda\Attr;

use Attribute;

#[Attribute(Attribute::TARGET_METHOD)]
class Kernel
{
    public function __construct(
        public string $name,
        public string $target = 'sm_60'
    ) {
    }
}

#[Attribute(Attribute::TARGET_METHOD)]
class Device
{
    public function __construct(
        public string $name,
        public string $target = 'sm_60'
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
class Tensor extends ParamAttribute
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
