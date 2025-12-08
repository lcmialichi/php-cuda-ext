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
class Input
{
    public function __construct(
        public string $dtype = 'float',
    ) {
    }
}

#[Attribute(Attribute::TARGET_PARAMETER)]
class Output
{
    public function __construct(
        public string $dtype = 'float',
    ) {
    }

}
