<?php

namespace Benchmarks\Support\Attr;

#[\Attribute(\Attribute::TARGET_METHOD)]
class InjectArgs
{
    public function __construct(private string $method) {}

    public function getMethod(): string
    {
        return $this->method;
    }
}
