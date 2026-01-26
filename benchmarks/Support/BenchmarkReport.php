<?php

namespace Benchmarks\Support;

class BenchmarkReport
{
    /**
     * @param array<> $results
     * @throws \Exception
     */
    public function __construct(private array $results)
    {
    }
}
