<?php

namespace Benchmarks\Support;

use Benchmarks\Support\BenchmarkResult;
use Benchmarks\Contracts\BenchmarkInterface;

class BenchmarkClassResult
{
    /**
     * @param BenchmarkInterface $handler
     * @param array<BenchmarkResult> $results
     */
    public function __construct(
        private BenchmarkInterface $handler,
        private array $results
    ) {}

    public function getHandler(): BenchmarkInterface
    {
        return $this->handler;
    }

    /**
     * @return BenchmarkResult[]
     */
    public function getResults(): array
    {
        return $this->results;
    }
}
