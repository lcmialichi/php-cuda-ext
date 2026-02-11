<?php

namespace Benchmarks\Support;

use Benchmarks\Support\BenchmarkResult;
use Benchmarks\Contracts\BenchmarkInterface;

class BenchmarkClassResult implements \JsonSerializable
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

    public function jsonSerialize(): mixed
    {
        return [
            "class" => get_class($this),
            "name" => $this->handler->name(),
            "description" => $this->handler->description(),
            "results" => $this->results
        ];
    }
}
