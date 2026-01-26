<?php

namespace Benchmarks\Support;

class BenchmarkResult
{
    public function __construct(
        private string $name,
        private string $type,
        private int $iterations,
        private array $times,
        private array $memoryUsages,
        private array $metadata = [],
    ) {}

    public function getName(): string
    {
        return $this->name;
    }

    public function getType(): string
    {
        return $this->type;
    }

    public function getIterations(): int
    {
        return $this->iterations;
    }

    public function getTimes(): array
    {
        return $this->times;
    }

    public function getMemoryUsages(): array
    {
        return $this->memoryUsages;
    }

    public function getMetadata(): array
    {
        return $this->metadata;
    }
}
