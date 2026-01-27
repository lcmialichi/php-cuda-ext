<?php

namespace Benchmarks\Support;

class BenchmarkResult implements \JsonSerializable
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

    public function getMinMemoryUsage(): float
    {
        return min($this->memoryUsages);
    }

    public function getMaxMemoryUsage(): float
    {
        return max($this->memoryUsages);
    }

    public function getAvgMemoryUsage(): float
    {
        return  array_sum($this->memoryUsages) / count($this->memoryUsages);
    }

    public function getMinTime(): float
    {
        return min($this->times);
    }

    public function getMaxTime(): float
    {
        return max($this->times);
    }

    public function getAvgTime(): float
    {
        return array_sum($this->times) / count($this->times);
    }

    public function jsonSerialize(): array
    {
        return [
            "name" => $this->getName(),
            "type" => $this->getType(),
            "iterations" => $this->getIterations(),
            "metadata" => $this->getMetadata(),
            "time" => [
                "format" => "MS",
                "min" => $this->getMinTime(),
                "max" => $this->getMaxTime(),
                "avg" => $this->getAvgTime(),
                "total" => array_sum($this->getTimes())
            ],
            "memory" => [
                "format" => "B",
                "min" => $this->getMinMemoryUsage(),
                "max" => $this->getMaxMemoryUsage(),
                "avg" => $this->getAvgMemoryUsage(),
                "total" => array_sum($this->getMemoryUsages())
            ],
        ];
    }
}
