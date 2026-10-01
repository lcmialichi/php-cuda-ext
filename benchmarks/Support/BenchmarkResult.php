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

    public function getMedianTime(): float
    {
        $times = $this->times;
        sort($times, SORT_NUMERIC);
        $middle = intdiv(count($times), 2);
        return count($times) % 2 ? $times[$middle] : ($times[$middle - 1] + $times[$middle]) / 2;
    }

    public function getP95Time(): float
    {
        $times = $this->times;
        sort($times, SORT_NUMERIC);
        return $times[(int)ceil(count($times) * 0.95) - 1];
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
                "median" => $this->getMedianTime(),
                "p95" => $this->getP95Time(),
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
