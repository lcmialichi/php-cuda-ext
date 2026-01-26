<?php

namespace Benchmarks\Contracts;

use Benchmarks\Support\BenchmarkResult;

interface BenchmarkInterface
{
    public function name(): string;
    public function description(): string;
    public function register(): array;
    public function run(
        string $name,
        callable $exec,
        string $type,
        int $iterations,
        array $args,
        bool $warmup,
        array $metadata,
    ): BenchmarkResult;
}
