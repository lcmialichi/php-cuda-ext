<?php

namespace Benchmarks\Handlers;

use Benchmarks\Support\BenchmarkResult;
use Benchmarks\Contracts\BenchmarkInterface;

abstract class Benchmark implements BenchmarkInterface
{
    public function run(
        string $name,
        callable $exec,
        string $type,
        int $iterations,
        array $args = [],
        bool $warmup = false,
        array $metadata = []
    ): BenchmarkResult {
        if ($warmup === true) {
            $this->warmup($exec, $args);
        }

        $times = [];
        $memoryUsages  = [];
        for ($i = 0; $i < $iterations; $i++) {
            $this->flush();
            [$time, $mem] = $this->doRun($exec, $args);

            $times[] = $time;
            $memoryUsages[] = $mem;
        }

        unset($args);

        return new BenchmarkResult(
            name: $name,
            type: $type,
            iterations: $iterations,
            times: $times,
            memoryUsages: $memoryUsages,
            metadata: $metadata
        );
    }

    protected function doRun(callable $exec, array $args): array
    {
        $memoryS = memory_get_usage(true);
        $timeS = hrtime(true);

        $exec(...$args);
        $timeE = hrtime(true);
        $memoryE = memory_get_usage(true);

        return [($timeE - $timeS) / 1e6, $memoryE - $memoryS];
    }

    protected function warmup(callable $exec, array $args = [], int $times = 10): void
    {
        for ($i = 0; $i < $times; $i++) {
            $exec(...$args);
        }
    }

    protected function flush(): void
    {
        gc_collect_cycles();
        gc_mem_caches();
    }
}
