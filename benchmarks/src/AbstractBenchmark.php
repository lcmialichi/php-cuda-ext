<?php

declare(strict_types=1);

require_once __DIR__ . '/AbstractBenchmark.php';

abstract class AbstractBenchmark
{
    protected const FLOAT_SIZE = 4;
    protected const RANDOM_SAMPLES = 1_000_000;

    protected array $results = [];
    protected array $config = [];
    protected float $startTime;
    protected string $benchmarkName;

    public function __construct(array $config = [])
    {
        $this->config = $config;
        $this->startTime = microtime(true);
        $this->benchmarkName = (new \ReflectionClass($this))->getShortName();
    }

    abstract public function run(): array;

    abstract public function getName(): string;

    abstract public function getDescription(): string;

    protected function warmup(callable $function, int $iterations = 3): void
    {
        for ($i = 0; $i < $iterations; $i++) {
            $function();
        }
        $this->flush();
    }

    protected function measure(callable $function, int $iterations = 5): array
    {
        for ($i = 0; $i < 2; $i++) {
            $function();
        }

        $times = [];
        $memoryUsages = [];

        for ($i = 0; $i < $iterations; $i++) {
            $this->flush();
            $memBefore = memory_get_usage(true);
            $start = hrtime(true);

            $function();

            $end = hrtime(true);
            $memAfter = memory_get_usage(true);

            $times[] = ($end - $start) / 1e9;
            $memoryUsages[] = $memAfter - $memBefore;
        }

        return [
            'time' => [
                'avg' => array_sum($times) / count($times),
                'min' => min($times),
                'max' => max($times),
                'std' => $this->calculateStdDev($times),
                'iterations' => $iterations,
                'unit' => 's'
            ],
            'memory' => [
                'avg' => array_sum($memoryUsages) / count($memoryUsages),
                'min' => min($memoryUsages),
                'max' => max($memoryUsages),
                'unit' => 'bytes'
            ],
            'raw_times' => $times,
            'raw_memory' => $memoryUsages
        ];
    }

    protected function benchmarkOperation(
        string $operationName,
        callable $operation,
        ?callable $native = null,
        array $metadata = [],
        int $iterations = 5
    ): array {
        $result = $this->measure($operation, $iterations);
        $resultNative = $native !== null ? $this->measure($native, $iterations) : null;

        $benchmarkResult = [
            'operation' => $operationName,
            'timestamp' => microtime(true),
            'metadata' => $metadata,
            'native' => $resultNative,
            'performance' => $result,
        ];

        $this->results[] = $benchmarkResult;
        return $benchmarkResult;
    }

    protected function calculateStdDev(array $values): float
    {
        if (count($values) < 2) {
            return 0.0;
        }

        $mean = array_sum($values) / count($values);
        $sumSquares = 0.0;

        foreach ($values as $value) {
            $sumSquares += pow($value - $mean, 2);
        }

        return sqrt($sumSquares / (count($values) - 1));
    }

    protected function formatBytes(float $bytes): string
    {
        $units = ['B', 'KB', 'MB', 'GB', 'TB'];
        $bytes = max($bytes, 0);
        $pow = floor(($bytes ? log($bytes) : 0) / log(1024));
        $pow = min($pow, count($units) - 1);
        $bytes /= (1 << (10 * $pow));

        return round($bytes, 2) . ' ' . $units[$pow];
    }

    protected function formatTime(float $seconds): string
    {
        if ($seconds < 0.001) {
            return round($seconds * 1e6, 2) . ' µs';
        } elseif ($seconds < 1) {
            return round($seconds * 1e3, 2) . ' ms';
        }
        return round($seconds, 4) . ' s';
    }

    protected function flush(): void
    {
        if ($this->config['performance']['gc_enabled'] ?? true) {
            gc_collect_cycles();
            gc_mem_caches();
        }
    }

    public function getResults(): array
    {
        return [
            'benchmark' => $this->benchmarkName,
            'name' => $this->getName(),
            'description' => $this->getDescription(),
            'total_time' => microtime(true) - $this->startTime,
            'system_info' => $this->getSystemInfo(),
            'results' => $this->results,
        ];
    }

    protected function getSystemInfo(): array
    {
        return [
            'php_version' => PHP_VERSION,
            'os' => PHP_OS,
            'memory_limit' => ini_get('memory_limit'),
            'extensions' => get_loaded_extensions(),
            'cuda_extension_loaded' => extension_loaded('cuda'),
            'timestamp' => date('Y-m-d H:i:s'),
        ];
    }

    public function toConsole(): string
    {
        $output = [];
        $output[] = "\n" . str_repeat("=", 80);
        $output[] = sprintf("BENCHMARK: %s", $this->getName());
        $output[] = str_repeat("=", 80);
        $output[] = sprintf("Description: %s", $this->getDescription());
        $output[] = sprintf("PHP: %s | OS: %s", PHP_VERSION, PHP_OS);
        $output[] = sprintf("Memory limit: %s", ini_get('memory_limit'));
        $output[] = "";

        foreach ($this->results as $result) {
            $output[] = sprintf("[%s]", strtoupper($result['operation']));

            if (!empty($result['metadata'])) {
                foreach ($result['metadata'] as $key => $value) {
                    $output[] = sprintf("  %s: %s", $key, $value);
                }
            }

            $perf = $result['performance']['time'];
            $output[] = sprintf(
                "  Time: %.4f s (min: %.4f, max: %.4f, std: %.4f, iterations: %d)",
                $perf['avg'],
                $perf['min'],
                $perf['max'],
                $perf['std'],
                $perf['iterations']
            );

            $mem = $result['performance']['memory'];
            $output[] = sprintf(
                "  Memory: %s (avg, min: %s, max: %s)",
                $this->formatBytes($mem['avg']),
                $this->formatBytes($mem['min']),
                $this->formatBytes($mem['max'])
            );

            $output[] = "";
        }

        return implode("\n", $output);
    }
}