<?php

declare(strict_types=1);

namespace Cuda\Benchmarks;

use Cuda\ContiguousArray;
use Cuda\CudaArray;

class ArrayPerformanceSuite
{
    private const ITERATIONS = 3;
    private const FLOAT_SIZE = 4;

    public static function run(): void
    {
        self::warmup();

        $scenarios = [
            '1D_HUGE'    => [10_000_000],
            '2D_SQUARE'  => [4000, 4000],
            '3D_TALL'    => [1024, 1024, 16],
            '3D_STRESS'  => [512, 512, 512],
            '4D_SMALL'   => [10, 10, 10, 10],
            '4D_LARGE'   => [100, 100, 50, 20],
            '5D_STRESS'  => [20, 20, 20, 20, 20],
        ];

        echo "CUDA ContiguousArray - Comprehensive Benchmark Suite\n";
        echo "PHP Version: " . PHP_VERSION . " | OS: " . PHP_OS . "\n";
        echo "Memory Limit: " . ini_get('memory_limit') . "\n";
        echo str_repeat("=", 95) . "\n";

        foreach ($scenarios as $label => $dims) {
            (new self())->executeScenario($label, $dims);
        }
    }

    private static function warmup(): void
    {
        $dummyDims = [100, 100];
        $cuda = CudaArray::rand($dummyDims);
        $contiguous = $cuda->toHost();
        $php = $contiguous->toArray();
        
        for ($i = 0; $i < 10; $i++) {
            $sum = 0.0;
            foreach ($php as $row) foreach ($row as $v) $sum += $v;
            for ($x = 0; $x < 100; $x++) for ($y = 0; $y < 100; $y++) $sum += $contiguous->at($x, $y);
        }
        unset($cuda, $contiguous, $php);
        gc_collect_cycles();
    }

    private function executeScenario(string $label, array $dims): void
    {
        $totalElements = (int) array_product($dims);
        $expectedRawSize = $totalElements * self::FLOAT_SIZE;

        echo "\n[SCENARIO: $label]\n";
        echo "Dimensions: " . implode('x', $dims) . " (" . number_format($totalElements) . " elements)\n";
        echo "Expected Raw Data Size: " . $this->formatBytes($expectedRawSize) . "\n";
        echo str_repeat("-", 95) . "\n";

        $this->flush();
        $memBase = memory_get_usage();
        $timeStartC = microtime(true);

        $cuda = CudaArray::rand($dims);
        $contiguous = $cuda->toHost();

        $allocationTimeC = microtime(true) - $timeStartC;
        $memUsedC = memory_get_usage() - $memBase;

        $this->flush();
        $memBaseP = memory_get_usage();
        $timeStartP = microtime(true);

        $phpArray = $contiguous->toArray();

        $allocationTimeP = microtime(true) - $timeStartP;
        $memUsedP = memory_get_usage() - $memBaseP;

        $accessAt = $this->benchmark(fn() => $this->traverseAt($contiguous, $dims));
        $accessBracketC = $this->benchmark(fn() => $this->traverseBracketsC($contiguous, $dims));
        $accessNative = $this->benchmark(fn() => $this->traverseNative($phpArray, $dims));

        $this->printReport([
            'Metric'          => ['CA .at()', 'CA [][]', 'PHP [][]', 'Comparison'],
            'Memory Usage'    => [$this->formatBytes($memUsedC), 'Shared', $this->formatBytes($memUsedP), $this->ratio($memUsedP, $memUsedC) . " more"],
            'Bytes per Elem'  => [round($memUsedC / $totalElements, 2), 'N/A', round($memUsedP / $totalElements, 2), ''],
            'Alloc. Time'     => [sprintf("%.4fs", $allocationTimeC), 'N/A', sprintf("%.4fs", $allocationTimeP), $this->ratio($allocationTimeP, $allocationTimeC) . " slower"],
            'Access Time'     => [sprintf("%.4fs", $accessAt), sprintf("%.4fs", $accessBracketC), sprintf("%.4fs", $accessNative), $this->getWinner($accessAt, $accessBracketC, $accessNative)],
        ]);

        unset($cuda, $contiguous, $phpArray);
    }

    private function benchmark(callable $work): float
    {
        $times = [];
        for ($i = 0; $i < self::ITERATIONS; $i++) {
            $start = microtime(true);
            $work();
            $times[] = microtime(true) - $start;
        }
        return array_sum($times) / self::ITERATIONS;
    }

    private function traverseAt(ContiguousArray $arr, array $dims): void
    {
        $sum = 0.0; $rank = count($dims);
        if ($rank === 1) {
            for ($i = 0; $i < $dims[0]; $i++) $sum += $arr->at($i);
        } elseif ($rank === 2) {
            for ($i = 0; $i < $dims[0]; $i++)
                for ($j = 0; $j < $dims[1]; $j++) $sum += $arr->at($i, $j);
        } elseif ($rank === 3) {
            for ($i = 0; $i < $dims[0]; $i++)
                for ($j = 0; $j < $dims[1]; $j++) $sum += $arr->at($i, $j, 0);
        } elseif ($rank === 4) {
            for ($i = 0; $i < $dims[0]; $i++)
                for ($j = 0; $j < $dims[1]; $j++) $sum += $arr->at($i, $j, 0, 0);
        } else {
            for ($i = 0; $i < $dims[0]; $i++)
                for ($j = 0; $j < $dims[1]; $j++) $sum += $arr->at($i, $j, 0, 0, 0);
        }
    }

    private function traverseBracketsC(ContiguousArray $arr, array $dims): void
    {
        $sum = 0.0; $rank = count($dims);
        if ($rank === 1) {
            for ($i = 0; $i < $dims[0]; $i++) $sum += $arr[$i];
        } elseif ($rank === 2) {
            for ($i = 0; $i < $dims[0]; $i++)
                for ($j = 0; $j < $dims[1]; $j++) $sum += $arr[$i][$j];
        } elseif ($rank === 3) {
            for ($i = 0; $i < $dims[0]; $i++)
                for ($j = 0; $j < $dims[1]; $j++) $sum += $arr[$i][$j][0];
        } elseif ($rank === 4) {
            for ($i = 0; $i < $dims[0]; $i++)
                for ($j = 0; $j < $dims[1]; $j++) $sum += $arr[$i][$j][0][0];
        } else {
            for ($i = 0; $i < $dims[0]; $i++)
                for ($j = 0; $j < $dims[1]; $j++) $sum += $arr[$i][$j][0][0][0];
        }
    }

    private function traverseNative(array $arr, array $dims): void
    {
        $sum = 0.0; $rank = count($dims);
        if ($rank === 1) {
            for ($i = 0; $i < $dims[0]; $i++) $sum += $arr[$i];
        } elseif ($rank === 2) {
            for ($i = 0; $i < $dims[0]; $i++)
                for ($j = 0; $j < $dims[1]; $j++) $sum += $arr[$i][$j];
        } elseif ($rank === 3) {
            for ($i = 0; $i < $dims[0]; $i++)
                for ($j = 0; $j < $dims[1]; $j++) $sum += $arr[$i][$j][0];
        } elseif ($rank === 4) {
            for ($i = 0; $i < $dims[0]; $i++)
                for ($j = 0; $j < $dims[1]; $j++) $sum += $arr[$i][$j][0][0];
        } else {
            for ($i = 0; $i < $dims[0]; $i++)
                for ($j = 0; $j < $dims[1]; $j++) $sum += $arr[$i][$j][0][0][0];
        }
    }

    private function printReport(array $data): void
    {
        foreach ($data as $key => $values) {
            printf("%-18s | %-15s | %-15s | %-15s | %-15s\n", $key, $values[0], $values[1], $values[2], $values[3]);
        }
    }

    private function getWinner(float $at, float $bracketC, float $native): string
    {
        $min = min($at, $bracketC, $native);
        if ($min === $at) return 'Cont::at() faster';
        if ($min === $bracketC) return 'Cont[][] faster';
        return 'PHP faster';
    }

    private function formatBytes(int $bytes): string
    {
        $units = ['B', 'KB', 'MB', 'GB'];
        $pow = floor(($bytes ? log($bytes) : 0) / log(1024));
        $bytes /= (1 << (10 * $pow));
        return round($bytes, 2) . ' ' . $units[$pow];
    }

    private function ratio(float $val1, float $val2): string
    {
        return $val2 > 0 ? round($val1 / $val2, 2) . 'x' : 'N/A';
    }

    private function flush(): void
    {
        gc_collect_cycles();
        gc_mem_caches();
    }
}

ArrayPerformanceSuite::run();