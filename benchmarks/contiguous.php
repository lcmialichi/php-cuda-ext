<?php

declare(strict_types=1);

namespace Cuda\Benchmarks;

use Cuda\CudaArray;
use Cuda\ContiguousArray;

final class ArrayPerformanceSuite
{
    private const FLOAT_SIZE = 4;
    private const ITERATIONS = 5;
    private const RANDOM_SAMPLES = 1_000_000;

    public static function run(): void
    {
        self::warmup();

        $scenarios = [
            '1D_HUGE'            => [10_000_000],
            '2D_SQUARE'          => [4000, 4000],

            '2D_WIDE_ROW'        => [1, 8_000_000],
            '2D_TALL_COL'        => [8_000_000, 1],

            '3D_TALL'            => [1024, 1024, 16],
            '3D_FLAT'            => [16, 1024, 1024],

            '3D_CUBE'            => [512, 512, 512],
            '4D_SMALL'           => [20, 20, 20, 20],
            '5D_STRESS'          => [20, 20, 20, 20, 20],
        ];

        echo "\nCUDA ContiguousArray – High Resolution Benchmark\n";
        echo "PHP: " . PHP_VERSION . " | OS: " . PHP_OS . "\n";
        echo "Memory limit: " . ini_get('memory_limit') . "\n";
        echo str_repeat("=", 110) . "\n";

        foreach ($scenarios as $label => $dims) {
            self::runScenario($label, $dims);
        }
    }

    private static function warmup(): void
    {
        $dims = [256, 256];
        $cuda = CudaArray::rand($dims);
        $cont = $cuda->toHost();
        $php  = $cont->toArray();

        for ($i = 0; $i < 3; $i++) {
            self::traverseAllAt($cont, $dims);
            self::traverseAllNative($php, $dims);
        }

        unset($cuda, $cont, $php);
        self::flush();
    }

    private static function runScenario(string $label, array $dims): void
    {
        $totalElements = (int) array_product($dims);
        $rawBytes = $totalElements * self::FLOAT_SIZE;

        echo "\n[SCENARIO: $label]\n";
        echo "Shape        : " . implode('x', $dims) . "\n";
        echo "Elements     : " . number_format($totalElements) . "\n";
        echo "Raw size     : " . self::formatBytes($rawBytes) . "\n";
        echo str_repeat("-", 110) . "\n";

        $cuda = CudaArray::rand($dims);
        self::flush();
        $mem0 = memory_get_usage();

        $t0 = microtime(true);
        $contiguous = $cuda->toHost();
        $tTransfer = microtime(true) - $t0;

        $memCont = memory_get_usage() - $mem0;

        self::flush();
        $mem1 = memory_get_usage();

        $t0 = microtime(true);
        $phpArray = $contiguous->toArray();
        $tMaterialize = microtime(true) - $t0;

        $memPhp = memory_get_usage() - $mem1;

        $seqAtNs       = self::benchNs(fn() => self::traverseAllAt($contiguous, $dims));
        $seqBracketNs  = self::benchNs(fn() => self::traverseAllBracket($contiguous, $dims));
        $seqPhpNs      = self::benchNs(fn() => self::traverseAllNative($phpArray, $dims));

        $rndAtNs       = self::benchNs(fn() => self::randomAccessAt($contiguous, $dims));
        $rndPhpNs      = self::benchNs(fn() => self::randomAccessNative($phpArray, $dims));

        printf("Transfer GPU → Host    : %.4f s\n", $tTransfer);
        printf("Materialize Host → PHP : %.4f s\n", $tMaterialize);

        echo "\nMemory usage:\n";
        printf(
            "  ContiguousArray      : %s (%.2f B/elem)\n",
            self::formatBytes($memCont),
            $memCont / $totalElements
        );
        printf(
            "  PHP array            : %s (%.2f B/elem)\n",
            self::formatBytes($memPhp),
            $memPhp / $totalElements
        );

        echo "\nSequential access (100% traversal):\n";
        self::printAccessNs($seqAtNs, $totalElements, 'Contiguous::at');
        self::printAccessNs($seqBracketNs, $totalElements, 'Contiguous[][]');
        self::printAccessNs($seqPhpNs, $totalElements, 'PHP[][]');

        echo "\nRandom access (" . number_format(self::RANDOM_SAMPLES) . " samples):\n";
        self::printAccessNs($rndAtNs, self::RANDOM_SAMPLES, 'Contiguous::at');
        self::printAccessNs($rndPhpNs, self::RANDOM_SAMPLES, 'PHP[][]');

        unset($cuda, $contiguous, $phpArray);
        self::flush();
    }

    private static function traverseAllAt(ContiguousArray $arr, array $dims): float
    {
        $sum = 0.0;
        $idx = array_fill(0, count($dims), 0);

        while (true) {
            $sum += $arr->at(...$idx);

            for ($d = count($dims) - 1; $d >= 0; $d--) {
                $idx[$d]++;
                if ($idx[$d] < $dims[$d]) {
                    break;
                }
                if ($d === 0) {
                    return $sum;
                }
                $idx[$d] = 0;
            }
        }
    }

    private static function traverseAllBracket(ContiguousArray $arr, array $dims): float
    {
        return self::traverseRecursive($arr, $dims);
    }

    private static function traverseAllNative(array $arr, array $dims): float
    {
        return self::traverseRecursive($arr, $dims);
    }

    private static function traverseRecursive($arr, array $dims, int $depth = 0): float
    {
        if ($depth === count($dims)) {
            return $arr;
        }

        $sum = 0.0;
        for ($i = 0; $i < $dims[$depth]; $i++) {
            $sum += self::traverseRecursive($arr[$i], $dims, $depth + 1);
        }
        return $sum;
    }

    private static function randomAccessAt(ContiguousArray $arr, array $dims): float
    {
        $sum = 0.0;
        $rank = count($dims);

        for ($i = 0; $i < self::RANDOM_SAMPLES; $i++) {
            $idx = [];
            for ($d = 0; $d < $rank; $d++) {
                $idx[] = random_int(0, $dims[$d] - 1);
            }
            $sum += $arr->at(...$idx);
        }
        return $sum;
    }

    private static function randomAccessNative(array $arr, array $dims): float
    {
        $sum = 0.0;
        $rank = count($dims);

        for ($i = 0; $i < self::RANDOM_SAMPLES; $i++) {
            $ref = $arr;
            for ($d = 0; $d < $rank; $d++) {
                $ref = $ref[random_int(0, $dims[$d] - 1)];
            }
            $sum += $ref;
        }
        return $sum;
    }

    private static function benchNs(callable $fn): int
    {
        for ($i = 0; $i < 2; $i++) {
            $fn();
        }

        $times = [];
        for ($i = 0; $i < self::ITERATIONS; $i++) {
            $t0 = hrtime(true);
            $fn();
            $times[] = hrtime(true) - $t0;
        }

        return (int) (array_sum($times) / count($times));
    }

    private static function printAccessNs(int $ns, int $elems, string $label): void
    {
        printf(
            "  %-18s : %.4f s (%.2f ns/elem)\n",
            $label,
            $ns / 1e9,
            $ns / $elems
        );
    }

    private static function formatBytes(int $bytes): string
    {
        $u = ['B', 'KB', 'MB', 'GB'];
        $p = (int) floor(log(max($bytes, 1), 1024));
        return round($bytes / (1024 ** $p), 2) . ' ' . $u[$p];
    }

    private static function flush(): void
    {
        gc_collect_cycles();
        gc_mem_caches();
    }
}

ArrayPerformanceSuite::run();
