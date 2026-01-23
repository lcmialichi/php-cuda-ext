<?php

use Cuda\ContiguousArray;
use Cuda\CudaArray;

class MultiDimBenchmark
{
    public static function run()
    {
        $scenarios = [
            "1D" => [1000000],
            "2D" => [1000, 1000],
            "3D" => [100, 100, 100],
            "3D (medium)" => [1024, 512, 64],
            "3D (large)" => [2048, 1024, 64],
        ];

        foreach ($scenarios as $name => $dims) {
            self::testScenario($name, $dims);
        }
    }

    private static function testScenario($name, $dims)
    {
        echo "\n=== $name [" . implode('x', $dims) . "] ===\n";

        $cuda = CudaArray::rand($dims);
        $cArray = $cuda->toHost();
        $ndims = count($dims);

        self::bench("Access via []", function () use ($cArray, $dims, $ndims) {
            if ($ndims == 1) {
                for ($i = 0; $i < $dims[0]; $i++)
                    $v = $cArray[$i];
            } elseif ($ndims == 2) {
                for ($i = 0; $i < $dims[0]; $i++)
                    for ($j = 0; $j < $dims[1]; $j++)
                        $v = $cArray[$i][$j];
            } else {
                for ($i = 0; $i < $dims[0]; $i++)
                    for ($j = 0; $j < $dims[1]; $j++)
                        $v = $cArray[$i][$j][0];
            }
        });

        self::bench("Access via get([])", function () use ($cArray, $dims, $ndims) {
            if ($ndims == 1) {
                for ($i = 0; $i < $dims[0]; $i++)
                    $v = $cArray->get([$i]);
            } elseif ($ndims == 2) {
                for ($i = 0; $i < $dims[0]; $i++)
                    for ($j = 0; $j < $dims[1]; $j++)
                        $v = $cArray->get([$i, $j]);
            } else {
                for ($i = 0; $i < $dims[0]; $i++)
                    for ($j = 0; $j < $dims[1]; $j++)
                        $v = $cArray->get([$i, $j, 0]);
            }
        });

        self::bench("Access via at(...)", function () use ($cArray, $dims, $ndims) {
            if ($ndims == 1) {
                for ($i = 0; $i < $dims[0]; $i++)
                    $v = $cArray->at($i);
            } elseif ($ndims == 2) {
                for ($i = 0; $i < $dims[0]; $i++)
                    for ($j = 0; $j < $dims[1]; $j++)
                        $v = $cArray->at($i, $j);
            } else {
                for ($i = 0; $i < $dims[0]; $i++)
                    for ($j = 0; $j < $dims[1]; $j++)
                        $v = $cArray->at($i, $j, 0);
            }
        });

        echo str_repeat("-", 50) . "\n";
    }

    private static function bench($name, $cb, $iters = 3)
    {
        $start = microtime(true);
        for ($i = 0; $i < $iters; $i++)
            $cb();
        $end = microtime(true);
        printf("%-25s: %8.5f s (avg)\n", $name, ($end - $start) / $iters);
    }
}

MultiDimBenchmark::run();