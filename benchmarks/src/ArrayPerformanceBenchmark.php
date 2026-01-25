<?php

declare(strict_types=1);

require_once __DIR__ . '/AbstractBenchmark.php';

use Cuda\CudaArray;
use Cuda\ContiguousArray;

class ArrayPerformanceBenchmark extends AbstractBenchmark
{
    public function getName(): string
    {
        return 'Array Performance Benchmark';
    }

    public function getDescription(): string
    {
        return 'Compara performance de transferência e acesso entre GPU, ContiguousArray e arrays PHP';
    }

    public function run(): array
    {
        $scenarios = $this->config['scenarios']['memory_intensive'] ?? [
            '1D_HUGE' => [10_000_000],
            '2D_SQUARE' => [4000, 4000],
            '2D_WIDE_ROW' => [1, 8_000_000],
            '2D_TALL_COL' => [8_000_000, 1],
            '3D_CUBE' => [512, 512, 512],
        ];

        $iterations = $this->config['iterations']['standard'] ?? 5;

        $this->warmup(function () {
            $temp = CudaArray::rand([100, 100]);
            $temp->toHost()->toArray();
        });

        foreach ($scenarios as $label => $dims) {
            $this->runScenario($label, $dims, $iterations);
        }

        return $this->getResults();
    }

    private function runScenario(string $label, array $dims, int $iterations): void
    {
        $totalElements = (int) array_product($dims);

        echo "Running scenario: {$label} (" . implode('x', $dims) . ")\n";

        $this->benchmarkOperation(
            "ContiguousArray::toHost()",
            function () use ($dims) {
                $cuda = CudaArray::rand($dims);
                $contiguous = $cuda->toHost();
                unset($cuda, $contiguous);
            },
            metadata: [
                'scenario' => $label,
                'shape' => implode('x', $dims),
                'elements' => $totalElements,
                'type' => 'transfer'
            ],
            iterations: $iterations
        );

        $cuda = CudaArray::rand($dims);
        $contiguous = $cuda->toHost();

        $this->benchmarkOperation(
            "ContiguousArray::toArray()",
            function () use ($contiguous) {
                $phpArray = $contiguous->toArray();
                unset($phpArray);
            },
            metadata: [
                'scenario' => $label,
                'shape' => implode('x', $dims),
                'elements' => $totalElements,
                'type' => 'materialization'
            ],
            iterations: $iterations
        );

        $phpArray = $contiguous->toArray();

        $operations = [
            [
                'name' => 'ContiguousArray::at()',
                'callback' => fn() => $this->traverseAllAt($contiguous, $dims),
                'native' => null,
                'type' => 'sequential'
            ],
            [
                'name' => 'ContiguousArray[]',
                'callback' => fn() => $this->traverseAllBracket($contiguous, $dims),
                'native' => fn() => $this->traverseAllNative($phpArray, $dims),
                'type' => 'sequential'
            ],
            [
                'name' => 'ContiguousArray::at() - Random',
                'callback' => fn() => $this->randomAccessAt($contiguous, $dims),
                'native' => fn() => $this->randomAccessNative($phpArray, $dims),
                'type' => 'random',
                'samples' => self::RANDOM_SAMPLES
            ],
        ];

        foreach ($operations as $op) {
            $metadata = [
                'scenario' => $label,
                'shape' => implode('x', $dims),
                'elements' => $totalElements,
                'type' => $op['type']
            ];

            if (isset($op['samples'])) {
                $metadata['samples'] = $op['samples'];
            }

            $this->benchmarkOperation(
                $op['name'],
                $op['callback'],
                $op['native'],
                metadata: $metadata,
                iterations: $iterations
            );
        }

        unset($cuda, $contiguous, $phpArray);
        $this->flush();
    }

    private function traverseAllAt(ContiguousArray $arr, array $dims): float
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
    

    private function traverseAllBracket(ContiguousArray $arr, array $dims): float
    {
        return $this->traverseRecursive($arr, $dims);
    }

    private function traverseAllNative(array $arr, array $dims): float
    {
        return $this->traverseRecursive($arr, $dims);
    }

    private function traverseRecursive($arr, array $dims, int $depth = 0): float
    {
        if ($depth === count($dims)) {
            return $arr;
        }

        $sum = 0.0;
        for ($i = 0; $i < $dims[$depth]; $i++) {
            $sum += $this->traverseRecursive($arr[$i], $dims, $depth + 1);
        }
        return $sum;
    }

    private function randomAccessAt(ContiguousArray $arr, array $dims): float
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

    private function randomAccessNative(array $arr, array $dims): float
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
}