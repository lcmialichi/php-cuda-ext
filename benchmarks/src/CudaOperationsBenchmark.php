<?php

declare(strict_types=1);

require_once __DIR__ . '/AbstractBenchmark.php';


use Cuda\CudaArray;

class CudaOperationsBenchmark extends AbstractBenchmark
{
    private const RUNS = 3;

    public function getName(): string
    {
        return 'CUDA Operations Benchmark';
    }

    public function getDescription(): string
    {
        return 'Benchmark de operações de tensor na GPU';
    }

    public function run(): array
    {
        $scenarios = $this->config['scenarios']['tensor_operations'] ?? [
            'SMALL_16x16x16' => [16, 16, 16],
            'MEDIUM_64x64x64' => [64, 64, 64],
            'LARGE_128x128x128' => [128, 128, 128],
            'XLARGE_256x256x64' => [256, 256, 64],
        ];

        $this->runElementWiseOperations($scenarios);
        $this->runBinaryOperations($scenarios);
        $this->runReductionOperations($scenarios);
        $this->runMatrixOperations($scenarios);
        $this->runTransformOperations($scenarios);

        return $this->getResults();
    }

    private function runElementWiseOperations(array $scenarios): void
    {
        $operations = [
            'Add' => fn($x) => $x + 1,
            'Multiply' => fn($x) => $x * 2.0,
            'Subtract' => fn($x) => $x - 0.5,
            'Power' => fn($x) => $x ** 2,
            'Sqrt' => fn($x) => $x->sqrt(),
            'Abs' => fn($x) => $x->abs(),
            'Sin' => fn($x) => $x->sin(),
            'Cos' => fn($x) => $x->cos(),
            'Exp' => fn($x) => $x->exp(),
            'Log' => fn($x) => $x->log(),
        ];

        foreach ($operations as $opName => $operation) {
            foreach ($scenarios as $label => $dims) {
                $this->benchmarkOperation(
                    "CudaArray::{$opName}()",
                    function () use ($operation, $dims) {
                        $a = CudaArray::rand($dims, 0.0, 1.0);
                        $result = $operation($a);
                        unset($a, $result);
                    },
                    metadata: [
                        'operation_type' => 'element_wise',
                        'shape' => implode('x', $dims),
                        'elements' => array_product($dims),
                        'scenario' => $label
                    ],
                    iterations: self::RUNS
                );
            }
        }
    }

    private function runBinaryOperations(array $scenarios): void
    {
        $operations = [
            'add' => fn($a, $b) => $a + $b,
            'subtract' => fn($a, $b) => $a - $b,
            'multiply' => fn($a, $b) => $a * $b,
            'divide' => fn($a, $b) => $a / $b,
        ];

        foreach ($operations as $opName => $operation) {
            foreach ($scenarios as $label => $dims) {
                $this->benchmarkOperation(
                    "CudaArray::{$opName}()",
                    function () use ($operation, $dims) {
                        $a = CudaArray::rand($dims, 0.0, 1.0);
                        $b = CudaArray::rand($dims, 0.0, 1.0);
                        $result = $operation($a, $b);
                        unset($a, $b, $result);
                    },
                    metadata: [
                        'operation_type' => 'binary',
                        'shape' => implode('x', $dims),
                        'elements' => array_product($dims),
                        'scenario' => $label
                    ],
                    iterations: self::RUNS
                );
            }
        }
    }

    private function runReductionOperations(array $scenarios): void
    {
        $operations = [
            'sum' => fn($a) => $a->sum(),
            'max' => fn($a) => $a->max(),
            'min' => fn($a) => $a->min(),
        ];

        $axes = [null, 0, 1, 2];

        foreach ($operations as $opName => $operation) {
            foreach ($axes as $axis) {
                foreach ($scenarios as $label => $dims) {
                    if ($axis !== null && $axis >= count($dims)) {
                        continue;
                    }

                    $a = CudaArray::rand($dims, 0.0, 1.0);

                    $this->benchmarkOperation(
                        "CudaArray::{$opName}(axis:{$axis})",
                        function () use ($operation, $axis, $dims, $a) {
                            $operation($a, $axis);
                        },
                        metadata: [
                            'operation_type' => 'reduction',
                            'shape' => implode('x', $dims),
                            'elements' => array_product($dims),
                            'axis' => $axis,
                            'scenario' => $label
                        ],
                        iterations: self::RUNS
                    );

                    unset($a, $result);
                }
            }
        }
    }

    private function runMatrixOperations(array $scenarios): void
    {
        $matrixShapes = [
            '32x32' => [32, 32],
            '128x128' => [128, 128],
            '512x512' => [512, 512],
        ];

        foreach ($matrixShapes as $label => $shape) {
            $a = CudaArray::rand($shape, 0.0, 1.0);
            $b = CudaArray::rand($shape, 0.0, 1.0);

            $this->benchmarkOperation(
                "CudaArray::matmul() - {$label}",
                function () use ($shape, $a, $b) {
                    $a->matmul($b);
                },
                metadata: [
                    'operation_type' => 'matrix',
                    'shape' => implode('x', $shape),
                    'elements' => array_product($shape),
                    'scenario' => $label
                ],
                iterations: self::RUNS
            );

            unset($a, $b);
        }
    }

    private function runTransformOperations(array $scenarios): void
    {
        $operations = [
            'reshape' => function ($a) {
                $shape = $a->getShape();
                $total = array_product($shape);
                return $a->reshape([$total]);
            },
            'transpose' => fn($a) => $a->transpose(),
            'flatten' => fn($a) => $a->flatten(),
            'concatenate' => function ($a) {
                return $a->concat([CudaArray::rand($a->getShape(), 0.0, 1.0)], axis: 0);
            },
        ];

        foreach ($operations as $opName => $operation) {
            foreach ($scenarios as $label => $dims) {
                $a = CudaArray::rand($dims, 0.0, 1.0);
                $this->benchmarkOperation(
                    "CudaArray::{$opName}()",
                    function () use ($operation, $dims, $a) {
                        $result = $operation($a);
                    },
                    metadata: [
                        'operation_type' => 'transform',
                        'shape' => implode('x', $dims),
                        'elements' => array_product($dims),
                        'scenario' => $label
                    ],
                    iterations: self::RUNS
                );

                unset($a, $result);
            }
        }
    }
}