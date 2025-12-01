<?php

if (!extension_loaded('cuda')) {
    die("Extension cuda not loaded. Read README.md to compile.\n");
}

class CudaBenchmark
{
    const RUNS = 3;
    private array $fastestOp = [];
    private array $slowestOp = [];
    private int $totalElements = 0;
    private int $totalOp = 0;
    private float $totalTime = 0;
    private float $tbeMinGlobal = INF;

    public function run()
    {
        echo "CUDA PHP EXTENSION - FULL BENCHMARK SUITE\n";
        echo "Running " . self::RUNS . " passes per operation.\n";

        $nameFormatted = $this->formatName("Operation");
        $labelFormatted = str_pad("Shape", 22);
        $elemFormatted = str_pad("elements", 12);

        echo str_repeat("=", 85) . "\n";
        $gpuFormatted = str_pad("Time", 10, " ", STR_PAD_RIGHT);
        echo " - {$nameFormatted} | {$labelFormatted} | {$elemFormatted}       | $gpuFormatted\n";

        $tests = [
            'SMALL TENSORS (16K - 1M)' => [
                '16x16x16' => [16, 16, 16],
                '64x64x64' => [64, 64, 64],
                '128x128x64' => [128, 128, 64],
            ],
            'MEDIUM MATRICES (2M - 4M)' => [
                '1024x1024x1' => [1024, 1024, 1],
                '128x128x128' => [128, 128, 128],
                '256x256x64' => [256, 256, 64],
            ],
            'LARGE MATRICES (4M - 32M)' => [
                '512x512x16' => [512, 512, 16],
                '1024x1024x4' => [1024, 1024, 4],
                '1024x1024x32' => [1024, 1024, 32],
            ],
        ];

        foreach ($tests as $suiteName => $dims) {
            echo "\n=== SUITE: $suiteName ===\n";
            $this->suiteElementWise($dims);
            $this->suiteBinaryOps($dims);
            $this->suiteMatmul($dims);
            $this->suiteReduction($dims);
            $this->suiteConcat($dims);
            $this->suiteComparison($dims);
            $this->suiteToArray($dims);
            $this->suiteView($dims);
        }

        $time = round($this->totalTime, 3);
        $elements = number_format($this->totalElements);

        echo "\n\nOVERVIEW: \n\tTotal time: {$time} s\n\tFastest op: \033[32m{$this->fastestOp['op']}\033[0m\n\tSlowest op: \033[91m{$this->slowestOp['op']}\033[0m";
        echo "\n\tTotal elements: {$elements}\n\tTotal operations: {$this->totalOp}";
        echo "\n\nBENCHMARK FINISHED!\n";

    }

    private function suiteElementWise($tests)
    {
        echo "\n[ ELEMENT-WISE OPERATIONS ]\n";

        $ops = [
            'Add (+)' => fn($x) => $x + 1,
            'Mul (*)' => fn($x) => $x * 2.0,
            'Sub (-)' => fn($x) => $x - 0.5,
            'Pow (**)' => fn($x) => $x ** 2,
            'Div (/)' => fn($x) => $x / 2,
            'Sqrt' => fn($x) => $x->sqrt(),
            'Abs' => fn($x) => $x->abs(),
            'Sin' => fn($x) => $x->sin(),
            'Cos' => fn($x) => $x->cos(),
            'Neg' => fn($x) => $x->neg(),
            'Exp' => fn($x) => $x->exp(),
            'Log' => fn($x) => $x->log(),
        ];

        foreach ($ops as $name => $gpuFn)
            foreach ($tests as $label => $dims)
                $this->runOp($name, $label, $dims, $gpuFn, null, unary: true);
    }

    private function suiteBinaryOps($tests)
    {
        echo "\n[ BINARY OPERATIONS (A OP B) ]\n";

        $ops = [
            'Add' => fn($a, $b) => $a + $b,
            'Sub' => fn($a, $b) => $a - $b,
            'Mul' => fn($a, $b) => $a * $b,
            'Div' => fn($a, $b) => $a / $b,
        ];

        foreach ($ops as $name => $gpuFn)
            foreach ($tests as $label => $dims)
                $this->runOp($name, "$label (A vs B)", $dims, $gpuFn, null, binary: true);
    }

    private function suiteView($tests)
    {
        echo "\n[ View Creation ]\n";

        $ops = [
            'reshape (n)' => function ($a) {
                [$x, $y, $z] = $a->getShape();
                return $a->reshape([$x * $y * $z]);
            },
            'reshape (n, z)' => function ($a) {
                [$x, $y, $z] = $a->getShape();
                return $a->reshape([$x * $y, $z]);
            },
            'reshape (n, x)' => function ($a) {
                [$x, $y, $z] = $a->getShape();
                return $a->reshape([$y * $z, $x]);
            },
            'flatten' => function ($a) {
                return $a->flatten();
            },
            'transpose' => function ($a) {
                return $a->transpose([2, 1, 0]);
            },
            't[0]' => function ($a) {
                return $a[0];
            },
            't[[x, x - 1]]' => function ($a) {
                [$x] = $a->getShape();
                return $a[[0, $x - 1]];
            },
            't(null, 0, [z, z -1] )' => function ($a) {
                [$x, $y, $z] = $a->getShape();
                return $a(null, 0, [0, $z - 1]);
            },
            'iterate a[i]' => function ($a) {
                [$x, $y, $z] = $a->getShape();
                for ($i = 0; $i < $x; $i++) {
                    $result = $a[$i];
                }
            },
        ];

        foreach ($ops as $name => $gpuFn)
            foreach ($tests as $label => $dims)
                $this->runOp($name, "$label)", $dims, $gpuFn, null);
    }

    private function suiteReduction($tests)
    {
        echo "\n[ REDUCTIONS ]\n";

        $ops = [
            'Sum axis=null' => fn($a) => $a->sum(axis: null),
            'Sum axis=0' => fn($a) => $a->sum(axis: 0),
            'Max axis=null' => fn($a) => $a->max(axis: null),
            'Max axis=0' => fn($a) => $a->max(axis: 0),
            'Min axis=null' => fn($a) => $a->min(axis: null),
            'Min axis=0' => fn($a) => $a->min(axis: 0),
            'Sum axis=1' => fn($a) => $a->sum(axis: 1),
            'Max axis=1' => fn($a) => $a->max(axis: 1),
            'Min axis=1' => fn($a) => $a->min(axis: 1),
            'Sum axis=2' => fn($a) => $a->sum(axis: 2),
            'Max axis=2' => fn($a) => $a->max(axis: 2),
            'Min axis=2' => fn($a) => $a->min(axis: 2),
            'argMax axis=null' => fn($a) => $a->argMax(axis: null),
            'argMax axis=0' => fn($a) => $a->argMax(axis: 0),
            'argMax axis=1' => fn($a) => $a->argMax(axis: 1),
            'argMax axis=2' => fn($a) => $a->argMax(axis: 2),
            'argMin axis=null' => fn($a) => $a->argMin(axis: null),
            'argMin axis=0' => fn($a) => $a->argMin(axis: 0),
            'argMin axis=1' => fn($a) => $a->argMin(axis: 1),
            'argMin axis=' => fn($a) => $a->argMin(axis: 2),

        ];

        foreach ($ops as $name => $gpuFn)
            foreach ($tests as $label => $dims)
                $this->runOp($name, $label, $dims, $gpuFn, null, reduction: true);
    }


    private function suiteMatmul($tests)
    {
        echo "\n[ Matrix multiplication ]\n";

        $ops = [
            'MATMUL' => function ($a, $b) {
                return $b->matmul($a);
            }

        ];

        foreach ($ops as $name => $gpuFn)
            foreach ($tests as $label => $dims)
                $this->runOp($name, $label, $dims, $gpuFn, null);

    }

    private function suiteToArray($tests)
    {
        echo "\n[ GPU to CPU transfer ]\n";

        $ops = [
            'toArray' => function ($a) {
                return $a->toArray();
            },
            '__construct' => function ($a) {
                [$x, $y, $z] = $a->getShape();
                return new Cuda\CudaArray(array_fill(0, $x - 1, array_fill(0, $y - 1, array_fill(0, $z - 1, 10))));
            },
            'ones' => fn($a) => Cuda\CudaArray::ones($a->getShape()),
            'zeros' => fn($a) => Cuda\CudaArray::zeros($a->getShape()),
            'full' => fn($a) => Cuda\CudaArray::full($a->getShape(), 1.5),
            'rand' => fn($a) => Cuda\CudaArray::rand($a->getShape(), -1, 1),
            'clone' => fn($a) => clone $a,

        ];

        foreach ($ops as $name => $gpuFn)
            foreach ($tests as $label => $dims)
                $this->runOp($name, $label, $dims, $gpuFn, null, reduction: true);

    }


    private function suiteConcat($tests)
    {
        echo "\n[ CONCAT OPERATIONS ]\n";

        foreach ($tests as $label => $dims) {
            [$r, $c, $d] = $dims;

            if ($c < 2)
                continue;

            $this->runOp(
                "Concat axis=0",
                $label,
                $dims,
                fn($a, $b) => $a->concat([$b], axis: 0),
                null,
                binary: true
            );

            $this->runOp(
                "Concat axis=1",
                $label,
                $dims,
                fn($a, $b) => $a->concat([$b], axis: 1),
                null,
                binary: true
            );

            $this->runOp(
                "Concat axis=2",
                $label,
                $dims,
                fn($a, $b) => $a->concat([$b], axis: 2),
                null,
                binary: true
            );
        }
    }

    private function suiteComparison($tests)
    {
        echo "\n[ COMPARISON OPERATIONS ]\n";

        $ops = [
            'EQ scalar' => fn($a) => $a->eq(1),
            'EQ tensor' => fn($a, $b) => $a->eq($b),
            'GT scalar' => fn($a) => $a->gt(0.5),
            'LT scalar' => fn($a) => $a->lt(5),
            'GT tensor' => fn($a, $b) => $a->gt($b),
            'NE Scalar' => fn($a, $b) => $a->ne(0.5),
            'NE tensor' => fn($a, $b) => $a->ne($b),
            'LE Scalar' => fn($a, $b) => $a->le(0.5),
            'LE tensor' => fn($a, $b) => $a->le($b),
        ];

        foreach ($ops as $name => $gpuFn)
            foreach ($tests as $label => $dims)
                $this->runOp($name, $label, $dims, $gpuFn, null, unary: !str_contains($name, 'tensor'), binary: str_contains($name, 'tensor'));
    }

    private function formatName($opName)
    {
        $len = 25;
        $label = strtoupper($opName);
        return str_pad($label, $len);
    }

    private function runOp(
        string $opName,
        string $label,
        array $dims,
        callable $gpuFn,
        ?callable $cpuFn,
        bool $unary = false,
        bool $binary = false,
        bool $reduction = false
    ) {
        [$x, $y, $z] = $dims;
        $count = $x * $y * $z;
        $elemCount = number_format($count);

        $this->totalElements += $count;

        $nameFormatted = $this->formatName($opName);
        $labelFormatted = str_pad($label, 22);
        $elemFormatted = str_pad($elemCount, 12);

        echo " - {$nameFormatted} | {$labelFormatted} | {$elemFormatted} elems | ";

        try {

            $a = Cuda\CudaArray::rand($dims, 0.0, 1.0);
            $b = Cuda\CudaArray::rand($dims, 0.0, 1.0);

            if ($opName === "MATMUL") {
                $b = $b->transpose([0, 2, 1]);
            }

            for ($i = 0; $i < self::RUNS; $i++) {
                $gpuFn($a, $b);
            }

            $t0 = microtime(true);
            for ($i = 0; $i < self::RUNS; $i++) {
                $gpuFn($a, $b);
            }

            $time = microtime(true) - $t0;
            $this->totalTime += $time;

            $gpuTime = $time * 1000 / self::RUNS;

            $time = round($gpuTime, 3);
            $gpuFormatted = str_pad("{$time} ms", 10, " ", STR_PAD_LEFT);

            if (!isset($this->slowestOp['time']) || $this->slowestOp['time'] < $time) {
                $this->slowestOp = [
                    "time" => $time,
                    "op" => "$opName ({$time} ms)"
                ];
            }

            if (!isset($this->fastestOp['time']) || $this->fastestOp['time'] > $time) {
                $this->fastestOp = [
                    "time" => $time,
                    "op" => "$opName ({$time} ms)"
                ];
            }

            $tbe = ($gpuTime / $count) * 1000000;

            $yellowLimit = $this->tbeMinGlobal * 1.25;

            $redLimit = $this->tbeMinGlobal * 2.0;

            $color = "\033[31m";

            if ($tbe <= $yellowLimit) {
                $color = "\033[32m";
            } elseif ($tbe <= $redLimit) {
                $color = "\033[33m";
            }
            $this->tbeMinGlobal = min($this->tbeMinGlobal, $tbe);

            $color = "\033[32m";
            if ($tbe < 1.25) {
                $color = "\033[32m";
            } elseif ($tbe < 2.0) {
                $color = "\033[33m";
            } else {
                $color = "\033[31m";
            }

            echo "GPU: {$color}{$gpuFormatted}\033[0m\n";
        } catch (Throwable $e) {
            var_dump($e);
            exit;
            echo "GPU: SKIP\n";
        }

        $this->totalOp++;
    }
}

$benchmark = new CudaBenchmark();
$benchmark->run();
