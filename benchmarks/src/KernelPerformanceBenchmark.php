<?php

declare(strict_types=1);

require_once __DIR__ . '/AbstractBenchmark.php';

use Cuda\Attr as Attr;
use Cuda\Compiler;
use Cuda\CudaArray;

class KernelPerformanceBenchmark extends AbstractBenchmark
{
    private array $kernels = [];
    private $compiler;
    private $module;

    public function getName(): string
    {
        return 'Kernel Performance Benchmark';
    }

    public function getDescription(): string
    {
        return 'Benchmark de compilação e execução de kernels CUDA';
    }

    public function run(): array
    {
        $this->registerKernels();

        $this->benchmarkCompilationPhase();
        $this->benchmarkExecutionPhase();
        $this->benchmarkAsyncOperations();

        return $this->getResults();
    }

    private function registerKernels(): void
    {
        $kernelClass = new class {
            #[Attr\Kernel(name: 'vector_add')]
            public function vectorAdd(
                #[Attr\TensorType] array $a,
                #[Attr\TensorType] array $b,
                #[Attr\TensorType] array &$c,
                #[Attr\IntType] int $n
            ): void {
                /** @var \Cuda\Runtime $cuda */
                $idx = $cuda->globalIdx();
                if ($idx < $n) {
                    $c[$idx] = $a[$idx] + $b[$idx];
                }
            }

            #[Attr\Kernel(name: 'vector_mul')]
            public function vectorMul(
                #[Attr\TensorType] array $a,
                #[Attr\TensorType] array $b,
                #[Attr\TensorType] array &$c,
                #[Attr\IntType] int $n
            ): void {
                /** @var \Cuda\Runtime $cuda */
                $idx = $cuda->globalIdx();
                if ($idx < $n) {
                    $c[$idx] = $a[$idx] * $b[$idx];
                }
            }

            #[Attr\Kernel(name: 'sigmoid')]
            public function sigmoid(
                #[Attr\TensorType] array $in,
                #[Attr\TensorType] array &$out,
                #[Attr\IntType] int $n
            ): void {
                /** @var \Cuda\Runtime $cuda */
                $idx = $cuda->globalIdx();
                if ($idx < $n) {
                    $out[$idx] = 1.0 / (1.0 +  $cuda->math->exp(-$in[$idx]));
                }
            }

            
        };

        $this->kernels = [
            'vector_add' => [$kernelClass, 'vectorAdd'],
            'vector_mul' => [$kernelClass, 'vectorMul'],
            'sigmoid' => [$kernelClass, 'sigmoid'],
        ];
    }

    private function benchmarkCompilationPhase(): void
    {
        $this->benchmarkOperation(
            'AST_Generation',
            function () {
                $this->compiler = new Compiler();
                foreach ($this->kernels as $kernel) {
                    $this->compiler->kernel($kernel);
                }
            },
            metadata: ['phase' => 'compilation', 'kernels' => count($this->kernels)]
        );

        $this->benchmarkOperation(
            'PTX_Compilation',
            function () {
                $this->module = $this->compiler->compile();
            },
            metadata: ['phase' => 'compilation']
        );

        $this->benchmarkOperation(
            'JIT_Initialization',
            function () {
                $this->module->initialize();
            },
            metadata: ['phase' => 'initialization']
        );
    }

    private function benchmarkExecutionPhase(): void
    {
        $scenarios = $this->config['scenarios']['kernel_operations'] ?? [
            'VECTOR_1K' => [1024],
            'VECTOR_10K' => [10240],
            'VECTOR_100K' => [102400],
            'VECTOR_1M' => [1024 * 1024],
        ];

        foreach ($scenarios as $label => $shape) {
            $n = $shape[0];
            $config = ['block' => [256, 1, 1], 'grid' => [(int) ceil($n / 256), 1, 1]];

            foreach ($this->kernels as $kernelName => $_) {
                $a = CudaArray::rand($shape, -1.0, 1.0);
                $b = CudaArray::rand($shape, -1.0, 1.0);
                $c = CudaArray::zeros($shape);

                $args = match ($kernelName) {
                    'sigmoid' => [$a, $c, $n],
                    default => [$a, $b, $c, $n]
                };

                $this->benchmarkOperation(
                    "ComipiledModule::run() - {$kernelName}_{$label}",
                    function () use ($kernelName, $args, $config) {
                        $this->module->run($kernelName, args: $args, config: $config);
                    },
                    metadata: [
                        'phase' => 'execution',
                        'kernel' => $kernelName,
                        'elements' => $n,
                        'block_size' => $config['block'][0],
                        'grid_size' => $config['grid'][0]
                    ]
                );

                unset($a, $b, $c);
            }
        }
    }

    private function benchmarkAsyncOperations(): void
    {
        $n = 1024 * 1024;
        $shape = [$n];
        $config = ['block' => [256, 1, 1], 'grid' => [(int) ceil($n / 256), 1, 1]];

        $batchSizes = [1, 5, 10, 20];

        foreach ($batchSizes as $batchSize) {
            $inputs = [];
            for ($i = 0; $i < $batchSize; $i++) {
                $inputs[] = [
                    'a' => CudaArray::rand($shape, -1.0, 1.0),
                    'b' => CudaArray::rand($shape, -1.0, 1.0),
                    'c' => CudaArray::zeros($shape),
                ];
            }

            $this->benchmarkOperation(
                "CompiledModule::run() - {$batchSize}",
                function () use ($inputs, $config) {
                    foreach ($inputs as $data) {
                        $this->module->run(
                            'vector_add',
                            args: [$data['a'], $data['b'], $data['c'], $data['a']->getSize()],
                            config: $config
                        );
                    }
                },
                metadata: [
                    'phase' => 'async_benchmark',
                    'mode' => 'sync',
                    'batch_size' => $batchSize,
                    'kernel' => 'vector_add'
                ]
            );

            $this->benchmarkOperation(
                "CompiledModule::runAsync() - {$batchSize}",
                function () use ($inputs, $config) {
                    foreach ($inputs as $data) {
                        $this->module->runAsync(
                            'vector_add',
                            args: [$data['a'], $data['b'], $data['c'], $data['a']->getSize()],
                            config: $config
                        );
                    }
                    $this->module->sync();
                },
                metadata: [
                    'phase' => 'async_benchmark',
                    'mode' => 'async',
                    'batch_size' => $batchSize,
                    'kernel' => 'vector_add'
                ]
            );

            foreach ($inputs as $data) {
                unset($data['a'], $data['b'], $data['c']);
            }
        }
    }
}