<?php

declare(strict_types=1);

require_once __DIR__ . '/benchmarks/src/ArrayPerformanceBenchmark.php';
require_once __DIR__ . '/benchmarks/src/CudaOperationsBenchmark.php';
require_once __DIR__ . '/benchmarks/src/KernelPerformanceBenchmark.php';
require_once __DIR__ . '/benchmarks/src/BenchmarkRunner.php';

if (!extension_loaded('cuda')) {
    die(" CUDA extension not loaded. Please compile and install the extension first.\n");
}

echo "Starting CUDA PHP Benchmark Suite\n";

$outputDir = __DIR__ . '/benchmarks/reports/generated';
if (!is_dir($outputDir)) {
    mkdir($outputDir, 0777, true);
}

$config = [
    'iterations' => [
        'quick' => 2,
        'standard' => 3,
        'thorough' => 5,
    ],
    'output' => [
        'format' => 'both',
        'output_dir' => $outputDir,
    ],
    'performance' => [
        'gc_enabled' => true,
        'precision' => 4,
    ]
];

$runner = new BenchmarkRunner($config);

$runner
    ->addBenchmark(new ArrayPerformanceBenchmark($config))
    ->addBenchmark(new CudaOperationsBenchmark($config))
    ->addBenchmark(new KernelPerformanceBenchmark($config));

try {
    $results = $runner->runAll();
    
    echo "\n" . str_repeat("=", 80) . "\n";
    echo " - BENCHMARK SUITE COMPLETED SUCCESSFULLY\n";
    echo str_repeat("=", 80) . "\n";
    
    $totalTime = 0;
    $totalOperations = 0;
    
    foreach ($results as $benchmarkName => $data) {
        $time = $data['total_time'] ?? 0;
        $operations = count($data['results'] ?? []);
        
        $totalTime += $time;
        $totalOperations += $operations;
        
        printf(" - %-30s: %4.1fs | %3d operations\n", 
            $benchmarkName, $time, $operations);
    }
    
    echo str_repeat("-", 80) . "\n";
    printf("TOTAL: %37.1fs | %3d operations\n", $totalTime, $totalOperations);
    echo str_repeat("=", 80) . "\n\n";
    
    echo " - Reports have been generated in: " . realpath($outputDir) . "\n";
    echo " - Tip: Open the HTML report in your browser for interactive charts\n\n";
    
} catch (Throwable $e) {
    echo "\n Error during benchmark execution:\n";
    echo "   " . $e->getMessage() . "\n";
    echo "   File: " . $e->getFile() . ":" . $e->getLine() . "\n";
    exit(1);
}