<?php

declare(strict_types=1);

require_once __DIR__ . "/vendor/autoload.php";

use Benchmarks\BenchmarkApplication;
use Benchmarks\Handlers\ContiguousArrayBenchmark;
use Benchmarks\Handlers\CudaArrayBasicMathOperations;
use Benchmarks\Handlers\CudaArrayLinearAlgebraBenchmark;
use Benchmarks\Handlers\CudaArrayMemoryCopyOperationsBenchmark;
use Benchmarks\Handlers\CudaArrayReductionBenchmark;
use Benchmarks\Handlers\CudaArrayShapeManiliplationBenchmark;

if (!extension_loaded('cuda')) {
    die(" CUDA extension not loaded. Please compile and install the extension first.\n");
}

$importOnly = in_array('--import', $argv, true);
$matmulOnly = in_array('--matmul', $argv, true);
if ($importOnly && $matmulOnly) {
    throw new InvalidArgumentException('Choose only one benchmark filter');
}
$benchmarks = $importOnly ? [new CudaArrayMemoryCopyOperationsBenchmark()] : ($matmulOnly ? [new CudaArrayLinearAlgebraBenchmark()] : [
    new CudaArrayBasicMathOperations(),
    new CudaArrayShapeManiliplationBenchmark(),
    new CudaArrayReductionBenchmark(),
    new CudaArrayMemoryCopyOperationsBenchmark(),
    new CudaArrayLinearAlgebraBenchmark(),
    // new ContiguousArrayBenchmark()
]);
$app = new BenchmarkApplication(
    $benchmarks,
    $importOnly ? ['cudaArrayImportPhp', 'cudaArrayImportBuffer', 'hostArrayToGpu', 'hostArrayPinnedToGpu', 'cudaArrayImportFile'] : ($matmulOnly ? ['cudaArrayMatmul'] : null)
);

$dir = __DIR__ . "/benchmarks/reports";

echo "Running PHP GPU Tensors benchmarks...\n";

$report = $app->run();
$jsonPath = $report->saveJSON($dir);
$htmlPath = $report->saveHTML($dir);

echo "┌──────────────────────────────────────────────────────────────────────────┐\n";
echo "│                            BENCHMARK REPORTS                             │\n";
echo "├──────────────────────────────────────────────────────────────────────────┤\n";
echo "│ 📁 JSON:  " . str_pad($jsonPath, 62) . " │\n";
echo "│ 🌐 HTML:  " . str_pad($htmlPath, 62) . " │\n";
echo "└──────────────────────────────────────────────────────────────────────────┘\n";

