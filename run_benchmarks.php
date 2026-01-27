<?php

declare(strict_types=1);

require_once __DIR__ . "/vendor/autoload.php";

use Benchmarks\Handlers\CudaArrayBenchmark;
use Benchmarks\BenchmarkApplication;

if (!extension_loaded('cuda')) {
    die(" CUDA extension not loaded. Please compile and install the extension first.\n");
}

$app = new BenchmarkApplication([
    new CudaArrayBenchmark()
]);

$dir = __DIR__ . "/benchmarks/reports";

$report = $app->run();
$jsonPath = $report->saveJSON($dir);
$htmlPath = $report->saveHTML($dir);

echo "JSON: {$jsonPath}\n";
echo "HTML: {$htmlPath}\n";

