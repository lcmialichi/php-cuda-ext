<?php

namespace Benchmarks\Exporters;

use Benchmarks\Contracts\ExporterInterface;
use Benchmarks\Support\BenchmarkReport;

class JsonExporter implements ExporterInterface
{
    public function export(BenchmarkReport $report, string $dir): string
    {
        $json = json_encode($report, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE);

        if (!is_dir($dir)) {
            mkdir($dir, 0755, true);
        }

        $path = $dir . DIRECTORY_SEPARATOR . "cuda-benchmark-" . time() . ".json";
        file_put_contents($path, $json);

        return $path;
    }
}
