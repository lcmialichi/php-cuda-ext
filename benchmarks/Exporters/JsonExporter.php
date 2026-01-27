<?php

namespace Benchmarks\Exporters;

use Benchmarks\Contracts\ExporterInterface;
use Benchmarks\Support\BenchmarkReport;

class JsonExporter implements ExporterInterface
{
    public function export(BenchmarkReport $report, string $dir): string
    {
        $json = json_encode($report, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE);

        $path = $dir . DIRECTORY_SEPARATOR . "cuda-benchmark-" . time() . ".json";
        if (!is_dir($dir)) {
            mkdir(dirname($dir), 0755, true);
        }

        file_put_contents($path, $json);

        return $path;
    }
}
