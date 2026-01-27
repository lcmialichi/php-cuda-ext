<?php

namespace Benchmarks\Exporters;

use Benchmarks\Contracts\ExporterInterface;
use Benchmarks\Support\BenchmarkReport;
use Benchmarks\Support\BenchmarkClassResult;
use Benchmarks\Support\BenchmarkResult;

class HtmlExporter implements ExporterInterface
{
    private string $cssTemplate;
    private string $htmlTemplate;

    public function __construct()
    {
        $this->cssTemplate = $this->getDefaultCss();
        $this->htmlTemplate = $this->getDefaultHtmlTemplate();
    }

    public function export(BenchmarkReport $report, string $dir): string
    {
        $path = $dir . DIRECTORY_SEPARATOR . "cuda-benchmark-" . time() . ".html";

        if (!is_dir($dir)) {
            mkdir($dir, 0755, true);
        }

        $htmlContent = $this->generateHtml($report);
        file_put_contents($path, $htmlContent);

        return $path;
    }

    private function generateHtml(BenchmarkReport $report): string
    {
        $benchmarksHtml = '';

        foreach ($report->getResults() as $classResult) {
            $benchmarksHtml .= $this->generateBenchmarkSection($classResult);
        }

        $html = str_replace(
            ['{{TITLE}}', '{{GENERATED_AT}}', '{{BENCHMARKS}}', "{{STYLE}}"],
            [
                'Benchmark Report - ' . date('Y-m-d H:i:s'),
                date('Y-m-d H:i:s'),
                $benchmarksHtml,
                $this->cssTemplate,
            ],
            $this->htmlTemplate
        );

        return $html;
    }

    private function generateBenchmarkSection(BenchmarkClassResult $classResult): string
    {
        $handler = $classResult->getHandler();

        $resultsHtml = '';
        foreach ($classResult->getResults() as $result) {
            $resultsHtml .= $this->generateResultHtml($result);
        }

        return sprintf(
            '
            <div class="benchmark-class">
                <h2>%s</h2>
                <p class="benchmark-description">%s</p>
                <div class="benchmark-results">
                    %s
                </div>
            </div>
        ',
            htmlspecialchars($handler->name()),
            htmlspecialchars($handler->description()),
            $resultsHtml
        );
    }

    private function generateResultHtml(BenchmarkResult $result): string
    {
        $metadataHtml = '';
        if (!empty($result->getMetadata())) {
            $metadataHtml = '<div class="metadata"><strong>Metadata:</strong><br>';
            foreach ($result->getMetadata() as $key => $value) {
                $metadataHtml .= sprintf('%s: %s<br>', htmlspecialchars($key), htmlspecialchars($value));
            }
            $metadataHtml .= '</div>';
        }

        return sprintf(
            '
            <div class="test-result">
                <h3>%s <span class="type-badge">%s</span></h3>
                <p class="test-info">
                    Iterations per run: %d | 
                    Total runs: %d
                </p>
                %s
                <div class="stats">
                    <div class="stat-column">
                        <h4>Time Statistics (ms)</h4>
                        <ul>
                            <li>Min: <strong>%.4f</strong></li>
                            <li>Max: <strong>%.4f</strong></li>
                            <li>Avg: <strong>%.4f</strong></li>
                            <li>Total: <strong>%.4f</strong></li>
                        </ul>
                    </div>
                    <div class="stat-column">
                        <h4>Memory Statistics</h4>
                        <ul>
                            <li>Min: <strong>%s</strong></li>
                            <li>Max: <strong>%s</strong></li>
                            <li>Avg: <strong>%s</strong></li>
                            <li>Total: <strong>%s</strong></li>
                        </ul>
                    </div>
                </div>
            </div>
        ',
            htmlspecialchars($result->getName()),
            htmlspecialchars($result->getType()),
            $result->getIterations(),
            $result->getIterations(),
            $metadataHtml,
            $result->getMinTime(),
            $result->getMaxTime(),
            $result->getAvgTime(),
            array_sum($result->getTimes()),
            $this->formatBytes($result->getMinMemoryUsage()),
            $this->formatBytes($result->getMaxMemoryUsage()),
            $this->formatBytes($result->getAvgMemoryUsage()),
            $this->formatBytes(array_sum($result->getMemoryUsages()))
        );
    }

    private function formatBytes($bytes): string
    {
        $units = ['B', 'KB', 'MB', 'GB'];
        $bytes = max($bytes, 0);
        $pow = floor(($bytes ? log($bytes) : 0) / log(1024));
        $pow = min($pow, count($units) - 1);
        $bytes /= pow(1024, $pow);

        return round($bytes, 2) . ' ' . $units[$pow];
    }

    private function getDefaultCss(): string
    {
        return file_get_contents(__DIR__ . "/../Support/stubs/benchmark-report.stub.css");
    }

    private function getDefaultHtmlTemplate(): string
    {
        return file_get_contents(__DIR__ . "/../Support/stubs/benchmark-report.stub.html");
    }
}
