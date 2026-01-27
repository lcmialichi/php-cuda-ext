<?php

namespace Benchmarks\Exporters;

use Benchmarks\Contracts\ExporterInterface;
use Benchmarks\Support\BenchmarkReport;
use Benchmarks\Support\BenchmarkClassResult;
use Benchmarks\Support\BenchmarkResult;

class HtmlExporter implements ExporterInterface
{
    private string $templateDir;
    private string $cssTemplate;
    private string $htmlTemplate;
    private string $jsTemplate;

    public function __construct(?string $templateDir = null)
    {
        $this->templateDir = $templateDir ?? __DIR__ . "/../Support/stubs";
        $this->loadTemplates();
    }

    private function loadTemplates(): void
    {
        $this->cssTemplate = file_get_contents($this->templateDir . "/benchmark-report.stub.css");
        $this->htmlTemplate = file_get_contents($this->templateDir . "/benchmark-report.stub.html");
        $this->jsTemplate = file_get_contents($this->templateDir . "/benchmark-report.stub.js");
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
        $benchmarksData = [];
        $allResults = [];

        foreach ($report->getResults() as $classResult) {
            $benchmarkData = $this->prepareBenchmarkData($classResult);
            $benchmarksData[] = $benchmarkData;
            $allResults = array_merge($allResults, $benchmarkData['results']);
        }

        $summaryHtml = $this->generateSummaryTable($benchmarksData);
        $comparisonHtml = $this->generateComparisonTable($allResults);
        $detailsHtml = $this->generateDetailsSections($benchmarksData);

        $html = str_replace(
            [
                '{{TITLE}}',
                '{{GENERATED_AT}}',
                '{{SUMMARY_SECTION}}',
                '{{COMPARISON_SECTION}}',
                '{{DETAILS_SECTION}}',
                '{{STYLE}}',
                '{{SCRIPTS}}'
            ],
            [
                'CUDA Benchmark Report - ' . date('Y-m-d H:i:s'),
                date('Y-m-d H:i:s'),
                $summaryHtml,
                $comparisonHtml,
                $detailsHtml,
                $this->cssTemplate,
                $this->jsTemplate
            ],
            $this->htmlTemplate
        );

        return $html;
    }

    private function prepareBenchmarkData(BenchmarkClassResult $classResult): array
    {
        $handler = $classResult->getHandler();
        $results = [];

        foreach ($classResult->getResults() as $result) {
            $results[] = [
                'name' => $result->getName(),
                'type' => $result->getType(),
                'iterations' => $result->getIterations(),
                'runs' => $result->getIterations(),
                'times' => $result->getTimes(),
                'memory_usage' => $result->getMemoryUsages(),
                'metadata' => $result->getMetadata(),
                'stats' => [
                    'time' => [
                        'min' => $result->getMinTime(),
                        'max' => $result->getMaxTime(),
                        'avg' => $result->getAvgTime(),
                        'total' => array_sum($result->getTimes()),
                        'std_dev' => $this->calculateStdDev($result->getTimes())
                    ],
                    'memory' => [
                        'min' => $this->formatBytes($result->getMinMemoryUsage()),
                        'max' => $this->formatBytes($result->getMaxMemoryUsage()),
                        'avg' => $this->formatBytes($result->getAvgMemoryUsage()),
                        'total' => $this->formatBytes(array_sum($result->getMemoryUsages()))
                    ]
                ]
            ];
        }

        return [
            'handler_name' => $handler->name(),
            'handler_description' => $handler->description(),
            'results' => $results
        ];
    }

    private function generateSummaryTable(array $benchmarksData): string
    {
        if (empty($benchmarksData)) {
            return '<div class="no-data">No benchmark data available</div>';
        }

        $html = '<div class="summary-table-container">';
        $html .= '<h2><i class="fas fa-chart-bar"></i> Performance Summary</h2>';
        $html .= '<div class="table-responsive">';
        $html .= '<table class="summary-table">';
        $html .= '<thead>';
        $html .= '<tr>';
        $html .= '<th>Benchmark</th>';
        $html .= '<th>Tests</th>';
        $html .= '<th>Avg Time (ms)</th>';
        $html .= '<th>Avg Memory</th>';
        $html .= '<th>Speed Ratio</th>';
        $html .= '<th>Status</th>';
        $html .= '</tr>';
        $html .= '</thead>';
        $html .= '<tbody>';

        foreach ($benchmarksData as $benchmark) {
            $avgTime = 0;
            $avgMemory = 0;
            $testCount = count($benchmark['results']);

            foreach ($benchmark['results'] as $result) {
                $avgTime += $result['stats']['time']['avg'];
                $avgMemory += $this->parseBytes($result['stats']['memory']['avg']);
            }

            $avgTime = $testCount > 0 ? $avgTime / $testCount : 0;
            $avgMemory = $testCount > 0 ? $this->formatBytes($avgMemory / $testCount) : '0 B';

            $speedRatio = $avgTime > 0 ? number_format(1 / $avgTime, 2) : 0;

            $html .= sprintf(
                '<tr class="summary-row" data-benchmark="%s">',
                htmlspecialchars($benchmark['handler_name'])
            );
            $html .= '<td>' . htmlspecialchars($benchmark['handler_name']) . '</td>';
            $html .= '<td>' . $testCount . '</td>';
            $html .= '<td><span class="time-value">' . number_format($avgTime, 4) . '</span></td>';
            $html .= '<td><span class="memory-value">' . $avgMemory . '</span></td>';
            $html .= '<td><span class="ratio-badge">' . $speedRatio . 'x</span></td>';
            $html .= '<td><span class="status-badge status-ok">✓</span></td>';
            $html .= '</tr>';
        }

        $html .= '</tbody>';
        $html .= '</table>';
        $html .= '</div>';
        $html .= '</div>';

        return $html;
    }

    private function generateComparisonTable(array $allResults): string
    {
        if (empty($allResults)) {
            return '';
        }

        $html = '<div class="comparison-section">';
        $html .= '<h2><i class="fas fa-balance-scale"></i> Test Comparison</h2>';
        $html .= '<div class="controls">';
        $html .= '<div class="filter-controls">';
        $html .= '<label><input type="checkbox" class="filter-checkbox" data-type="time" checked> Show Time</label>';
        $html .= '<label><input type="checkbox" class="filter-checkbox" data-type="memory" checked> Show Memory</label>';
        $html .= '</div>';
        $html .= '<div class="sort-controls">';
        $html .= '<button class="sort-btn" data-sort="time">Sort by Time</button>';
        $html .= '<button class="sort-btn" data-sort="memory">Sort by Memory</button>';
        $html .= '<button class="sort-btn" data-sort="name">Sort by Name</button>';
        $html .= '</div>';
        $html .= '</div>';

        $html .= '<div class="comparison-grid">';

        foreach ($allResults as $result) {
            $timePercent = $this->calculateTimePercentage($result['stats']['time']['avg'], $allResults);
            $memoryPercent = $this->calculateMemoryPercentage(
                $this->parseBytes($result['stats']['memory']['avg']),
                $allResults
            );

            $html .= '<div class="comparison-card" data-time="' . $result['stats']['time']['avg'] . '" 
                     data-memory="' . $this->parseBytes($result['stats']['memory']['avg']) . '">';
            $html .= '<div class="card-header">';
            $html .= '<h3>' . htmlspecialchars($result['name']) . '</h3>';
            $html .= '<span class="type-tag">' . htmlspecialchars($result['type']) . '</span>';
            $html .= '</div>';
            
            $html .= '<div class="card-body">';
            $html .= '<div class="metric time-metric">';
            $html .= '<div class="metric-label">Time</div>';
            $html .= '<div class="metric-value">' . number_format($result['stats']['time']['avg'], 4) . ' ms</div>';
            $html .= '<div class="progress-bar">';
            $html .= '<div class="progress-fill" style="width: ' . $timePercent . '%"></div>';
            $html .= '</div>';
            $html .= '</div>';
            
            $html .= '<div class="metric memory-metric">';
            $html .= '<div class="metric-label">Memory</div>';
            $html .= '<div class="metric-value">' . $result['stats']['memory']['avg'] . '</div>';
            $html .= '<div class="progress-bar">';
            $html .= '<div class="progress-fill" style="width: ' . $memoryPercent . '%"></div>';
            $html .= '</div>';
            $html .= '</div>';
            
            $html .= '<div class="metric-details" style="display: none;">';
            $html .= '<div class="detail-item">';
            $html .= '<span>Min:</span>';
            $html .= '<span>' . number_format($result['stats']['time']['min'], 4) . ' ms</span>';
            $html .= '</div>';
            $html .= '<div class="detail-item">';
            $html .= '<span>Max:</span>';
            $html .= '<span>' . number_format($result['stats']['time']['max'], 4) . ' ms</span>';
            $html .= '</div>';
            $html .= '</div>';
            $html .= '</div>';
            
            $html .= '<div class="card-footer">';
            $html .= '<button class="toggle-details-btn">Show Details</button>';
            $html .= '</div>';
            $html .= '</div>';
        }

        $html .= '</div>';
        $html .= '</div>';

        return $html;
    }

    private function generateDetailsSections(array $benchmarksData): string
    {
        $html = '<div class="details-section">';
        $html .= '<h2><i class="fas fa-list-alt"></i> Detailed Results</h2>';

        foreach ($benchmarksData as $benchmark) {
            $html .= '<div class="benchmark-details">';
            $html .= '<h3 class="details-header">';
            $html .= '<i class="fas fa-caret-right"></i> ';
            $html .= htmlspecialchars($benchmark['handler_name']);
            $html .= '</h3>';
            $html .= '<p class="benchmark-description">' . htmlspecialchars($benchmark['handler_description']) . '</p>';
            
            $html .= '<div class="details-content">';
            foreach ($benchmark['results'] as $result) {
                $html .= $this->generateResultDetails($result);
            }
            $html .= '</div>';
            $html .= '</div>';
        }

        $html .= '</div>';
        return $html;
    }

    private function generateResultDetails(array $result): string
    {
        $metadataHtml = '';
        if (!empty($result['metadata'])) {
            $metadataHtml = '<div class="metadata-grid">';
            foreach ($result['metadata'] as $key => $value) {
                $metadataHtml .= '<div class="metadata-item">';
                $metadataHtml .= '<strong>' . htmlspecialchars($key) . ':</strong>';
                $metadataHtml .= '<span>' . htmlspecialchars($value) . '</span>';
                $metadataHtml .= '</div>';
            }
            $metadataHtml .= '</div>';
        }

        return sprintf(
            '<div class="result-details">
                <div class="result-header">
                    <h4>%s</h4>
                    <div class="result-meta">
                        <span class="meta-item">Type: %s</span>
                        <span class="meta-item">Iterations: %d</span>
                        <span class="meta-item">Runs: %d</span>
                    </div>
                </div>
                %s
                <div class="stats-grid">
                    <div class="stat-box time-stats">
                        <h5><i class="fas fa-clock"></i> Time Statistics (ms)</h5>
                        <div class="stat-values">
                            <div class="stat-value">
                                <span class="stat-label">Min:</span>
                                <span class="stat-number">%.4f</span>
                            </div>
                            <div class="stat-value">
                                <span class="stat-label">Max:</span>
                                <span class="stat-number">%.4f</span>
                            </div>
                            <div class="stat-value">
                                <span class="stat-label">Avg:</span>
                                <span class="stat-number">%.4f</span>
                            </div>
                            <div class="stat-value">
                                <span class="stat-label">Total:</span>
                                <span class="stat-number">%.4f</span>
                            </div>
                            <div class="stat-value">
                                <span class="stat-label">Std Dev:</span>
                                <span class="stat-number">%.4f</span>
                            </div>
                        </div>
                    </div>
                    <div class="stat-box memory-stats">
                        <h5><i class="fas fa-memory"></i> Memory Statistics</h5>
                        <div class="stat-values">
                            <div class="stat-value">
                                <span class="stat-label">Min:</span>
                                <span class="stat-number">%s</span>
                            </div>
                            <div class="stat-value">
                                <span class="stat-label">Max:</span>
                                <span class="stat-number">%s</span>
                            </div>
                            <div class="stat-value">
                                <span class="stat-label">Avg:</span>
                                <span class="stat-number">%s</span>
                            </div>
                            <div class="stat-value">
                                <span class="stat-label">Total:</span>
                                <span class="stat-number">%s</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>',
            htmlspecialchars($result['name']),
            htmlspecialchars($result['type']),
            $result['iterations'],
            $result['runs'],
            $metadataHtml,
            $result['stats']['time']['min'],
            $result['stats']['time']['max'],
            $result['stats']['time']['avg'],
            $result['stats']['time']['total'],
            $result['stats']['time']['std_dev'],
            $result['stats']['memory']['min'],
            $result['stats']['memory']['max'],
            $result['stats']['memory']['avg'],
            $result['stats']['memory']['total']
        );
    }

    private function calculateStdDev(array $values): float
    {
        if (count($values) < 2) {
            return 0;
        }

        $mean = array_sum($values) / count($values);
        $sum = 0;

        foreach ($values as $value) {
            $sum += pow($value - $mean, 2);
        }

        return sqrt($sum / count($values));
    }

    private function calculateTimePercentage(float $value, array $allResults): float
    {
        $max = 0;
        foreach ($allResults as $result) {
            $max = max($max, $result['stats']['time']['avg']);
        }

        return $max > 0 ? min(100, ($value / $max) * 100) : 0;
    }

    private function calculateMemoryPercentage(float $value, array $allResults): float
    {
        $max = 0;
        foreach ($allResults as $result) {
            $max = max($max, $this->parseBytes($result['stats']['memory']['avg']));
        }

        return $max > 0 ? min(100, ($value / $max) * 100) : 0;
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

    private function parseBytes(string $formatted): float
    {
        $units = ['B' => 1, 'KB' => 1024, 'MB' => 1048576, 'GB' => 1073741824];
        $parts = explode(' ', $formatted);
        
        if (count($parts) !== 2 || !isset($units[$parts[1]])) {
            return 0;
        }

        return floatval($parts[0]) * $units[$parts[1]];
    }
}
