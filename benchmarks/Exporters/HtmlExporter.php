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
                '{{DEVICE}}',
                '{{GENERATED_AT}}',
                '{{SUMMARY_SECTION}}',
                '{{COMPARISON_SECTION}}',
                '{{DETAILS_SECTION}}',
                '{{STYLE}}',
                '{{SCRIPTS}}'
            ],
            [
                'CUDA Benchmark Report - ' . date('Y-m-d H:i:s'),
                $report->getDevice(),
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

        $groupedResults = [];
        foreach ($allResults as $result) {
            $testName = $result['name'];
            $metadataHash = $this->getMetadataHash($result['metadata']);

            if (!isset($groupedResults[$testName])) {
                $groupedResults[$testName] = [];
            }

            if (!isset($groupedResults[$testName][$metadataHash])) {
                $groupedResults[$testName][$metadataHash] = [
                    'metadata' => $result['metadata'],
                    'results' => [],
                    'type' => $result['type']
                ];
            }

            $groupedResults[$testName][$metadataHash]['results'][] = $result;
        }

        $html = '<div class="comparison-section">';
        $html .= '<h2><i class="fas fa-balance-scale"></i> Test Comparison (Grouped by Test Name)</h2>';
        $html .= '<div class="controls">';
        $html .= '<div class="sort-controls">';
        $html .= '<select class="sort-select">';
        $html .= '<option value="name">Sort by Name</option>';
        $html .= '<option value="time">Sort by Avg Time</option>';
        $html .= '<option value="memory">Sort by Avg Memory</option>';
        $html .= '<option value="runs">Sort by Number of Runs</option>';
        $html .= '</select>';
        $html .= '</div>';
        $html .= '</div>';

        $html .= '<div class="comparison-groups">';

        foreach ($groupedResults as $testName => $metadataGroups) {
            $totalRuns = 0;
            foreach ($metadataGroups as $group) {
                $totalRuns += count($group['results']);
            }

            $avgTime = 0;
            $avgMemory = 0;
            $runCount = 0;

            foreach ($metadataGroups as $group) {
                foreach ($group['results'] as $result) {
                    $avgTime += $result['stats']['time']['avg'];
                    $avgMemory += $this->parseBytes($result['stats']['memory']['avg']);
                    $runCount++;
                }
            }

            $avgTime = $runCount > 0 ? $avgTime / $runCount : 0;
            $avgMemory = $runCount > 0 ? $this->formatBytes($avgMemory / $runCount) : '0 B';

            $html .= '<div class="test-group" data-name="' . htmlspecialchars($testName) . '" 
                 data-time="' . $avgTime . '" data-memory="' . $this->parseBytes($avgMemory) . '" 
                 data-runs="' . $totalRuns . '">';

            $html .= '<div class="group-header">';
            $html .= '<div class="group-title">';
            $html .= '<h3>' . htmlspecialchars($testName) . '</h3>';
            $html .= '<div class="group-stats">';
            $html .= '<span class="stat-badge"><i class="fas fa-play-circle"></i> ' . $totalRuns . ' runs</span>';
            $html .= '<span class="stat-badge"><i class="fas fa-clock"></i> ' . number_format($avgTime, 4) . ' ms avg</span>';
            $html .= '<span class="stat-badge"><i class="fas fa-memory"></i> ' . $avgMemory . ' avg</span>';
            $html .= '</div>';
            $html .= '</div>';
            $html .= '<button class="toggle-group-btn"><i class="fas fa-chevron-down"></i></button>';
            $html .= '</div>';

            $html .= '<div class="group-body">';
            $currentRun = 0;

            foreach ($metadataGroups as $metadataHash => $group) {
                $metadataHtml = $this->formatMetadataForComparison($group['metadata']);
                $groupResultCount = count($group['results']);
                $groupAvgTime = 0;
                $groupAvgMemory = 0;
                $currentRun++;

                foreach ($group['results'] as $result) {
                    $groupAvgTime += $result['stats']['time']['avg'];
                    $groupAvgMemory += $this->parseBytes($result['stats']['memory']['avg']);
                }

                $groupAvgTime = $groupResultCount > 0 ? $groupAvgTime / $groupResultCount : 0;
                $groupAvgMemory = $groupResultCount > 0 ? $this->formatBytes($groupAvgMemory / $groupResultCount) : '0 B';

                $html .= '<div class="metadata-group">';
                $html .= '<div class="metadata-group-header">';
                $html .= '<h4>Configuration</h4>';
                $html .= '<div class="metadata-content">' . $metadataHtml . '</div>';
                $html .= '<div class="metadata-stats">';
                $html .= '<span><i class="fas fa-running"></i> ' . $groupResultCount . ' runs</span>';
                $html .= '<span><i class="fas fa-clock"></i> ' . number_format($groupAvgTime, 4) . ' ms avg</span>';
                $html .= '</div>';
                $html .= '</div>';

                $html .= '<div class="runs-container">';
                foreach ($group['results'] as $index => $result) {
                    $timePercent = $this->calculateTimePercentage($result['stats']['time']['avg'], $allResults);
                    $memoryPercent = $this->calculateMemoryPercentage(
                        $this->parseBytes($result['stats']['memory']['avg']),
                        $allResults
                    );

                    $html .= '<div class="run-card">';
                    $html .= '<div class="run-header">';
                    $html .= '<span class="run-number">Run ' . $currentRun . '</span>';
                    $html .= '<span class="run-type">' . htmlspecialchars($result['type']) . '</span>';
                    $html .= '</div>';

                    $html .= '<div class="run-body">';
                    $html .= '<div class="metric-row">';
                    $html .= '<div class="metric-cell">';
                    $html .= '<div class="metric-label">Time</div>';
                    $html .= '<div class="metric-value">' . number_format($result['stats']['time']['avg'], 4) . ' ms</div>';
                    $html .= '<div class="progress-bar">';
                    $html .= '<div class="progress-fill" style="width: ' . $timePercent . '%"></div>';
                    $html .= '</div>';
                    $html .= '</div>';

                    $html .= '<div class="metric-cell">';
                    $html .= '<div class="metric-label">Memory</div>';
                    $html .= '<div class="metric-value">' . $result['stats']['memory']['avg'] . '</div>';
                    $html .= '<div class="progress-bar">';
                    $html .= '<div class="progress-fill" style="width: ' . $memoryPercent . '%"></div>';
                    $html .= '</div>';
                    $html .= '</div>';
                    $html .= '</div>';

                    $html .= '<div class="run-details-toggle">';
                    $html .= '<button class="toggle-run-details-btn" data-run="' .
                        htmlspecialchars($testName . '-' . $metadataHash . '-' . $index) . '">';
                    $html .= '<i class="fas fa-chart-bar"></i> View Details';
                    $html .= '</button>';
                    $html .= '</div>';

                    $html .= '<div class="run-details" id="' .
                        htmlspecialchars($testName . '-' . $metadataHash . '-' . $index) . '">';
                    $html .= $this->generateRunDetails($result, $index + 1);
                    $html .= '</div>';
                    $html .= '</div>';
                    $html .= '</div>';
                }

                $html .= '</div>';
                $html .= '</div>';
            }

            $html .= '</div>';
            $html .= '</div>';
        }

        $html .= '</div>';
        $html .= '</div>';

        return $html;
    }

    private function getMetadataHash(array $metadata): string
    {
        ksort($metadata);
        return md5(json_encode($metadata));
    }

    private function formatMetadataForComparison(array $metadata): string
    {
        if (empty($metadata)) {
            return '<span class="no-metadata">No metadata</span>';
        }

        $items = [];
        foreach ($metadata as $key => $value) {
            if (is_array($value)) {
                $value = json_encode($value);
            }
            $items[] = '<div class="metadata-item"><strong>' .
                htmlspecialchars($key) . ':</strong> ' .
                htmlspecialchars((string)$value) . '</div>';
        }

        return '<div class="metadata-items">' . implode('', $items) . '</div>';
    }

    private function generateRunDetails(array $result, int $runNumber): string
    {
        $html = '<div class="run-details-content">';
        $html .= '<div class="details-grid">';

        $html .= '<div class="detail-block time-stats">';
        $html .= '<h5><i class="fas fa-clock"></i> Time Statistics (ms)</h5>';
        $html .= '<div class="stat-row">';
        $html .= '<span class="stat-label">Min:</span>';
        $html .= '<span class="stat-value">' . number_format($result['stats']['time']['min'], 4) . '</span>';
        $html .= '</div>';
        $html .= '<div class="stat-row">';
        $html .= '<span class="stat-label">Max:</span>';
        $html .= '<span class="stat-value">' . number_format($result['stats']['time']['max'], 4) . '</span>';
        $html .= '</div>';
        $html .= '<div class="stat-row">';
        $html .= '<span class="stat-label">Avg:</span>';
        $html .= '<span class="stat-value">' . number_format($result['stats']['time']['avg'], 4) . '</span>';
        $html .= '</div>';
        $html .= '<div class="stat-row">';
        $html .= '<span class="stat-label">Total:</span>';
        $html .= '<span class="stat-value">' . number_format($result['stats']['time']['total'], 4) . '</span>';
        $html .= '</div>';
        $html .= '<div class="stat-row">';
        $html .= '<span class="stat-label">Std Dev:</span>';
        $html .= '<span class="stat-value">' . number_format($result['stats']['time']['std_dev'], 4) . '</span>';
        $html .= '</div>';
        $html .= '</div>';

        $html .= '<div class="detail-block memory-stats">';
        $html .= '<h5><i class="fas fa-memory"></i> Memory Statistics</h5>';
        $html .= '<div class="stat-row">';
        $html .= '<span class="stat-label">Min:</span>';
        $html .= '<span class="stat-value">' . $result['stats']['memory']['min'] . '</span>';
        $html .= '</div>';
        $html .= '<div class="stat-row">';
        $html .= '<span class="stat-label">Max:</span>';
        $html .= '<span class="stat-value">' . $result['stats']['memory']['max'] . '</span>';
        $html .= '</div>';
        $html .= '<div class="stat-row">';
        $html .= '<span class="stat-label">Avg:</span>';
        $html .= '<span class="stat-value">' . $result['stats']['memory']['avg'] . '</span>';
        $html .= '</div>';
        $html .= '<div class="stat-row">';
        $html .= '<span class="stat-label">Total:</span>';
        $html .= '<span class="stat-value">' . $result['stats']['memory']['total'] . '</span>';
        $html .= '</div>';
        $html .= '</div>';

        $html .= '<div class="detail-block run-info">';
        $html .= '<h5><i class="fas fa-info-circle"></i> Run Information</h5>';
        $html .= '<div class="stat-row">';
        $html .= '<span class="stat-label">Run #:</span>';
        $html .= '<span class="stat-value">' . $runNumber . '</span>';
        $html .= '</div>';
        $html .= '<div class="stat-row">';
        $html .= '<span class="stat-label">Iterations:</span>';
        $html .= '<span class="stat-value">' . $result['iterations'] . '</span>';
        $html .= '</div>';
        $html .= '<div class="stat-row">';
        $html .= '<span class="stat-label">Type:</span>';
        $html .= '<span class="stat-value">' . htmlspecialchars($result['type']) . '</span>';
        $html .= '</div>';
        $html .= '</div>';

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
