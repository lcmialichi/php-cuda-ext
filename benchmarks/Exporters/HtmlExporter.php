<?php

namespace Benchmarks\Exporters;

use Benchmarks\Contracts\ExporterInterface;
use Benchmarks\Support\BenchmarkReport;
use Benchmarks\Support\BenchmarkClassResult;

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
        $comparisonHtml = $this->generateComparisonTable($allResults, $benchmarksData);
        $chartsHtml = $this->generateChartsSection($benchmarksData);

        $benchmarksDataJson = json_encode($benchmarksData, JSON_HEX_TAG | JSON_HEX_APOS | JSON_HEX_QUOT | JSON_HEX_AMP);

        $html = str_replace(
            [
                '{{TITLE}}',
                '{{DEVICE}}',
                '{{GENERATED_AT}}',
                '{{TIMESTAMP}}',
                '{{SUMMARY_SECTION}}',
                '{{COMPARISON_SECTION}}',
                '{{CHARTS_SECTION}}',
                '{{STYLE}}',
                '{{SCRIPTS}}',
                '{{BENCHMARKS_DATA_JSON}}'
            ],
            [
                'CUDA Benchmark Report - ' . date('Y-m-d H:i:s'),
                $report->getDevice(),
                date('Y-m-d H:i:s'),
                time(),
                $summaryHtml,
                $comparisonHtml,
                $chartsHtml,
                $this->cssTemplate,
                $this->jsTemplate,
                $benchmarksDataJson
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
            $avgTime = $result->getAvgTime();
            $opsPerSecond = $this->calculateOpsPerSecond($avgTime);
            $times = $result->getTimes();
            $memoryUsages = $result->getMemoryUsages();

            $results[] = [
                'name' => $result->getName(),
                'type' => $result->getType(),
                'iterations' => $result->getIterations(),
                'runs' => $result->getIterations(),
                'times' => $times,
                'memory_usage' => $memoryUsages,
                'metadata' => $result->getMetadata(),
                'stats' => [
                    'time' => [
                        'min' => $result->getMinTime(),
                        'max' => $result->getMaxTime(),
                        'avg' => $avgTime,
                        'total' => array_sum($times),
                        'std_dev' => $this->calculateStdDev($times)
                    ],
                    'memory' => [
                        'min' => $result->getMinMemoryUsage(),
                        'max' => $result->getMaxMemoryUsage(),
                        'avg' => $result->getAvgMemoryUsage(),
                        'total' => array_sum($memoryUsages)
                    ],
                    'ops' => [
                        'per_second' => $opsPerSecond,
                        'formatted' => $this->formatOpsPerSecond($opsPerSecond)
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
            return '<div class="alert alert-info">No benchmark data available</div>';
        }

        $html = '<div class="card glass-card">';
        $html .= '<div class="card-header">';
        $html .= '<h5 class="card-title mb-0"><i class="fas fa-chart-bar me-2"></i>Performance Summary</h5>';
        $html .= '</div>';
        $html .= '<div class="card-body">';

        $html .= '<div class="table-responsive">';
        $html .= '<table class="table table-hover table-sm">';
        $html .= '<thead>';
        $html .= '<tr>';
        $html .= '<th>Benchmark</th>';
        $html .= '<th>Tests</th>';
        $html .= '<th>Avg Time</th>';
        $html .= '<th>Avg Memory</th>';
        $html .= '<th>Ops/Sec</th>';
        $html .= '<th>Status</th>';
        $html .= '</tr>';
        $html .= '</thead>';
        $html .= '<tbody>';

        foreach ($benchmarksData as $benchmark) {
            $totalTime = 0;
            $totalMemoryBytes = 0;
            $totalOps = 0;
            $testCount = count($benchmark['results']);

            if ($testCount === 0) {
                continue;
            }

            foreach ($benchmark['results'] as $result) {
                $totalTime += $result['stats']['time']['avg'];
                $totalMemoryBytes += $result['stats']['memory']['avg'];
                $totalOps += $result['stats']['ops']['per_second'];
            }

            $avgTime = $totalTime / $testCount;
            $avgMemoryBytes = $totalMemoryBytes / $testCount;
            $avgOps = $totalOps / $testCount;

            $formattedAvgTime = $this->formatTime($avgTime);
            $formattedOps = $this->formatOpsPerSecond($avgOps);
            $formattedMemory = $this->formatBytes($avgMemoryBytes);

            $html .= '<tr class="benchmark-row" data-benchmark="' . htmlspecialchars($benchmark['handler_name']) . '">';
            $html .= '<td><strong>' . htmlspecialchars($benchmark['handler_name']) . '</strong></td>';
            $html .= '<td>' . $testCount . '</td>';
            $html .= '<td><span class="badge bg-time">' . $formattedAvgTime . '</span></td>';
            $html .= '<td><span class="badge bg-memory">' . $formattedMemory . '</span></td>';
            $html .= '<td><span class="badge bg-ops">' . $formattedOps . '</span></td>';
            $html .= '<td><span class="status-badge status-success"><i class="fas fa-check"></i></span></td>';
            $html .= '</tr>';
        }

        $html .= '</tbody>';
        $html .= '</table>';
        $html .= '</div>';

        $html .= $this->generateSummaryStats($benchmarksData);

        $html .= '</div>';
        $html .= '</div>';

        return $html;
    }

    private function generateSummaryStats(array $benchmarksData): string
    {
        if (empty($benchmarksData)) {
            return '';
        }

        $totalTests = 0;
        $totalTime = 0;
        $totalMemory = 0;
        $totalOps = 0;

        foreach ($benchmarksData as $benchmark) {
            $totalTests += count($benchmark['results']);
            foreach ($benchmark['results'] as $result) {
                $totalTime += $result['stats']['time']['avg'];
                $totalMemory += $result['stats']['memory']['avg'];
                $totalOps += $result['stats']['ops']['per_second'];
            }
        }

        $avgTime = $totalTests > 0 ? $totalTime / $totalTests : 0;
        $avgMemory = $totalTests > 0 ? $totalMemory / $totalTests : 0;
        $avgOps = $totalTests > 0 ? $totalOps / $totalTests : 0;

        $html = '<div class="row mt-4 g-3">';

        $html .= '<div class="col-md-3">';
        $html .= '<div class="stat-card">';
        $html .= '<div class="stat-icon">';
        $html .= '<i class="fas fa-vial"></i>';
        $html .= '</div>';
        $html .= '<div class="stat-content">';
        $html .= '<div class="stat-value">' . $totalTests . '</div>';
        $html .= '<div class="stat-label">Total Tests</div>';
        $html .= '</div>';
        $html .= '</div>';
        $html .= '</div>';

        $html .= '<div class="col-md-3">';
        $html .= '<div class="stat-card">';
        $html .= '<div class="stat-icon">';
        $html .= '<i class="fas fa-clock"></i>';
        $html .= '</div>';
        $html .= '<div class="stat-content">';
        $html .= '<div class="stat-value">' . $this->formatTime($avgTime) . '</div>';
        $html .= '<div class="stat-label">Avg Time</div>';
        $html .= '</div>';
        $html .= '</div>';
        $html .= '</div>';

        $html .= '<div class="col-md-3">';
        $html .= '<div class="stat-card">';
        $html .= '<div class="stat-icon">';
        $html .= '<i class="fas fa-memory"></i>';
        $html .= '</div>';
        $html .= '<div class="stat-content">';
        $html .= '<div class="stat-value">' . $this->formatBytes($avgMemory) . '</div>';
        $html .= '<div class="stat-label">Avg Memory</div>';
        $html .= '</div>';
        $html .= '</div>';
        $html .= '</div>';

        $html .= '<div class="col-md-3">';
        $html .= '<div class="stat-card">';
        $html .= '<div class="stat-icon">';
        $html .= '<i class="fas fa-tachometer-alt"></i>';
        $html .= '</div>';
        $html .= '<div class="stat-content">';
        $html .= '<div class="stat-value">' . $this->formatOpsPerSecond($avgOps) . '</div>';
        $html .= '<div class="stat-label">Avg Ops/Sec</div>';
        $html .= '</div>';
        $html .= '</div>';
        $html .= '</div>';

        $html .= '</div>';

        return $html;
    }

    private function generateComparisonTable(array $allResults, array $benchmarksData): string
    {
        if (empty($allResults)) {
            return '<div class="alert alert-info">No comparison data available</div>';
        }

        $groupedByHandler = [];
        foreach ($allResults as $result) {
            $handler = $this->findHandlerForResult($result, $benchmarksData);
            $testName = $result['name'];

            if (!isset($groupedByHandler[$handler])) {
                $groupedByHandler[$handler] = [];
            }

            if (!isset($groupedByHandler[$handler][$testName])) {
                $groupedByHandler[$handler][$testName] = [];
            }

            $groupedByHandler[$handler][$testName][] = $result;
        }

        $html = '<div class="card glass-card">';
        $html .= '<div class="card-header">';
        $html .= '<h5 class="card-title mb-0"><i class="fas fa-balance-scale me-2"></i>Test Comparison</h5>';
        $html .= '</div>';
        $html .= '<div class="card-body">';

        $html .= '<div class="row mb-4">';
        $html .= '<div class="col-md-6">';
        $html .= '<div class="input-group">';
        $html .= '<span class="input-group-text"><i class="fas fa-search"></i></span>';
        $html .= '<input type="text" class="form-control search-input" placeholder="Search tests...">';
        $html .= '</div>';
        $html .= '</div>';
        $html .= '<div class="col-md-6">';
        $html .= '<select class="form-select benchmark-filter">';
        $html .= '<option value="">All Benchmarks</option>';
        foreach ($benchmarksData as $benchmark) {
            $html .= '<option value="' . htmlspecialchars($benchmark['handler_name']) . '">' .
                htmlspecialchars($benchmark['handler_name']) . '</option>';
        }
        $html .= '</select>';
        $html .= '</div>';
        $html .= '</div>';

        $html .= '<div class="row" id="comparison-grid">';

        foreach ($groupedByHandler as $handlerName => $tests) {
            foreach ($tests as $testName => $testResults) {
                $bestTime = PHP_FLOAT_MAX;
                $worstTime = 0;
                $bestOps = 0;
                $totalMemory = 0;
                $resultCount = count($testResults);

                foreach ($testResults as $result) {
                    $time = $result['stats']['time']['avg'];
                    $ops = $result['stats']['ops']['per_second'];
                    $memory = $result['stats']['memory']['avg'];

                    $bestTime = min($bestTime, $time);
                    $worstTime = max($worstTime, $time);
                    $bestOps = max($bestOps, $ops);
                    $totalMemory += $memory;
                }

                $avgMemory = $resultCount > 0 ? $totalMemory / $resultCount : 0;

                $html .= '<div class="col-md-6 col-lg-4 mb-4" 
                    data-benchmark="' . htmlspecialchars($handlerName) . '"
                    data-test="' . htmlspecialchars($testName) . '">';

                $html .= '<div class="test-card">';
                $html .= '<div class="test-card-header">';
                $html .= '<h6>' . htmlspecialchars($testName) . '</h6>';
                $html .= '<small class="text-muted">' . htmlspecialchars($handlerName) . '</small>';
                $html .= '</div>';

                $html .= '<div class="test-card-body">';
                $html .= '<div class="metrics-grid">';

                $html .= '<div class="metric">';
                $html .= '<div class="metric-label"><i class="bi bi-clock"></i> Best Time</div>';
                $html .= '<div class="metric-value">' . $this->formatTime($bestTime) . '</div>';
                $html .= '</div>';

                $html .= '<div class="metric">';
                $html .= '<div class="metric-label"><i class="bi bi-speedometer2"></i> Ops/Sec</div>';
                $html .= '<div class="metric-value">' . $this->formatOpsPerSecond($bestOps) . '</div>';
                $html .= '</div>';

                $html .= '<div class="metric">';
                $html .= '<div class="metric-label"><i class="bi bi-memory"></i> Memory</div>';
                $html .= '<div class="metric-value">' . $this->formatBytes($avgMemory) . '</div>';
                $html .= '</div>';

                $html .= '<div class="metric">';
                $html .= '<div class="metric-label"><i class="bi bi-layers"></i> Configs</div>';
                $html .= '<div class="metric-value">' . $resultCount . '</div>';
                $html .= '</div>';
                $html .= '</div>';

                if ($bestTime > 0 && $worstTime > 0 && $worstTime > $bestTime) {
                    $timeRange = (($worstTime - $bestTime) / $bestTime) * 100;
                    $html .= '<div class="performance-bar mt-3">';
                    $html .= '<div class="bar-labels">';
                    $html .= '<span>' . $this->formatTime($bestTime) . '</span>';
                    $html .= '<span>' . round($timeRange, 1) . '% range</span>';
                    $html .= '<span>' . $this->formatTime($worstTime) . '</span>';
                    $html .= '</div>';
                    $html .= '<div class="bar-track">';
                    $html .= '<div class="bar-fill" style="width: ' . min(100, $timeRange) . '%"></div>';
                    $html .= '</div>';
                    $html .= '</div>';
                }

                $html .= '</div>';

                $html .= '<div class="test-card-footer">';
                $html .= '<button class="btn view-details-btn" 
                    data-handler="' . htmlspecialchars($handlerName) . '"
                    data-test="' . htmlspecialchars($testName) . '">';
                $html .= '<i class="fas fa-chart-bar me-1"></i> View Results';
                $html .= '</button>';
                $html .= '</div>';

                $html .= '</div>';
                $html .= '</div>';
            }
        }

        $html .= '</div>';
        $html .= '</div>';
        $html .= '</div>';

        return $html;
    }

    private function generateChartsSection(array $benchmarksData): string
    {
        if (empty($benchmarksData)) {
            return '<div class="alert alert-info">No chart data available</div>';
        }

        $testGroups = [];
        foreach ($benchmarksData as $benchmark) {
            $handlerName = $benchmark['handler_name'];

            foreach ($benchmark['results'] as $result) {
                $testName = $result['name'];

                if (!isset($testGroups[$testName])) {
                    $testGroups[$testName] = [
                        'count' => 0,
                        'total_time' => 0,
                        'total_memory' => 0,
                        'total_ops' => 0,
                        'handlers' => [],
                        'test' => $testName
                    ];
                }

                $testGroups[$testName]['count']++;
                $testGroups[$testName]['total_time'] += $result['stats']['time']['avg'];
                $testGroups[$testName]['total_memory'] += $result['stats']['memory']['avg'];
                $testGroups[$testName]['total_ops'] += $result['stats']['ops']['per_second'];

                if (!in_array($handlerName, $testGroups[$testName]['handlers'])) {
                    $testGroups[$testName]['handlers'][] = $handlerName;
                }
            }
        }

        $chartLabels = [];
        $timeData = [];
        $memoryData = [];
        $opsData = [];

        foreach ($testGroups as $testName => $group) {
            $chartLabels[] = $testName;
            $timeData[] = $group['total_time'] / $group['count'];
            $memoryData[] = ($group['total_memory'] / $group['count']) / (1024 * 1024);
            $opsData[] = ($group['total_ops'] / $group['count']) / 1000;
        }

        $chartDataJson = json_encode([
            'labels' => $chartLabels,
            'time' => $timeData,
            'memory' => $memoryData,
            'ops' => $opsData,
            'testGroups' => $testGroups
        ], JSON_HEX_TAG | JSON_HEX_APOS | JSON_HEX_QUOT | JSON_HEX_AMP);

        $html = '<div class="card glass-card">';
        $html .= '<div class="card-header">';
        $html .= '<h5 class="card-title mb-0"><i class="bi bi-graph-up me-2"></i>Performance Charts</h5>';
        $html .= '</div>';
        $html .= '<div class="card-body">';

        $html .= '<div class="chart-controls mb-4">';
        $html .= '<div class="row g-3">';
        $html .= '<div class="col-md-4">';
        $html .= '<label class="form-label">Metric</label>';
        $html .= '<select class="form-select metric-select">';
        $html .= '<option value="time">Time (ms)</option>';
        $html .= '<option value="memory">Memory (MB)</option>';
        $html .= '<option value="ops">Operations/sec (K)</option>';
        $html .= '</select>';
        $html .= '</div>';
        $html .= '<div class="col-md-4">';
        $html .= '<label class="form-label">Sort By</label>';
        $html .= '<select class="form-select sort-select">';
        $html .= '<option value="name">Name</option>';
        $html .= '<option value="time">Time</option>';
        $html .= '<option value="memory">Memory</option>';
        $html .= '<option value="ops">Operations</option>';
        $html .= '</select>';
        $html .= '</div>';
        $html .= '<div class="col-md-4">';
        $html .= '<label class="form-label">Limit Results</label>';
        $html .= '<input type="number" class="form-control chart-limit" value="100" min="1" max="500">';
        $html .= '</div>';
        $html .= '</div>';
        $html .= '</div>';

        $html .= '<div class="row mb-4">';
        $html .= '<div class="col-md-12">';
        $html .= '<div class="chart-container horizontal-scroll">';
        $html .= '<canvas id="mainPerformanceChart" data-chart-data="' . htmlspecialchars($chartDataJson) . '"></canvas>';
        $html .= '</div>';
        $html .= '</div>';
        $html .= '</div>';

        $html .= '<div class="row">';
        $html .= '<div class="col-md-6">';
        $html .= '<h6 class="mb-3">Top 10 Slowest Tests</h6>';
        $html .= '<div class="chart-container" style="height: 300px;">';
        $html .= '<canvas id="slowestTestsChart"></canvas>';
        $html .= '</div>';
        $html .= '</div>';
        $html .= '<div class="col-md-6">';
        $html .= '<h6 class="mb-3">Top 10 Fastest Tests</h6>';
        $html .= '<div class="chart-container" style="height: 300px;">';
        $html .= '<canvas id="fastestTestsChart"></canvas>';
        $html .= '</div>';
        $html .= '</div>';
        $html .= '</div>';

        $html .= '</div>';
        $html .= '</div>';

        return $html;
    }

    private function findHandlerForResult(array $result, array $benchmarksData): string
    {
        foreach ($benchmarksData as $benchmark) {
            foreach ($benchmark['results'] as $benchmarkResult) {
                if ($benchmarkResult['name'] === $result['name']) {
                    return $benchmark['handler_name'];
                }
            }
        }
        return 'Unknown';
    }

    private function formatBytes(int $bytes): string
    {
        $units = ['B', 'KB', 'MB', 'GB'];
        $i = 0;

        while ($bytes >= 1024 && $i < count($units) - 1) {
            $bytes /= 1024;
            $i++;
        }

        return number_format($bytes, 2) . ' ' . $units[$i];
    }

    private function calculateStdDev(array $values): float
    {
        $n = count($values);
        if ($n < 2) {
            return 0.0;
        }

        $mean = array_sum($values) / $n;
        $sumSquares = 0.0;

        foreach ($values as $value) {
            $sumSquares += ($value - $mean) * ($value - $mean);
        }

        $variance = $sumSquares / ($n - 1);

        return sqrt($variance);
    }

    private function calculateOpsPerSecond(float $timeMs): float
    {
        if ($timeMs <= 0) {
            return 0;
        }

        $seconds = $timeMs / 1000;
        return 1 / $seconds;
    }

    private function formatTime(float $timeMs): string
    {
        if ($timeMs < 0.001) {
            return number_format($timeMs * 1_000_000, 2) . ' ns';
        } elseif ($timeMs < 1) {
            return number_format($timeMs * 1000, 2) . ' μs';
        } elseif ($timeMs < 1000) {
            return number_format($timeMs, 3) . ' ms';
        } else {
            return number_format($timeMs / 1000, 3) . ' s';
        }
    }

    private function formatOpsPerSecond(float $ops): string
    {
        if ($ops >= 1_000_000_000) {
            return number_format($ops / 1_000_000_000, 2) . ' B op/s';
        } elseif ($ops >= 1_000_000) {
            return number_format($ops / 1_000_000, 2) . ' M op/s';
        } elseif ($ops >= 1_000) {
            return number_format($ops / 1_000, 2) . ' K op/s';
        } elseif ($ops < 1) {
            return number_format(1 / $ops, 2) . ' s/op';
        } else {
            return number_format($ops, 2) . ' op/s';
        }
    }
}