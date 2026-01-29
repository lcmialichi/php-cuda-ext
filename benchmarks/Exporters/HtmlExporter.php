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

        $html = str_replace(
            [
                '{{TITLE}}',
                '{{DEVICE}}',
                '{{GENERATED_AT}}',
                '{{TIMESTAMP}}',
                '{{SUMMARY_SECTION}}',
                '{{COMPARISON_SECTION}}',
                '{{STYLE}}',
                '{{SCRIPTS}}'
            ],
            [
                'CUDA Benchmark Report - ' . date('Y-m-d H:i:s'),
                $report->getDevice(),
                date('Y-m-d H:i:s'),
                time(),
                $summaryHtml,
                $comparisonHtml,
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
            $avgTime = $result->getAvgTime();
            $opsPerSecond = $this->calculateOpsPerSecond($avgTime);
            
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
                        'avg' => $avgTime,
                        'total' => array_sum($result->getTimes()),
                        'std_dev' => $this->calculateStdDev($result->getTimes())
                    ],
                    'memory' => [
                        'min' => $this->formatBytes($result->getMinMemoryUsage()),
                        'max' => $this->formatBytes($result->getMaxMemoryUsage()),
                        'avg' => $this->formatBytes($result->getAvgMemoryUsage()),
                        'total' => $this->formatBytes(array_sum($result->getMemoryUsages()))
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
        $html .= '<th>Avg Time</th>';
        $html .= '<th>Avg Memory</th>';
        $html .= '<th>OP/Second</th>';
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
                $totalMemoryBytes += $this->parseMemoryToBytes($result['stats']['memory']['avg']);
                $totalOps += $result['stats']['ops']['per_second'];
            }

            $avgTime = $totalTime / $testCount;
            $avgMemoryBytes = $totalMemoryBytes / $testCount;
            $avgOps = $totalOps / $testCount;

            $formattedAvgTime = $this->formatTime($avgTime);
            $formattedOps = $this->formatOpsPerSecond($avgOps);
            $formattedMemory = $this->formatBytes($avgMemoryBytes);

            $html .= sprintf(
                '<tr class="summary-row" data-benchmark="%s">',
                htmlspecialchars($benchmark['handler_name'])
            );
            
            $html .= '<td>' . htmlspecialchars($benchmark['handler_name']) . '</td>';
            $html .= '<td>' . $testCount . '</td>';
            $html .= '<td><span class="time-value" title="Average time">' . $formattedAvgTime . ' (avg)</span></td>';
            $html .= '<td><span class="memory-value" title="Average memory usage">' . $formattedMemory . '</span></td>';
            $html .= '<td><span class="ratio-badge" title="Operations per second">' . $formattedOps . '</span></td>';
            $html .= '<td><span class="status-badge status-excellent">✓</span></td>';
            $html .= '</tr>';
        }

        $html .= '</tbody>';
        $html .= '</table>';
        $html .= '</div>';
        
        $html .= $this->generateSummaryStats($benchmarksData);
        
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
                $totalMemory += $this->parseMemoryToBytes($result['stats']['memory']['avg']);
                $totalOps += $result['stats']['ops']['per_second'];
            }
        }
        
        $avgTime = $totalTime / $totalTests;
        $avgMemory = $totalMemory / $totalTests;
        $avgOps = $totalOps / $totalTests;
        
        $html = '<div class="summary-stats">';
        $html .= '<div class="stats-grid">';
        
        $html .= '<div class="stat-card">';
        $html .= '<div class="stat-icon"><i class="fas fa-vial"></i></div>';
        $html .= '<div class="stat-content">';
        $html .= '<div class="stat-value">' . $totalTests . '</div>';
        $html .= '<div class="stat-label">Total Tests</div>';
        $html .= '</div>';
        $html .= '</div>';
        
        $html .= '<div class="stat-card">';
        $html .= '<div class="stat-icon"><i class="fas fa-clock"></i></div>';
        $html .= '<div class="stat-content">';
        $html .= '<div class="stat-value">' . $this->formatTime($avgTime) . '</div>';
        $html .= '<div class="stat-label">Avg Time</div>';
        $html .= '</div>';
        $html .= '</div>';
        
        $html .= '<div class="stat-card">';
        $html .= '<div class="stat-icon"><i class="fas fa-memory"></i></div>';
        $html .= '<div class="stat-content">';
        $html .= '<div class="stat-value">' . $this->formatBytes($avgMemory) . '</div>';
        $html .= '<div class="stat-label">Avg Memory</div>';
        $html .= '</div>';
        $html .= '</div>';
        
        $html .= '<div class="stat-card">';
        $html .= '<div class="stat-icon"><i class="fas fa-tachometer-alt"></i></div>';
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
            return '';
        }

        $groupedByHandler = [];
        foreach ($allResults as $result) {
            $handler = $this->findHandlerForResult($result, $benchmarksData);
            $testName = $result['name'];
            $metadataHash = $this->getMetadataHash($result['metadata']);

            if (!isset($groupedByHandler[$handler])) {
                $groupedByHandler[$handler] = [];
            }

            if (!isset($groupedByHandler[$handler][$testName])) {
                $groupedByHandler[$handler][$testName] = [];
            }

            if (!isset($groupedByHandler[$handler][$testName][$metadataHash])) {
                $groupedByHandler[$handler][$testName][$metadataHash] = [
                    'metadata' => $result['metadata'],
                    'type' => $result['type'],
                    'runs' => []
                ];
            }

            $groupedByHandler[$handler][$testName][$metadataHash]['runs'][] = $result;
        }

        $html = '<div class="comparison-section">';
        $html .= '<h2><i class="fas fa-balance-scale"></i> Test Comparison</h2>';
        
        $html .= '<div class="comparison-stats-header">';
        $html .= '<div class="stats-overview">';
        $html .= '<span class="stat-item"><i class="fas fa-layer-group"></i> ' . count($benchmarksData) . ' Benchmarks</span>';
        $html .= '<span class="stat-item"><i class="fas fa-list"></i> ' . count($allResults) . ' Tests</span>';
        $html .= '<span class="stat-item"><i class="fas fa-play"></i> ' . array_sum(array_map('count', $groupedByHandler)) . ' Configurations</span>';
        $html .= '</div>';
        $html .= '</div>';
        
        $html .= '<div class="controls">';
        $html .= '<div class="filter-controls">';
        $html .= '<input type="text" class="search-input" placeholder="Search tests...">';
        $html .= '<select class="benchmark-filter">';
        $html .= '<option value="">All Benchmarks</option>';
        foreach ($benchmarksData as $benchmark) {
            $html .= '<option value="' . htmlspecialchars($benchmark['handler_name']) . '">' . 
                     htmlspecialchars($benchmark['handler_name']) . '</option>';
        }
        $html .= '</select>';
        $html .= '</div>';
        
        $html .= '<div class="sort-controls">';
        $html .= '<select class="sort-select">';
        $html .= '<option value="name-asc">Sort by Name (A-Z)</option>';
        $html .= '<option value="name-desc">Sort by Name (Z-A)</option>';
        $html .= '<option value="time-asc">Sort by Time (Fastest)</option>';
        $html .= '<option value="time-desc">Sort by Time (Slowest)</option>';
        $html .= '<option value="memory-asc">Sort by Memory (Lowest)</option>';
        $html .= '<option value="memory-desc">Sort by Memory (Highest)</option>';
        $html .= '<option value="ops-desc">Sort by Ops/Sec (Highest)</option>';
        $html .= '<option value="ops-asc">Sort by Ops/Sec (Lowest)</option>';
        $html .= '</select>';
        $html .= '</div>';
        $html .= '</div>';

        $html .= '<div class="comparison-container">';

        foreach ($groupedByHandler as $handlerName => $tests) {
            $html .= '<div class="benchmark-group" data-benchmark="' . htmlspecialchars($handlerName) . '">';
            $html .= '<div class="benchmark-group-header">';
            $html .= '<h3><i class="fas fa-microchip"></i> ' . htmlspecialchars($handlerName) . '</h3>';
            $html .= '<span class="test-count">' . count($tests) . ' tests</span>';
            $html .= '</div>';
            
            $html .= '<div class="tests-grid">';
            
            foreach ($tests as $testName => $metadataGroups) {
                $totalRuns = 0;
                $bestTime = PHP_FLOAT_MAX;
                $worstTime = 0;
                $bestOps = 0;
                $totalMemory = 0;
                $configCount = count($metadataGroups);
                
                foreach ($metadataGroups as $group) {
                    $totalRuns += count($group['runs']);
                    foreach ($group['runs'] as $run) {
                        $time = $run['stats']['time']['avg'];
                        $ops = $run['stats']['ops']['per_second'];
                        $memory = $this->parseMemoryToBytes($run['stats']['memory']['avg']);
                        
                        $bestTime = min($bestTime, $time);
                        $worstTime = max($worstTime, $time);
                        $bestOps = max($bestOps, $ops);
                        $totalMemory += $memory;
                    }
                }
                
                $avgMemory = $totalRuns > 0 ? $totalMemory / $totalRuns : 0;
                $timeRange = $bestTime > 0 ? (($worstTime - $bestTime) / $bestTime * 100) : 0;
                
                $html .= '<div class="test-card" 
                    data-name="' . htmlspecialchars($testName) . '"
                    data-time="' . $bestTime . '"
                    data-memory="' . $avgMemory . '"
                    data-ops="' . $bestOps . '"
                    data-runs="' . $totalRuns . '">';
                
                $html .= '<div class="test-card-header">';
                $html .= '<h4>' . htmlspecialchars($testName) . '</h4>';
                $html .= '<span class="config-count">' . $configCount . ' configs</span>';
                $html .= '</div>';
                
                $html .= '<div class="test-card-body">';
                
                $html .= '<div class="performance-metrics">';
                $html .= '<div class="metric">';
                $html .= '<div class="metric-label"><i class="fas fa-clock"></i> Best Time</div>';
                $html .= '<div class="metric-value time-metric">' . $this->formatTime($bestTime) . '</div>';
                $html .= '</div>';
                
                $html .= '<div class="metric">';
                $html .= '<div class="metric-label"><i class="fas fa-tachometer-alt"></i> Ops/Sec</div>';
                $html .= '<div class="metric-value ops-metric">' . $this->formatOpsPerSecond($bestOps) . '</div>';
                $html .= '</div>';
                
                $html .= '<div class="metric">';
                $html .= '<div class="metric-label"><i class="fas fa-memory"></i> Memory</div>';
                $html .= '<div class="metric-value memory-metric">' . $this->formatBytes($avgMemory) . '</div>';
                $html .= '</div>';
                $html .= '</div>';
                
                if ($timeRange > 0) {
                    $html .= '<div class="performance-bar">';
                    $html .= '<div class="bar-labels">';
                    $html .= '<span>' . $this->formatTime($bestTime) . '</span>';
                    $html .= '<span>' . round($timeRange, 1) . '% range</span>';
                    $html .= '<span>' . $this->formatTime($worstTime) . '</span>';
                    $html .= '</div>';
                    $html .= '<div class="bar-container">';
                    $html .= '<div class="bar-fill" style="width: ' . min(100, $timeRange) . '%"></div>';
                    $html .= '</div>';
                    $html .= '</div>';
                }
                
                $html .= '<div class="config-summary">';
                foreach ($metadataGroups as $metadataHash => $group) {
                    if (empty($group['metadata'])) {
                        $html .= '<span class="config-tag">Default</span>';
                    } else {
                        $configText = implode(', ', array_map(
                            fn($k, $v) => $k . ': ' . (is_array($v) ? json_encode($v) : $v),
                            array_keys($group['metadata']),
                            array_values($group['metadata'])
                        ));
                        $html .= '<span class="config-tag" title="' . htmlspecialchars($configText) . '">' . 
                                 htmlspecialchars(substr($configText, 0, 30)) . 
                                 (strlen($configText) > 30 ? '...' : '') . '</span>';
                    }
                }
                $html .= '</div>';
                
                $html .= '</div>';
                
                $html .= '<div class="test-card-footer">';
                $html .= '<button class="view-details-btn" data-test="' . htmlspecialchars($testName) . '">';
                $html .= '<i class="fas fa-chart-bar"></i> View Details';
                $html .= '</button>';
                $html .= '</div>';
                
                $html .= '</div>';
            }
            
            $html .= '</div>';
            $html .= '</div>';
        }

        $html .= '</div>';
        
        $html .= $this->generateDetailsModal();
        
        $html .= '</div>';

        return $html;
    }

    private function findHandlerForResult(array $result, array $benchmarksData): string
    {
        foreach ($benchmarksData as $benchmark) {
            foreach ($benchmark['results'] as $benchmarkResult) {
                if ($benchmarkResult['name'] === $result['name'] && 
                    $benchmarkResult['type'] === $result['type'] &&
                    $this->getMetadataHash($benchmarkResult['metadata']) === $this->getMetadataHash($result['metadata'])) {
                    return $benchmark['handler_name'];
                }
            }
        }
        return 'Unknown';
    }

    private function generateDetailsModal(): string
    {
        return '
        <div class="details-modal" id="detailsModal">
            <div class="modal-overlay"></div>
            <div class="modal-content">
                <div class="modal-header">
                    <h3><i class="fas fa-chart-bar"></i> Test Details</h3>
                    <button class="close-modal">&times;</button>
                </div>
                <div class="modal-body" id="modalBody">
                    <!-- Details will be loaded here -->
                </div>
            </div>
        </div>';
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

    private function getMetadataHash(array $metadata): string
    {
        ksort($metadata);
        return md5(json_encode($metadata));
    }

    private function formatMetadataForComparison(array $metadata): string
    {
        if (empty($metadata)) {
            return '<div class="metadata-empty">No configuration metadata</div>';
        }

        $items = [];
        foreach ($metadata as $key => $value) {
            if (is_array($value)) {
                $value = json_encode($value);
            }
            $items[] = '<div class="metadata-item">' .
                '<span class="metadata-key">' . htmlspecialchars($key) . ':</span>' .
                '<span class="metadata-value">' . htmlspecialchars((string) $value) . '</span>' .
                '</div>';
        }

        return '<div class="metadata-grid">' . implode('', $items) . '</div>';
    }

    private function parseMemoryToBytes(string $memoryString): int
    {
        $units = ['B' => 1, 'KB' => 1024, 'MB' => 1024 ** 2, 'GB' => 1024 ** 3];
        $memoryString = trim($memoryString);

        foreach ($units as $unit => $multiplier) {
            if (str_ends_with($memoryString, $unit)) {
                $value = (float) str_replace($unit, '', $memoryString);
                return (int) ($value * $multiplier);
            }
        }

        return (int) $memoryString;
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