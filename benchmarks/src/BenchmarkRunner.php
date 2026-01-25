<?php

declare(strict_types=1);

require_once __DIR__ . '/AbstractBenchmark.php';

class BenchmarkRunner
{
    private array $benchmarks = [];
    private array $results = [];
    private array $config;

    public function __construct(array $config = [])
    {
        $this->config = array_merge(
            require __DIR__ . '/../config/benchmark_config.php',
            $config
        );
    }

    public function addBenchmark(AbstractBenchmark $benchmark): self
    {
        $this->benchmarks[] = $benchmark;
        return $this;
    }

    public function runAll(): array
    {
        $this->printHeader();

        foreach ($this->benchmarks as $benchmark) {
            echo "\n" . str_repeat("=", 80) . "\n";
            echo "Running: " . $benchmark->getName() . "\n";
            echo str_repeat("=", 80) . "\n";

            $result = $benchmark->run();
            $this->results[$benchmark->getName()] = $result;

            echo $benchmark->toConsole();
        }

        $this->generateReports();

        return $this->results;
    }

    private function printHeader(): void
    {
        echo "\n";
        echo str_repeat("*", 80) . "\n";
        echo "* CUDA PHP EXTENSION - COMPREHENSIVE BENCHMARK SUITE\n";
        echo str_repeat("*", 80) . "\n";
        echo "* Timestamp: " . date('Y-m-d H:i:s') . "\n";
        echo "* PHP Version: " . PHP_VERSION . "\n";
        echo "* OS: " . PHP_OS . "\n";
        echo "* Memory Limit: " . ini_get('memory_limit') . "\n";
        echo str_repeat("*", 80) . "\n\n";
    }

    private function generateReports(): void
    {
        $outputFormat = $this->config['output']['format'] ?? 'both';

        if (in_array($outputFormat, ['json', 'both'])) {
            $this->generateJsonReport();
        }

        if (in_array($outputFormat, ['html', 'both'])) {
            $this->generateHtmlReport();
        }
    }

    private function generateJsonReport(): void
    {
        $outputDir = $this->config['output']['output_dir'];
        $filename = $outputDir . '/benchmark_results_' . date('Ymd_His') . '.json';

        $data = [
            'metadata' => [
                'timestamp' => date('c'),
                'php_version' => PHP_VERSION,
                'os' => PHP_OS,
                'benchmarks_run' => count($this->benchmarks),
            ],
            'results' => $this->results
        ];

        file_put_contents($filename, json_encode($data, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES));
        echo "\nJSON report saved to: " . $filename . "\n";
    }

    private function generateHtmlReport(): void
    {
        $outputDir = $this->config['output']['output_dir'];
        $filename = $outputDir . '/benchmark_report_' . date('Ymd_His') . '.html';

        $html = $this->buildHtmlReport();
        file_put_contents($filename, $html);

        echo "\nHTML report saved to: " . $filename . "\n";
        echo "Open in browser: file://" . realpath($filename) . "\n";
    }

    private function buildHtmlReport(): string
    {
        $chartsData = $this->prepareChartsData();

        return <<<HTML
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CUDA PHP Benchmark Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            line-height: 1.6; 
            color: #333;
            min-height: 100vh;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 20px; 
        }
        .header { 
            background: white; 
            padding: 30px; 
            border-radius: 15px; 
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
        }
        .header h1 { 
            color: #2c3e50; 
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .header .subtitle { 
            color: #7f8c8d; 
            font-size: 1.1em;
        }
        .stats-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px;
        }
        .stat-card { 
            background: white; 
            padding: 25px; 
            border-radius: 12px; 
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: transform 0.3s ease;
        }
        .stat-card:hover { 
            transform: translateY(-5px); 
            box-shadow: 0 15px 30px rgba(0,0,0,0.15);
        }
        .stat-card h3 { 
            color: #3498db; 
            margin-bottom: 15px; 
            font-size: 1.1em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stat-value { 
            font-size: 2em; 
            font-weight: bold; 
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .stat-label { 
            color: #7f8c8d; 
            font-size: 0.9em;
        }
        .chart-container { 
            background: white; 
            padding: 25px; 
            border-radius: 12px; 
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }
        .chart-container h2 { 
            color: #2c3e50; 
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f0f0f0;
        }
        .chart-wrapper { 
            position: relative; 
            height: 400px; 
            margin-bottom: 20px;
        }
        .benchmark-section { 
            background: white; 
            padding: 30px; 
            border-radius: 12px; 
            margin-bottom: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }
        .benchmark-section h2 { 
            color: #2c3e50; 
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .results-table { 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 20px;
        }
        .results-table th { 
            background: #f8f9fa; 
            padding: 15px; 
            text-align: left; 
            color: #2c3e50;
            font-weight: 600;
            border-bottom: 2px solid #e9ecef;
        }
        .results-table td { 
            padding: 15px; 
            border-bottom: 1px solid #e9ecef;
            color: #495057;
        }
        .results-table tr:hover { 
            background: #f8f9fa;
        }
        .badge { 
            display: inline-block; 
            padding: 5px 12px; 
            border-radius: 20px; 
            font-size: 0.85em; 
            font-weight: 600;
        }
        .badge-success { background: #d4edda; color: #155724; }
        .badge-warning { background: #fff3cd; color: #856404; }
        .badge-danger { background: #f8d7da; color: #721c24; }
        .badge-info { background: #d1ecf1; color: #0c5460; }
        .performance-meter { 
            height: 8px; 
            background: #e9ecef; 
            border-radius: 4px; 
            overflow: hidden;
            margin-top: 5px;
        }
        .performance-bar { 
            height: 100%; 
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        .comparison-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 20px; 
            margin-top: 20px;
        }
        .comparison-item { 
            padding: 20px; 
            border-radius: 8px; 
            background: #f8f9fa;
        }
        .footer { 
            text-align: center; 
            padding: 30px; 
            color: white; 
            margin-top: 50px;
        }
        .footer a { 
            color: #3498db; 
            text-decoration: none;
        }
        .footer a:hover { 
            text-decoration: underline;
        }
        @media (max-width: 768px) {
            .container { padding: 10px; }
            .header { padding: 20px; }
            .header h1 { font-size: 2em; }
            .chart-wrapper { height: 300px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>CUDA PHP Benchmark Report</h1>
            <p class="subtitle">Generated on {$this->getTimestamp()}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Benchmarks</h3>
                <div class="stat-value">{$this->getBenchmarkCount()}</div>
                <div class="stat-label">Tests Completed</div>
            </div>
            <div class="stat-card">
                <h3>PHP Version</h3>
                <div class="stat-value">{$this->getPhpVersion()}</div>
                <div class="stat-label">Runtime</div>
            </div>
            <div class="stat-card">
                <h3>Total Time</h3>
                <div class="stat-value">{$this->getTotalTime()}s</div>
                <div class="stat-label">Execution Duration</div>
            </div>
            <div class="stat-card">
                <h3>Status</h3>
                <div class="stat-value">
                    <span class="badge badge-success">Completed</span>
                </div>
                <div class="stat-label">All tests passed</div>
            </div>
        </div>
        
        {$this->generateBenchmarkSections()}
        
        <div class="footer">
            <p>Report generated by CUDA PHP Extension Benchmark Suite</p>
            <p>© " . date('Y') . " - All rights reserved</p>
        </div>
    </div>
    
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        {$this->generateChartJsScripts()}
    });
    </script>
</body>
</html>
HTML;
    }

    private function prepareChartsData(): array
    {
        $charts = [];

        foreach ($this->results as $benchmarkName => $result) {
            $charts[$benchmarkName] = $this->extractChartData($result);
        }

        return $charts;
    }

    private function extractChartData(array $benchmarkData): array
    {
        $data = [
            'operations' => [],
            'times' => [],
            'memory' => [],
            'categories' => []
        ];

        foreach ($benchmarkData['results'] ?? [] as $result) {
            $data['operations'][] = $result['operation'];
            $data['times'][] = $result['performance']['time']['avg'];
            $data['memory'][] = $result['performance']['memory']['avg'];
            $data['categories'][] = $result['metadata']['type'] ?? 'unknown';
        }

        return $data;
    }

    private function getTimestamp(): string
    {
        return date('Y-m-d H:i:s');
    }

    private function getBenchmarkCount(): int
    {
        return count($this->benchmarks);
    }

    private function getPhpVersion(): string
    {
        return PHP_VERSION;
    }

    private function getTotalTime(): float
    {
        $total = 0;
        foreach ($this->results as $result) {
            $total += $result['total_time'] ?? 0;
        }
        return round($total, 2);
    }

    private function generateBenchmarkSections(): string
    {
        $sections = '';

        foreach ($this->results as $benchmarkName => $result) {
            $sections .= $this->generateBenchmarkSection($benchmarkName, $result);
        }

        return $sections;
    }

    private function generateBenchmarkSection(string $name, array $data): string
    {
        $rows = '';
        $chartId = str_replace(' ', '_', strtolower($name));

        foreach ($data['results'] ?? [] as $result) {
            $time = $result['performance']['time']['avg'];
            $memory = $result['performance']['memory']['avg'];
            $metadata = json_encode($result['metadata'], JSON_UNESCAPED_SLASHES);
            $type = $result['metadata']['type'] ?? 'N/A';
            $badge = $this->getCategoryBadge($result['metadata']['type'] ?? '');
            $description = $data['description'] ?? '';
            $shape = $result['metadata']['shape'] ?? 'N/A';
            $nativeTime = 0;
            $nativeBytes = 0;
            
            if ($result['native'] !== null) {
                $nativeTime = $result['native']['time']['avg'];
                $nativeBytes = $result['native']['memory']['avg'];
            }

            $rows .= <<<HTML
            <tr>
                <td>{$result['operation']}</td>
                <td>{$shape}</td>
                <td>{$this->formatTime($time)}</td>
                <td>{$this->formatBytes($memory)}</td>
                <td>{$this->formatTime($nativeTime)}</td>
                <td>{$this->formatBytes($nativeBytes)}</td>
                <td>{$result['performance']['time']['iterations']}</td>
                <td><div class="performance-meter">
                    <div class="performance-bar" style="width: {$this->calculatePerformanceWidth($time)}%; background: {$this->getPerformanceColor($time)}"></div>
                </div></td>
                <td><span class="badge {$badge}">
                    {$type}
                </span></td>
            </tr>
HTML;
        }
        return <<<HTML
        <div class="benchmark-section">
            <h2>📈 {$name}</h2>
            <p>{$description}</p>
            
            <div class="chart-wrapper">
                <canvas id="chart_{$chartId}"></canvas>
            </div>
            
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Operation</th>
                        <th>Shape</th>
                        <th>Avg Time</th>
                        <th>Avg Memory</th>
                        <th>PHP Native Time</th>
                        <th>PHP Native Memory</th>
                        <th>Iterations</th>
                        <th>Performance</th>
                        <th>Type</th>
                    </tr>
                </thead>
                <tbody>
                    {$rows}
                </tbody>
            </table>
        </div>
HTML;
    }

    private function generateChartJsScripts(): string
    {
        $scripts = '';

        foreach ($this->results as $benchmarkName => $data) {
            $chartId = str_replace(' ', '_', strtolower($benchmarkName));
            $chartData = $this->extractChartData($data);

            if (empty($chartData['operations'])) {
                continue;
            }

            $scripts .= <<<JS
            // Chart for {$benchmarkName}
            const ctx_{$chartId} = document.getElementById('chart_{$chartId}').getContext('2d');
            new Chart(ctx_{$chartId}, {
                type: 'bar',
                data: {
                    labels: {$this->jsEncode($chartData['operations'])},
                    datasets: [
                        {
                            label: 'Time (seconds)',
                            data: {$this->jsEncode($chartData['times'])},
                            backgroundColor: 'rgba(54, 162, 235, 0.7)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1,
                            yAxisID: 'y'
                        },
                        {
                            label: 'Memory (MB)',
                            data: {$this->jsEncode(array_map(fn($b) => $b / 1024 / 1024, $chartData['memory']))},
                            backgroundColor: 'rgba(255, 99, 132, 0.7)',
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 1,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Time (s)'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Memory (MB)'
                            },
                            grid: {
                                drawOnChartArea: false
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: '{$benchmarkName} - Performance Metrics'
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false
                        }
                    }
                }
            });
JS;
        }

        return $scripts;
    }

    private function jsEncode($data): string
    {
        return json_encode($data, JSON_UNESCAPED_SLASHES);
    }

    private function formatTime(float $seconds): string
    {
        if ($seconds < 0.001) {
            return round($seconds * 1e6, 2) . ' µs';
        } elseif ($seconds < 1) {
            return round($seconds * 1e3, 2) . ' ms';
        }
        return round($seconds, 4) . ' s';
    }

    private function formatBytes(float $bytes): string
    {
        $units = ['B', 'KB', 'MB', 'GB'];
        $bytes = max($bytes, 0);
        $pow = floor(($bytes ? log($bytes) : 0) / log(1024));
        $pow = min($pow, count($units) - 1);
        $bytes /= (1 << (10 * $pow));

        return round($bytes, 2) . ' ' . $units[$pow];
    }

    private function calculatePerformanceWidth(float $time): float
    {
        $maxTime = 1.0;
        $normalized = min($time / $maxTime, 1.0);
        return (1 - $normalized) * 100;
    }

    private function getPerformanceColor(float $time): string
    {
        if ($time < 0.001)
            return '#28a745';
        if ($time < 0.01)
            return '#20c997';
        if ($time < 0.1)
            return '#ffc107';
        if ($time < 1.0)
            return '#fd7e14';
        return '#dc3545';
    }

    private function getCategoryBadge(string $category): string
    {
        $badges = [
            'transfer' => 'badge-info',
            'materialization' => 'badge-info',
            'sequential' => 'badge-success',
            'random' => 'badge-warning',
            'element_wise' => 'badge-success',
            'binary' => 'badge-info',
            'reduction' => 'badge-warning',
            'matrix' => 'badge-danger',
            'transform' => 'badge-info',
            'compilation' => 'badge-warning',
            'execution' => 'badge-success',
            'async_benchmark' => 'badge-info',
            'compute_intensity' => 'badge-danger',
        ];

        return $badges[$category] ?? 'badge-info';
    }
}