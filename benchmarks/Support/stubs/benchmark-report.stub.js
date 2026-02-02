class BenchmarkReport {
    constructor() {
        this.benchmarkData = window.benchmarkData || [];
        this.charts = {};
        this.currentChartData = null;
        this.init();
    }

    init() {
        this.initTabs();
        this.initSearchFilter();

        if (document.querySelector('#nav-charts-tab').classList.contains('active')) {
            setTimeout(() => this.initCharts(), 100);
        }

        this.initClickHandlers();
        this.initModal();
        this.initChartControls();
        this.initChartClickHandlers();
    }

    initTabs() {
        const tabTriggers = document.querySelectorAll('[data-bs-toggle="tab"]');
        tabTriggers.forEach(trigger => {
            trigger.addEventListener('shown.bs.tab', (event) => {
                const target = event.target.getAttribute('data-bs-target');
                if (target === '#nav-charts') {
                    setTimeout(() => this.initCharts(), 100);
                }
            });
        });
    }

    initSearchFilter() {
        const searchInput = document.querySelector('.search-input');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                const term = e.target.value.toLowerCase();
                const cards = document.querySelectorAll('#comparison-grid > div');

                cards.forEach(card => {
                    const testName = card.querySelector('.test-card-header h6').textContent.toLowerCase();
                    const benchmark = card.getAttribute('data-benchmark').toLowerCase();

                    if (testName.includes(term) || benchmark.includes(term)) {
                        card.style.display = 'block';
                    } else {
                        card.style.display = 'none';
                    }
                });
            });
        }

        const filterSelect = document.querySelector('.benchmark-filter');
        if (filterSelect) {
            filterSelect.addEventListener('change', (e) => {
                const selected = e.target.value;
                const cards = document.querySelectorAll('#comparison-grid > div');

                cards.forEach(card => {
                    const benchmark = card.getAttribute('data-benchmark');

                    if (!selected || benchmark === selected) {
                        card.style.display = 'block';
                    } else {
                        card.style.display = 'none';
                    }
                });
            });
        }
    }

    initClickHandlers() {
        document.querySelectorAll('.benchmark-row').forEach(row => {
            row.addEventListener('click', (e) => {
                const benchmark = row.getAttribute('data-benchmark');
                const filterSelect = document.querySelector('.benchmark-filter');

                if (filterSelect) {
                    filterSelect.value = benchmark;
                    filterSelect.dispatchEvent(new Event('change'));

                    const comparisonTab = document.querySelector('#nav-comparison-tab');
                    if (comparisonTab) {
                        bootstrap.Tab.getOrCreateInstance(comparisonTab).show();
                    }
                }
            });
        });

        document.addEventListener('click', (e) => {
            if (e.target.closest('.view-details-btn')) {
                const button = e.target.closest('.view-details-btn');
                const handler = button.getAttribute('data-handler');
                const test = button.getAttribute('data-test');

                this.showTestDetails(handler, test);
            }
        });
    }

    initChartClickHandlers() {
        document.addEventListener('click', (e) => {
            if (e.target.closest('#mainPerformanceChart')) {
                const canvas = e.target.closest('#mainPerformanceChart');
                const chart = this.charts.main;

                if (!chart) return;

                const activePoints = chart.getElementsAtEventForMode(
                    e.nativeEvent,
                    'nearest',
                    { intersect: true },
                    true
                );

                if (activePoints.length > 0) {
                    const firstPoint = activePoints[0];
                    const label = chart.data.labels[firstPoint.index];
                    const testName = label;

                    const handlers = this.findHandlersForTest(testName);
                    if (handlers.length > 0) {
                        this.showTestDetails(handlers[0], testName);
                    }
                }
            }
        });
    }

    findHandlersForTest(testName) {
        const handlers = [];
        this.benchmarkData.forEach(benchmark => {
            const hasTest = benchmark.results.some(result => result.name === testName);
            if (hasTest) {
                handlers.push(benchmark.handler_name);
            }
        });
        return handlers;
    }

    initModal() {
        const modal = document.getElementById('detailsModal');
        if (modal) {
            modal.addEventListener('hidden.bs.modal', () => {
                document.getElementById('modalBody').innerHTML = `
                    <div class="text-center py-5">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="mt-3 text-muted">Loading test details...</p>
                    </div>
                `;

                if (this.charts.detail) {
                    this.charts.detail.destroy();
                    delete this.charts.detail;
                }
            });
        }
    }

    showTestDetails(handlerName, testName) {
        const testData = this.findTestData(handlerName, testName);

        if (!testData || testData.length === 0) {
            this.showError('No data found for this test');
            return;
        }

        const modalBody = document.getElementById('modalBody');
        if (!modalBody) return;

        const detailsHtml = this.generateTestDetailsHTML(testData, handlerName, testName);
        modalBody.innerHTML = detailsHtml;

        const modal = new bootstrap.Modal(document.getElementById('detailsModal'));
        modal.show();

        setTimeout(() => this.initTestDetailChart(testData, handlerName, testName), 100);
    }

    findTestData(handlerName, testName) {
        if (!this.benchmarkData || this.benchmarkData.length === 0) {
            console.warn('No benchmark data available');
            return [];
        }

        const benchmark = this.benchmarkData.find(b => b.handler_name === handlerName);
        if (!benchmark) {
            console.warn(`Benchmark ${handlerName} not found`);
            return [];
        }

        const results = benchmark.results.filter(r => r.name === testName);
        console.log(`Found ${results.length} results for ${handlerName} - ${testName}`, results);
        return results;
    }

    generateTestDetailsHTML(testData, handlerName, testName) {
        const sortedData = [...testData].sort((a, b) => a.stats.time.avg - b.stats.time.avg);

        let html = `
            <div class="test-details">
                <div class="row mb-4">
                    <div class="col-md-8">
                        <h4 class="fw-bold mb-1">${this.escapeHtml(testName)}</h4>
                        <p class="text-muted mb-0">
                            <i class="bi bi-cpu me-1"></i> ${this.escapeHtml(handlerName)}
                            <span class="mx-2">•</span>
                            <i class="bi bi-layer-group me-1"></i> ${sortedData.length} configurations
                        </p>
                    </div>
                    <div class="col-md-4 text-end">
                        <button class="btn btn-sm btn-outline-primary" id="exportChartBtn">
                            <i class="bi bi-download me-1"></i> Export Chart
                        </button>
                    </div>
                </div>
                
                <div class="row mb-4">
                    <div class="col-md-12">
                        <div class="chart-container" style="height: 400px;">
                            <canvas id="detailChart"></canvas>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-12">
                        <div class="table-responsive">
                            <table class="table table-hover table-sm">
                                <thead>
                                    <tr>
                                        <th width="50">#</th>
                                        <th>Configuration</th>
                                        <th width="120">Avg Time</th>
                                        <th width="120">Memory</th>
                                        <th width="120">Ops/Sec</th>
                                        <th width="100">Iterations</th>
                                        <th width="100">Std Dev</th>
                                    </tr>
                                </thead>
                                <tbody>`;

        sortedData.forEach((result, index) => {
            const metadata = result.metadata || {};
            let configLabel = '';

            if (Object.keys(metadata).length === 0) {
                configLabel = '<span class="text-muted">Default configuration</span>';
            } else {
                const metadataItems = Object.entries(metadata).map(([k, v]) =>
                    `<span class="metadata-item"><span class="metadata-key">${k}:</span> <span class="metadata-value">${this.formatValue(v)}</span></span>`
                ).join('');
                configLabel = `<div class="metadata-items">${metadataItems}</div>`;
            }

            html += `
                <tr>
                    <td class="text-center">${index + 1}</td>
                    <td>
                        <div class="metadata-preview small">
                            ${configLabel}
                        </div>
                    </td>
                    <td class="text-nowrap"><span class="badge bg-time">${this.formatTime(result.stats.time.avg)}</span></td>
                    <td class="text-nowrap"><span class="badge bg-memory">${this.formatBytes(result.stats.memory.avg)}</span></td>
                    <td class="text-nowrap"><span class="badge bg-ops">${result.stats.ops.formatted}</span></td>
                    <td class="text-center">${result.iterations}</td>
                    <td class="text-center">${result.stats.time.std_dev.toFixed(3)} ms</td>
                </tr>`;
        });

        html += `
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>`;

        return html;
    }

    initCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        this.charts = {};

        const canvas = document.getElementById('mainPerformanceChart');
        if (!canvas) return;

        const chartDataStr = canvas.getAttribute('data-chart-data');
        if (!chartDataStr) return;

        try {
            const chartData = JSON.parse(chartDataStr);
            this.currentChartData = chartData;
            this.createHorizontalChart(chartData);
            this.createComparisonCharts(chartData);
        } catch (error) {
            console.error('Error parsing chart data:', error);
        }
    }

    createHorizontalChart(chartData) {
        const ctx = document.getElementById('mainPerformanceChart');
        if (!ctx) return;

        const chartLimit = document.querySelector('.chart-limit');
        const limit = chartLimit ? parseInt(chartLimit.value) : 100;

        const metricSelect = document.querySelector('.metric-select');
        const metric = metricSelect ? metricSelect.value : 'time';

        const sortSelect = document.querySelector('.sort-select');
        const sortBy = sortSelect ? sortSelect.value : 'name';

        const indexedData = chartData.labels.map((label, index) => ({
            label,
            time: chartData.time[index],
            memory: chartData.memory[index],
            ops: chartData.ops[index],
            index
        }));

        switch (sortBy) {
            case 'name':
                indexedData.sort((a, b) => a.label.localeCompare(b.label));
                break;
            case 'time':
                indexedData.sort((a, b) => b.time - a.time); // Maior primeiro
                break;
            case 'memory':
                indexedData.sort((a, b) => b.memory - a.memory);
                break;
            case 'ops':
                indexedData.sort((a, b) => b.ops - a.ops);
                break;
        }

        const limitedData = indexedData.slice(0, limit);
        const labels = limitedData.map(d => d.label);
        let data = [];
        let label = '';
        let backgroundColor = '';
        let borderColor = '';

        switch (metric) {
            case 'time':
                data = limitedData.map(d => d.time);
                label = 'Average Time (ms)';
                backgroundColor = 'rgba(99, 102, 241, 0.7)';
                borderColor = 'rgba(99, 102, 241, 1)';
                break;
            case 'memory':
                data = limitedData.map(d => d.memory);
                label = 'Average Memory (MB)';
                backgroundColor = 'rgba(139, 92, 246, 0.7)';
                borderColor = 'rgba(139, 92, 246, 1)';
                break;
            case 'ops':
                data = limitedData.map(d => d.ops);
                label = 'Average Operations/Second (K)';
                backgroundColor = 'rgba(16, 185, 129, 0.7)';
                borderColor = 'rgba(16, 185, 129, 1)';
                break;
        }

        const chartHeight = Math.max(400, labels.length * 25);
        ctx.parentNode.style.height = `${chartHeight}px`;
        ctx.height = chartHeight;

        this.charts.main = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: label,
                    data: data,
                    backgroundColor: backgroundColor,
                    borderColor: borderColor,
                    borderWidth: 1,
                    borderRadius: 4,
                    barPercentage: 0.8,
                    categoryPercentage: 0.9
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                let value = context.parsed.x;

                                if (metric === 'time') {
                                    return `${this.formatTime(value)}`;
                                } else if (metric === 'memory') {
                                    return `${value.toFixed(2)} MB`;
                                } else {
                                    return `${(value * 1000).toLocaleString()} ops/s`;
                                }
                            },
                            title: (tooltipItems) => {
                                return tooltipItems[0].label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: label,
                            color: '#94a3b8',
                            font: {
                                size: 12,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#94a3b8',
                            callback: (value) => {
                                if (metric === 'time') {
                                    if (value < 1) return value.toFixed(2);
                                    if (value < 1000) return value.toFixed(0);
                                    return (value / 1000).toFixed(1) + 'K';
                                } else if (metric === 'memory') {
                                    return value.toFixed(0) + ' MB';
                                } else {
                                    return (value * 1000).toLocaleString();
                                }
                            }
                        }
                    },
                    y: {
                        ticks: {
                            color: '#e2e8f0',
                            font: {
                                size: 10
                            },
                            callback: (value, index) => {
                                const label = labels[index];
                                if (label.length > 60) {
                                    return label.substring(0, 57) + '...';
                                }
                                return label;
                            }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                animation: {
                    duration: 1000,
                    easing: 'easeOutQuart'
                }
            }
        });
    }

    createComparisonCharts(chartData) {
        const slowestCtx = document.getElementById('slowestTestsChart');
        if (slowestCtx) {
            const indexedData = chartData.labels.map((label, index) => ({
                label,
                time: chartData.time[index]
            }));

            const slowest = [...indexedData].sort((a, b) => b.time - a.time).slice(0, 10);
            const slowestLabels = slowest.map(d => {
                if (d.label.length > 30) {
                    return d.label.substring(0, 27) + '...';
                }
                return d.label;
            });
            const slowestTimes = slowest.map(d => d.time);

            this.charts.slowest = new Chart(slowestCtx, {
                type: 'bar',
                data: {
                    labels: slowestLabels,
                    datasets: [{
                        label: 'Time (ms)',
                        data: slowestTimes,
                        backgroundColor: 'rgba(239, 68, 68, 0.7)',
                        borderColor: 'rgba(239, 68, 68, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Time (ms)',
                                color: '#94a3b8'
                            },
                            ticks: {
                                color: '#94a3b8'
                            }
                        },
                        y: {
                            ticks: {
                                color: '#e2e8f0',
                                font: {
                                    size: 9
                                }
                            }
                        }
                    }
                }
            });
        }

        const fastestCtx = document.getElementById('fastestTestsChart');
        if (fastestCtx) {
            const indexedData = chartData.labels.map((label, index) => ({
                label,
                time: chartData.time[index]
            }));

            const fastest = [...indexedData].sort((a, b) => a.time - b.time).slice(0, 10);
            const fastestLabels = fastest.map(d => {
                if (d.label.length > 30) {
                    return d.label.substring(0, 27) + '...';
                }
                return d.label;
            });
            const fastestTimes = fastest.map(d => d.time);

            this.charts.fastest = new Chart(fastestCtx, {
                type: 'bar',
                data: {
                    labels: fastestLabels,
                    datasets: [{
                        label: 'Time (ms)',
                        data: fastestTimes,
                        backgroundColor: 'rgba(16, 185, 129, 0.7)',
                        borderColor: 'rgba(16, 185, 129, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Time (ms)',
                                color: '#94a3b8'
                            },
                            ticks: {
                                color: '#94a3b8'
                            }
                        },
                        y: {
                            ticks: {
                                color: '#e2e8f0',
                                font: {
                                    size: 9
                                }
                            }
                        }
                    }
                }
            });
        }
    }

    initChartControls() {
        const metricSelect = document.querySelector('.metric-select');
        const sortSelect = document.querySelector('.sort-select');
        const chartLimit = document.querySelector('.chart-limit');

        const updateChart = () => {
            if (this.currentChartData && this.charts.main) {
                this.charts.main.destroy();
                this.createHorizontalChart(this.currentChartData);
            }
        };

        if (metricSelect) {
            metricSelect.addEventListener('change', updateChart);
        }
        if (sortSelect) {
            sortSelect.addEventListener('change', updateChart);
        }
        if (chartLimit) {
            chartLimit.addEventListener('change', updateChart);
        }
    }

    initTestDetailChart(testData, handlerName, testName) {
        const ctx = document.getElementById('detailChart');
        if (!ctx) return;

        if (this.charts.detail) {
            this.charts.detail.destroy();
        }

        const labels = testData.map((result, index) => {
            const metadata = result.metadata || {};
            if (Object.keys(metadata).length === 0) {
                return `Config ${index + 1}`;
            }

            const keys = Object.keys(metadata);
            if (keys.length === 1) {
                const key = keys[0];
                const value = metadata[key];
                return `${key}: ${this.truncateValue(value, 20)}`;
            } else {
                const firstKey = keys[0];
                const firstValue = metadata[firstKey];
                return `${firstKey}: ${this.truncateValue(firstValue, 15)}...`;
            }
        });

        const times = testData.map(r => r.stats.time.avg);
        const memory = testData.map(r => r.stats.memory.avg / (1024 * 1024));
        const ops = testData.map(r => r.stats.ops.per_second / 1000);

        const chartHeight = Math.max(300, testData.length * 25);
        ctx.parentNode.style.height = `${chartHeight}px`;
        ctx.height = chartHeight;

        this.charts.detail = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Time (ms)',
                        data: times,
                        backgroundColor: 'rgba(99, 102, 241, 0.7)',
                        borderColor: 'rgba(99, 102, 241, 1)',
                        borderWidth: 1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Ops/Sec (K)',
                        data: ops,
                        backgroundColor: 'rgba(16, 185, 129, 0.7)',
                        borderColor: 'rgba(16, 185, 129, 1)',
                        borderWidth: 1,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Time (ms) / Ops/Sec (K)',
                            color: '#94a3b8'
                        },
                        ticks: {
                            color: '#94a3b8'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#e2e8f0',
                            font: {
                                size: 11
                            }
                        }
                    },
                    y1: {
                        display: false
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                if (context.dataset.label === 'Time (ms)') {
                                    return `${this.formatTime(context.parsed.x)} ms`;
                                } else {
                                    return `${(context.parsed.x * 1000).toLocaleString()} ops/sec`;
                                }
                            }
                        }
                    }
                }
            }
        });

        const exportBtn = document.getElementById('exportChartBtn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                this.exportChartAsImage(ctx);
            });
        }
    }

    exportChartAsImage(canvas) {
        const link = document.createElement('a');
        link.download = `benchmark-chart-${Date.now()}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
    }

    showError(message) {
        const modal = new bootstrap.Modal(document.getElementById('detailsModal'));
        const modalBody = document.getElementById('modalBody');

        modalBody.innerHTML = `
            <div class="text-center py-5">
                <div class="alert alert-danger" role="alert">
                    <i class="bi bi-exclamation-triangle me-2"></i>
                    ${message}
                </div>
                <button class="btn btn-primary mt-3" data-bs-dismiss="modal">
                    <i class="bi bi-x-circle me-1"></i> Close
                </button>
            </div>
        `;

        modal.show();
    }

    escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    formatValue(value) {
        if (Array.isArray(value)) {
            return JSON.stringify(value);
        }
        if (typeof value === 'object' && value !== null) {
            return JSON.stringify(value);
        }
        return String(value);
    }

    truncateValue(value, maxLength) {
        const str = String(value);
        if (str.length > maxLength) {
            return str.substring(0, maxLength - 3) + '...';
        }
        return str;
    }

    formatTime(ms) {
        if (ms <= 0) return '0 ms';
        if (ms < 0.001) {
            return (ms * 1000).toFixed(2) + ' μs';
        }
        if (ms < 1000) return ms.toFixed(3) + ' ms';
        return (ms / 1000).toFixed(3) + ' s';
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.benchmarkReport = new BenchmarkReport();
});