<?php

namespace Benchmarks\Contracts;

use Benchmarks\Support\BenchmarkReport;

interface ExporterInterface
{
    public function export(BenchmarkReport $report, string $path): string;
}