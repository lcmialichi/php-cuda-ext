<?php

namespace Benchmarks\Support;

use Benchmarks\Exporters\HtmlExporter;
use Benchmarks\Exporters\JsonExporter;

class BenchmarkReport implements \JsonSerializable
{
    /**
     * @param array<BenchmarkClassResult> $results
     * @throws \Exception
     */
    public function __construct(private array $results) {}

    public function getResults(): array
    {
        return $this->results;
    }

    public function saveJSON(string $dir): string
    {
        $exporter = new JsonExporter();
        return $exporter->export($this, $dir);
    }

     public function saveHTML(string $dir): string
    {
        $exporter = new HtmlExporter();
        return $exporter->export($this, $dir);
    }

    public function getDevice(): string
    {
        return cuda_get_device_info()["name"] ?? "";
    }

    public function jsonSerialize(): array
    {
        return [
            "generated_at" => date('Y-m-d H:i:s'),
            "device" => $this->getDevice(),
            "benchmarks" => $this->results
        ];
    }
}
