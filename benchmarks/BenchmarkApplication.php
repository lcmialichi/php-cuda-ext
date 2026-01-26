<?php

namespace Benchmarks;

use Benchmarks\Support\BenchmarkReport;
use Benchmarks\Contracts\BenchmarkInterface;
use Benchmarks\Support\BenchmarkClassResult;

class BenchmarkApplication
{
  public function __construct(private array $benchmarks) {}

  public function run(): BenchmarkReport
  {
    foreach ($this->benchmarks as $benchmark) {

      $classResults = [];
      if (!$benchmark instanceof BenchmarkInterface) {
        throw new \Exception($benchmark::class . " must implement BenchmarkInterface");
      }

      $classResults[] = $this->dispatch($benchmark);
    }

    return new BenchmarkReport($classResults);
  }

  private function dispatch(BenchmarkInterface $benchmark): BenchmarkClassResult
  {
    $result = [];
    foreach ($benchmark->register() as $config) {

      $handler = $config["handler"] ?? null;
      if (!is_callable([$benchmark, $handler])) {
        throw new \Exception($benchmark::class . "::$handler must be callable");
      }

      $runCount = $config["run"] ?? 1;
      $argHandler = "args" . ucfirst($handler);

      for ($i = 0; $i < $runCount; $i++) {
        $args  = [];
        $metadata = [];

        if (isset($config["metadata"])) {
          $metadata = $config["metadata"][$i] ?? [];
        }

        if (method_exists($benchmark, $argHandler)) {
          $args = call_user_func([$benchmark, $argHandler], ($i + 1));
        }

        $result[] = $benchmark->run(
          name: $config["name"] ?? "Not defined",
          exec: [$benchmark, $config["handler"]],
          type: $config["type"] ?? "",
          iterations: $config["iterations"] ?? 10,
          args: $args,
          warmup: $config["warmup"] ?? false,
          metadata: $metadata,
        );
      }
    }

    return new BenchmarkClassResult($benchmark, $result);
  }
}
