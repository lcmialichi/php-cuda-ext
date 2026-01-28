<?php

namespace Benchmarks;

use Benchmarks\Support\BenchmarkReport;
use Benchmarks\Contracts\BenchmarkInterface;
use Benchmarks\Support\Attr\InjectArgs;
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
      for ($i = 0; $i < $runCount; $i++) {
        $metadata = [];

        if (isset($config["metadata"])) {
          $metadata = $config["metadata"][$i] ?? [];
        }

        $result[] = $benchmark->run(
          name: $config["name"] ?? "Not defined",
          exec: [$benchmark, $config["handler"]],
          type: $config["type"] ?? "",
          iterations: $config["iterations"] ?? 10,
          args: $this->getArgs($benchmark, $handler, ($i + 1)),
          warmup: $config["warmup"] ?? false,
          metadata: $metadata,
        );
      }
    }

    return new BenchmarkClassResult($benchmark, $result);
  }

  private function getArgs(BenchmarkInterface $class, string $handler, int $run): array
  {
    $reflection = new \ReflectionMethod($class, $handler);
    $attrs = $reflection->getAttributes(InjectArgs::class);

    if (empty($attrs)) {
      return [];
    }

    return call_user_func([$class, $attrs[0]->newInstance()->getMethod()], $run);
  }
}
