<?php

use Cuda\Compiler;
use Cuda\CudaArray;

function fail(string $message): never
{
    throw new RuntimeException($message);
}

function assertTrue(bool $condition, string $message): void
{
    if (!$condition) {
        fail($message);
    }
}

function assertArrayClose(array $actual, array $expected, float $epsilon, string $label): void
{
    assertTrue(count($actual) === count($expected), "$label: array length mismatch");

    foreach ($expected as $index => $expectedValue) {
        $actualValue = $actual[$index];
        if (is_bool($expectedValue)) {
            assertTrue($actualValue === $expectedValue, "{$label}[$index]: expected bool value");
            continue;
        }

        assertTrue(abs((float)$actualValue - (float)$expectedValue) <= $epsilon, "{$label}[$index]: expected $expectedValue, got $actualValue");
    }
}

function millis(float $start): float
{
    return (microtime(true) - $start) * 1000.0;
}

$dtypeCases = [
    'float32' => [[1.25, -2.5, 3.75], 1e-5],
    'float64' => [[1.25, -2.5, 3.75], 1e-9],
    'int8' => [[-8, 0, 7], 0.0],
    'int16' => [[-1024, 0, 2048], 0.0],
    'int32' => [[-100000, 0, 100000], 0.0],
    'int64' => [[-10000000000, 0, 10000000000], 0.0],
    'uint8' => [[0, 8, 255], 0.0],
    'uint16' => [[0, 1024, 65535], 0.0],
    'uint32' => [[0, 65536, 2147483647], 0.0],
    'uint64' => [[0, 65536, 2147483647], 0.0],
    'bool' => [[false, true, true], 0.0],
];

foreach ($dtypeCases as $dtype => [$values, $epsilon]) {
    $array = new CudaArray($values, $dtype);
    assertTrue($array->dtype() === $dtype, "dtype() mismatch for $dtype");
    assertArrayClose($array->toArray(), $values, $epsilon, "toArray($dtype)");
}

echo "CudaArray dtype round-trip OK\n";

$compiler = new Compiler();
$compiler->kernel(
    'logistic_regression_step',
    <<<'CUDA'
extern "C" __global__ void logistic_regression_step(
    const float *x,
    const float *y,
    float *w,
    float *pred,
    float *loss,
    int n,
    int d,
    float lr)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) {
        return;
    }

    float z = w[d];
    for (int col = 0; col < d; col++) {
        z += x[row * d + col] * w[col];
    }

    float p = 1.0f / (1.0f + expf(-z));
    pred[row] = p;

    float target = y[row];
    float clipped = fminf(fmaxf(p, 1.0e-6f), 1.0f - 1.0e-6f);
    loss[row] = -(target * logf(clipped) + (1.0f - target) * logf(1.0f - clipped));

    float scale = lr * (p - target) / (float)n;
    for (int col = 0; col < d; col++) {
        atomicAdd(&w[col], -scale * x[row * d + col]);
    }
    atomicAdd(&w[d], -scale);
}
CUDA,
    [
        ['name' => 'x', 'type' => 'array', 'dtype' => 'float32'],
        ['name' => 'y', 'type' => 'array', 'dtype' => 'float32'],
        ['name' => 'w', 'type' => 'array', 'dtype' => 'float32'],
        ['name' => 'pred', 'type' => 'array', 'dtype' => 'float32'],
        ['name' => 'loss', 'type' => 'array', 'dtype' => 'float32'],
        ['name' => 'n', 'dtype' => 'int32'],
        ['name' => 'd', 'dtype' => 'int32'],
        ['name' => 'lr', 'dtype' => 'float32'],
    ]
);

$compileStart = microtime(true);
$module = $compiler->compile();
$compileMs = millis($compileStart);

$serialized = serialize($module);
$unserializeStart = microtime(true);
$module = unserialize($serialized);
$unserializeMs = millis($unserializeStart);

assertTrue($module->hasKernel('logistic_regression_step'), 'unserialized module lost kernel metadata');
assertTrue(is_string($module->getPtx()) && str_contains($module->getPtx(), '.version'), 'unserialized module lost PTX');

echo sprintf(
    "Compiled module in %.2f ms; unserialized cached PTX in %.2f ms; serialized size=%d bytes\n",
    $compileMs,
    $unserializeMs,
    strlen($serialized)
);

$n = 4;
$d = 2;
$x = new CudaArray([
    0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0,
], 'float32');
$y = new CudaArray([0.0, 1.0, 1.0, 1.0], 'float32');
$weights = CudaArray::zeros([$d + 1], 'float32');
$pred = CudaArray::zeros([$n], 'float32');
$loss = CudaArray::zeros([$n], 'float32');
$config = $module->autoGrid('logistic_regression_step', $n);

for ($epoch = 0; $epoch < 250; $epoch++) {
    $module->launch('logistic_regression_step', $config, [$x, $y, $weights, $pred, $loss, $n, $d, 0.8]);
}

$predictions = $pred->toArray();
$trainedWeights = $weights->toArray();
$lossValues = $loss->toArray();
$meanLoss = array_sum($lossValues) / count($lossValues);

assertTrue($predictions[0] < 0.55, 'logistic regression should keep [0,0] near the negative class');
assertTrue($predictions[1] > 0.70, 'logistic regression should classify [0,1] as positive');
assertTrue($predictions[2] > 0.70, 'logistic regression should classify [1,0] as positive');
assertTrue($predictions[3] > 0.90, 'logistic regression should classify [1,1] as strongly positive');
assertTrue($meanLoss < 0.35, "expected trained mean loss below 0.35, got $meanLoss");

echo "Logistic regression predictions: " . json_encode($predictions) . "\n";
echo "Trained weights: " . json_encode($trainedWeights) . "\n";
echo sprintf("Mean loss: %.6f\n", $meanLoss);
echo "All checks passed\n";
