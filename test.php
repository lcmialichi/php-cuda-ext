<?php

use Cuda\Compiler;
use Cuda\CudaArray;

function millis(float $start): float
{
    return (microtime(true) - $start) * 1000.0;
}

// Helpers para cores no terminal
class CLI
{
    public static function green(string $text)
    {
        return "\033[32m" . $text . "\033[0m";
    }
    public static function red(string $text)
    {
        return "\033[31m" . $text . "\033[0m";
    }
    public static function yellow(string $text)
    {
        return "\033[33m" . $text . "\033[0m";
    }
    public static function blue(string $text)
    {
        return "\033[34m" . $text . "\033[0m";
    }
    public static function magenta(string $text)
    {
        return "\033[35m" . $text . "\033[0m";
    }
    public static function cyan(string $text)
    {
        return "\033[36m" . $text . "\033[0m";
    }
    public static function bold(string $text)
    {
        return "\033[1m" . $text . "\033[0m";
    }
}

echo CLI::bold(CLI::cyan("\n======================================================\n"));
echo CLI::bold(CLI::cyan("🚀 CUDA PHP Neural Network: 0-9 Digit Recognizer 🚀\n"));
echo CLI::bold(CLI::cyan("======================================================\n\n"));

$modelPath = __DIR__ . '/trained_model_stable.dat';
$modelVersion = 'softmax-ce-v1';
$datasetPath = __DIR__ . '/optdigits.csv';
$inputFeatures = 64;
$numClasses = 10;
$hiddenNodes = 64;

// DEFININDO O TAMANHO DO LOTE PARA POUPAR VRAM NA MTX 570
$batchSize = 256;

// ============================================================================
// 1. DATASET DOWNLOADING AND PARSING
// ============================================================================
if (!file_exists($datasetPath)) {
    echo CLI::yellow("⏳ Downloading Optdigits Dataset (8x8 digits) from UCI...\n");
    $context = stream_context_create(['http' => ['timeout' => 15]]);
    $csvData = file_get_contents("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", false, $context);
    if ($csvData === false)
        throw new RuntimeException("Failed to download dataset.");
    file_put_contents($datasetPath, $csvData);
    echo CLI::green("✅ Download complete.\n\n");
}

echo CLI::blue("📊 Parsing CSV dataset...\n");
$lines = file($datasetPath, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
$xHost = [];
$yHost = [];
$xTestHost = [];
$yTestTargets = [];
$totalSamples = count($lines);
$trainSamples = (int) ($totalSamples * 0.8);
$testSamples = $totalSamples - $trainSamples;
$samples = [];

foreach ($lines as $index => $line) {
    $values = explode(',', $line);
    $target = (int) array_pop($values);
    $normalizedPixels = array_map(fn($v) => (float) $v / 16.0, $values);
    $oneHot = array_fill(0, $numClasses, 0.0);
    $oneHot[$target] = 1.0;

    $samples[] = [$normalizedPixels, $oneHot, $target];
}

mt_srand(1337);
shuffle($samples);

foreach ($samples as $index => [$normalizedPixels, $oneHot, $target]) {
    if ($index < $trainSamples) {
        $xHost = array_merge($xHost, $normalizedPixels);
        $yHost = array_merge($yHost, $oneHot);
    } else {
        $xTestHost = array_merge($xTestHost, $normalizedPixels);
        $yTestTargets[] = $target;
    }
}
echo CLI::green("✅ Loaded $trainSamples training samples and $testSamples testing samples.\n\n");

// ============================================================================
// 2. KERNEL COMPILATION
// ============================================================================
$compiler = new Compiler();

$kernels = <<<'CUDA'
extern "C" {
    __global__ void update_weights(float* weights, const float* gradients, float lr, int size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            float gradient = gradients[i];
            if (gradient != gradient) {
                return;
            }
            gradient = fminf(fmaxf(gradient, -5.0f), 5.0f);
            weights[i] -= lr * gradient;
        }
    }
    __global__ void relu_backward(const float* input, const float* grad_output, float* grad_input, int size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) { grad_input[i] = (input[i] > 0.0f) ? grad_output[i] : 0.0f; }
    }
    __global__ void relu_forward(const float* input, float* output, int size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) { output[i] = fmaxf(0.0f, input[i]); }
    }
    __global__ void softmax_cross_entropy(const float* logits, const float* target, float* probs, float* grad_logits, float* loss, int rows, int classes) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= rows) { return; }

        int base = row * classes;
        float max_logit = logits[base];
        for (int c = 1; c < classes; c++) {
            max_logit = fmaxf(max_logit, logits[base + c]);
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < classes; c++) {
            float p = expf(logits[base + c] - max_logit);
            probs[base + c] = p;
            sum_exp += p;
        }

        float row_loss = 0.0f;
        for (int c = 0; c < classes; c++) {
            float p = probs[base + c] / sum_exp;
            float y = target[base + c];
            probs[base + c] = p;
            grad_logits[base + c] = (p - y) / (float)rows;
            if (y > 0.0f) {
                row_loss -= logf(fmaxf(p, 1.0e-7f));
            }
        }

        loss[row] = row_loss;
    }
}
CUDA;

$compiler->addSource($kernels);
$compiler->kernel('relu_forward', null, [['name' => 'input', 'type' => 'array', 'dtype' => 'float32'], ['name' => 'output', 'type' => 'array', 'dtype' => 'float32'], ['name' => 'size', 'dtype' => 'int32']]);
$compiler->kernel('relu_backward', null, [['name' => 'input', 'type' => 'array', 'dtype' => 'float32'], ['name' => 'grad_output', 'type' => 'array', 'dtype' => 'float32'], ['name' => 'grad_input', 'type' => 'array', 'dtype' => 'float32'], ['name' => 'size', 'dtype' => 'int32']]);
$compiler->kernel('update_weights', null, [['name' => 'weights', 'type' => 'array', 'dtype' => 'float32'], ['name' => 'gradients', 'type' => 'array', 'dtype' => 'float32'], ['name' => 'lr', 'dtype' => 'float32'], ['name' => 'size', 'dtype' => 'int32']]);
$compiler->kernel('softmax_cross_entropy', null, [['name' => 'logits', 'type' => 'array', 'dtype' => 'float32'], ['name' => 'target', 'type' => 'array', 'dtype' => 'float32'], ['name' => 'probs', 'type' => 'array', 'dtype' => 'float32'], ['name' => 'grad_logits', 'type' => 'array', 'dtype' => 'float32'], ['name' => 'loss', 'type' => 'array', 'dtype' => 'float32'], ['name' => 'rows', 'dtype' => 'int32'], ['name' => 'classes', 'dtype' => 'int32']]);
$module = $compiler->compile();

function applyRelu($module, CudaArray $input)
{
    $output = CudaArray::zeros($input->getShape(), 'float32');
    $size = $input->getSize();
    $module->launch('relu_forward', config: $module->autoGrid('relu_forward', $size), args: [$input, $output, $size]);
    return $output;
}

// ============================================================================
// 3. TRAINING OR LOADING PRE-TRAINED MODEL
// ============================================================================
$modelData = null;
if (file_exists($modelPath)) {
    echo CLI::magenta("💾 Found serialized model on disk. Loading weights...\n");
    $startLoad = microtime(true);
    $candidateModel = unserialize(file_get_contents($modelPath));
    if (($candidateModel['version'] ?? null) === $modelVersion) {
        $modelData = $candidateModel;
    } else {
        echo CLI::yellow("⚠️ Saved model is from an older training configuration. Re-training...\n");
    }
}

if ($modelData) {
    $W1 = unserialize($modelData['W1']);
    $b1 = unserialize($modelData['b1']);
    $W2 = unserialize($modelData['W2']);
    $b2 = unserialize($modelData['b2']);
    echo CLI::green("✅ Model restored to GPU in " . sprintf("%.2f ms", millis($startLoad)) . "!\n\n");
} else {
    echo CLI::yellow("⚠️ No saved model found. Initializing training from scratch...\n");

    $W1 = CudaArray::rand([$inputFeatures, $hiddenNodes], -0.1, 0.1, 'float32');
    $b1 = CudaArray::zeros([1, $hiddenNodes], 'float32');
    $W2 = CudaArray::rand([$hiddenNodes, $numClasses], -0.1, 0.1, 'float32');
    $b2 = CudaArray::zeros([1, $numClasses], 'float32');

    $learningRate = 0.05;
    $epochs = 350;

    echo CLI::bold(CLI::blue("\n🔥 Starting GPU Training (MTX 570) with Mini-Batches ($batchSize) for $epochs epochs...\n"));
    $trainStart = microtime(true);

    $numBatches = (int) ceil($trainSamples / $batchSize);

    for ($epoch = 0; $epoch < $epochs; $epoch++) {
        $epochLoss = 0.0;

        for ($b = 0; $b < $numBatches; $b++) {
            $startIdx = $b * $batchSize;
            $currentBatchSize = min($batchSize, $trainSamples - $startIdx);

            $batchXHost = array_slice($xHost, $startIdx * $inputFeatures, $currentBatchSize * $inputFeatures);
            $batchYHost = array_slice($yHost, $startIdx * $numClasses, $currentBatchSize * $numClasses);

            $X = (new CudaArray($batchXHost, 'float32'))->reshape([$currentBatchSize, $inputFeatures]);
            $Y = (new CudaArray($batchYHost, 'float32'))->reshape([$currentBatchSize, $numClasses]);
            $tmpZ1 = $X->matmul($W1);
            $Z1 = $tmpZ1->add($b1);
            $A1 = applyRelu($module, $Z1);

            $tmpZ2 = $A1->matmul($W2);
            $Z2 = $tmpZ2->add($b2);
            $probs = CudaArray::zeros([$currentBatchSize, $numClasses], 'float32');
            $dZ2 = CudaArray::zeros([$currentBatchSize, $numClasses], 'float32');
            $batchLoss = CudaArray::zeros([$currentBatchSize], 'float32');
            $module->launch('softmax_cross_entropy', config: $module->autoGrid('softmax_cross_entropy', $currentBatchSize), args: [$Z2, $Y, $probs, $dZ2, $batchLoss, $currentBatchSize, $numClasses]);

            if ($epoch % 50 === 0 || $epoch === $epochs - 1) {
                $sum = $batchLoss->sum();
                $epochLoss += $sum->toArray()[0];
                unset($sum);
            }

            // BACKWARD PASS
            $A1_T = $A1->transpose([1, 0]);
            $dW2 = $A1_T->matmul($dZ2);

            $tmpDb2 = $dZ2->sum(0);
            $db2 = $tmpDb2->reshape([1, $numClasses]);

            $W2_T = $W2->transpose([1, 0]);
            $dA1 = $dZ2->matmul($W2_T);

            // Inline Relu Backward
            $dZ1 = CudaArray::zeros($Z1->getShape(), 'float32');
            $size = $Z1->getSize();
            $module->launch('relu_backward', config: $module->autoGrid('relu_backward', $size), args: [$Z1, $dA1, $dZ1, $size]);

            $X_T = $X->transpose([1, 0]);
            $dW1 = $X_T->matmul($dZ1);

            $tmpDb1 = $dZ1->sum(0);
            $db1 = $tmpDb1->reshape([1, $hiddenNodes]);

            // Inline Weight Update
            foreach ([
                [$W1, $dW1],
                [$b1, $db1],
                [$W2, $dW2],
                [$b2, $db2]
            ] as [$w, $g]) {
                $sz = $w->getSize();
                $module->launch('update_weights', config: $module->autoGrid('update_weights', $sz), args: [$w, $g, $learningRate, $sz]);
            }

            cuda_synchronize();

            unset(
                $X,
                $Y,
                $tmpZ1,
                $Z1,
                $A1,
                $tmpZ2,
                $Z2,
                $probs,
                $batchLoss,
                $dZ2,
                $A1_T,
                $dW2,
                $tmpDb2,
                $db2,
                $W2_T,
                $dA1,
                $dZ1,
                $X_T,
                $dW1,
                $tmpDb1,
                $db1
            );
        }

        if ($epoch % 50 === 0 || $epoch === $epochs - 1) {
            $lossVal = $epochLoss / $trainSamples;
            echo "   " . CLI::cyan(sprintf("Epoch %4d", $epoch)) . " -> Cross-Entropy Loss: " . CLI::yellow(sprintf("%.5f", $lossVal)) . "\n";

            if (!is_finite($lossVal)) {
                throw new RuntimeException('Training diverged: non-finite loss detected.');
            }
        }
    }

    echo CLI::green("\n✅ Training completed in " . sprintf("%.2f ms", millis($trainStart)) . ".\n");

    echo CLI::magenta("💾 Saving (Serializing) trained CudaArrays to disk...\n");
    $serializedModel = serialize([
        'version' => $modelVersion,
        'W1' => serialize($W1),
        'b1' => serialize($b1),
        'W2' => serialize($W2),
        'b2' => serialize($b2),
    ]);
    file_put_contents($modelPath, $serializedModel);
    echo CLI::green("✅ Model saved to: $modelPath\n\n");
}

// ============================================================================
// 4. INFERENCE & ACCURACY CHECK
// ============================================================================
echo CLI::bold(CLI::blue("🧪 Running Inference on Test Dataset...\n"));
$xTest = (new CudaArray($xTestHost, 'float32'))->reshape([$testSamples, $inputFeatures]);
$testTmpZ1 = $xTest->matmul($W1);
$testZ1 = $testTmpZ1->add($b1);
$testA1 = applyRelu($module, $testZ1);
$testTmpPredictions = $testA1->matmul($W2);
$testPredictions = $testTmpPredictions->add($b2);

$predictedClasses = $testPredictions->argMax(1)->toArray();
$correct = 0;
for ($i = 0; $i < $testSamples; $i++) {
    if ((int) $predictedClasses[$i] === $yTestTargets[$i])
        $correct++;
}
$accuracy = ($correct / $testSamples) * 100.0;
$accColor = $accuracy > 90 ? "\033[32m" : "\033[33m";
echo CLI::bold("   Test Accuracy: ") . $accColor . sprintf("%.2f%%", $accuracy) . CLI::bold(" ($correct / $testSamples)\n\n\033[0m");

unset($xTest, $testTmpZ1, $testZ1, $testA1, $testTmpPredictions, $testPredictions, $predictedClasses);

if ($accuracy < 80.0) {
    throw new RuntimeException(sprintf('Accuracy too low after stable training: %.2f%%', $accuracy));
}

echo CLI::green("✅ Stable training check passed.\n");