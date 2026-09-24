<?php

use Cuda\Compiler;
use Cuda\CudaArray;

class CLI
{
    public static function green(string $text): string
    {
        return "\033[32m" . $text . "\033[0m";
    }
    public static function red(string $text): string
    {
        return "\033[31m" . $text . "\033[0m";
    }
    public static function yellow(string $text): string
    {
        return "\033[33m" . $text . "\033[0m";
    }
    public static function blue(string $text): string
    {
        return "\033[34m" . $text . "\033[0m";
    }
    public static function magenta(string $text): string
    {
        return "\033[35m" . $text . "\033[0m";
    }
    public static function cyan(string $text): string
    {
        return "\033[36m" . $text . "\033[0m";
    }
    public static function bold(string $text): string
    {
        return "\033[1m" . $text . "\033[0m";
    }
}

class DatasetManager
{
    private string $datasetPath;
    private int $numClasses;

    public array $trainX = [];
    public array $trainY = [];
    public array $testX = [];
    public array $testTargets = [];

    public int $trainSamples = 0;
    public int $testSamples = 0;

    public function __construct(string $datasetPath, int $numClasses = 10)
    {
        $this->datasetPath = $datasetPath;
        $this->numClasses = $numClasses;
    }

    public function loadDataset(float $trainRatio = 0.8): void
    {
        $this->downloadIfNeeded();

        echo CLI::blue("Parsing CSV dataset...\n");
        $lines = file($this->datasetPath, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
        $totalSamples = count($lines);
        $this->trainSamples = (int) ($totalSamples * $trainRatio);
        $this->testSamples = $totalSamples - $this->trainSamples;

        $samples = [];
        foreach ($lines as $line) {
            $values = explode(',', $line);
            $target = (int) array_pop($values);
            $normalizedPixels = array_map(fn($v) => (float) $v / 16.0, $values);

            $oneHot = array_fill(0, $this->numClasses, 0.0);
            $oneHot[$target] = 1.0;

            $samples[] = [$normalizedPixels, $oneHot, $target];
        }

        mt_srand(1337);
        shuffle($samples);

        foreach ($samples as $index => [$normalizedPixels, $oneHot, $target]) {
            if ($index < $this->trainSamples) {
                $this->trainX = array_merge($this->trainX, $normalizedPixels);
                $this->trainY = array_merge($this->trainY, $oneHot);
            } else {
                $this->testX = array_merge($this->testX, $normalizedPixels);
                $this->testTargets[] = $target;
            }
        }

        echo CLI::green("Loaded {$this->trainSamples} training samples and {$this->testSamples} testing samples.\n\n");
    }

    private function downloadIfNeeded(): void
    {
        if (file_exists($this->datasetPath)) {
            return;
        }

        echo CLI::yellow("⏳ Downloading Optdigits Dataset (8x8 digits) from UCI...\n");
        $context = stream_context_create([
            'http' => ['timeout' => 15],
            "ssl" => [
                "verify_peer" => false,
                "verify_peer_name" => false,
            ],
        ]);

        $csvData = file_get_contents("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", false, $context);

        if ($csvData === false) {
            throw new RuntimeException("Failed to download dataset.");
        }

        file_put_contents($this->datasetPath, $csvData);
        echo CLI::green("✅ Download complete.\n\n");
    }
}

class CudaKernelProvider
{
    private $module;

    public function __construct()
    {
        $this->compileKernels();
    }

    private function compileKernels(): void
    {
        $compiler = new Compiler(source: $this->getKernelSource());

        $compiler->kernel('relu_backward', [
            ['name' => 'input', 'type' => 'array', 'dtype' => 'float32'],
            ['name' => 'grad_output', 'type' => 'array', 'dtype' => 'float32'],
            ['name' => 'grad_input', 'type' => 'array', 'dtype' => 'float32'],
            ['name' => 'size', 'dtype' => 'int32']
        ]);

        $compiler->kernel('update_weights', [
            ['name' => 'weights', 'type' => 'array', 'dtype' => 'float32'],
            ['name' => 'gradients', 'type' => 'array', 'dtype' => 'float32'],
            ['name' => 'lr', 'dtype' => 'float32'],
            ['name' => 'size', 'dtype' => 'int32']
        ]);
        
        $compiler->kernel('softmax_cross_entropy', [
            ['name' => 'logits', 'type' => 'array', 'dtype' => 'float32'],
            ['name' => 'target', 'type' => 'array', 'dtype' => 'float32'],
            ['name' => 'probs', 'type' => 'array', 'dtype' => 'float32'],
            ['name' => 'grad_logits', 'type' => 'array', 'dtype' => 'float32'],
            ['name' => 'loss', 'type' => 'array', 'dtype' => 'float32'],
            ['name' => 'rows', 'dtype' => 'int32'],
            ['name' => 'classes', 'dtype' => 'int32']
        ]);

        $compiler->kernel('linear_forward_relu', [
            ['name' => 'X', 'type' => 'array', 'dtype' => 'float32'],
            ['name' => 'W', 'type' => 'array', 'dtype' => 'float32'],
            ['name' => 'b', 'type' => 'array', 'dtype' => 'float32'],
            ['name' => 'output', 'type' => 'array', 'dtype' => 'float32'],
            ['name' => 'batchSize', 'dtype' => 'int32'],
            ['name' => 'inFeatures', 'dtype' => 'int32'],
            ['name' => 'outFeatures', 'dtype' => 'int32']
        ]);

        $compiler->kernel('linear_forward', [
            ['name' => 'X', 'type' => 'array', 'dtype' => 'float32'],
            ['name' => 'W', 'type' => 'array', 'dtype' => 'float32'],
            ['name' => 'b', 'type' => 'array', 'dtype' => 'float32'],
            ['name' => 'output', 'type' => 'array', 'dtype' => 'float32'],
            ['name' => 'batchSize', 'dtype' => 'int32'],
            ['name' => 'inFeatures', 'dtype' => 'int32'],
            ['name' => 'outFeatures', 'dtype' => 'int32']
        ]);

        $this->module = $compiler->compile();
        $this->module->initialize();
    }

    public function getModule()
    {
        return $this->module;
    }

    private function getKernelSource(): string
    {
        return <<<'CUDA'
extern "C" {
    __global__ void update_weights(float* weights, const float* gradients, float lr, int size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            float gradient = gradients[i];
            if (gradient != gradient) return;
            gradient = fminf(fmaxf(gradient, -5.0f), 5.0f);
            weights[i] -= lr * gradient;
        }
    }
    __global__ void relu_backward(const float* input, const float* grad_output, float* grad_input, int size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) { grad_input[i] = (input[i] > 0.0f) ? grad_output[i] : 0.0f; }
    }
    
    __global__ void linear_forward_relu(const float* X, const float* W, const float* b, float* output, int batchSize, int inFeatures, int outFeatures) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int totalElements = batchSize * outFeatures;
        
        if (i < totalElements) {
            int row = i / outFeatures;
            int col = i % outFeatures;
            
            float sum = 0.0f;
            for (int k = 0; k < inFeatures; k++) {
                sum += X[row * inFeatures + k] * W[k * outFeatures + col];
            }
            sum += b[col];
            output[i] = fmaxf(0.0f, sum); // ReLU imbutido no loop
        }
    }

    __global__ void linear_forward(const float* X, const float* W, const float* b, float* output, int batchSize, int inFeatures, int outFeatures) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int totalElements = batchSize * outFeatures;
        
        if (i < totalElements) {
            int row = i / outFeatures;
            int col = i % outFeatures;
            
            float sum = 0.0f;
            for (int k = 0; k < inFeatures; k++) {
                sum += X[row * inFeatures + k] * W[k * outFeatures + col];
            }
            sum += b[col];
            output[i] = sum;
        }
    }

    __global__ void softmax_cross_entropy(const float* logits, const float* target, float* probs, float* grad_logits, float* loss, int rows, int classes) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= rows) return;

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
    }
}

class NeuralNetwork
{
    private CudaArray $W1;
    private CudaArray $b1;
    private CudaArray $W2;
    private CudaArray $b2;

    private CudaKernelProvider $kernels;

    private int $inputFeatures;
    private int $hiddenNodes;
    private int $numClasses;

    public function __construct(CudaKernelProvider $kernels, int $inputFeatures = 64, int $hiddenNodes = 64, int $numClasses = 10)
    {
        $this->kernels = $kernels;
        $this->inputFeatures = $inputFeatures;
        $this->hiddenNodes = $hiddenNodes;
        $this->numClasses = $numClasses;
    }

    public function initWeights(): void
    {
        $this->W1 = CudaArray::rand([$this->inputFeatures, $this->hiddenNodes], -0.1, 0.1, 'float32');
        $this->b1 = CudaArray::zeros([1, $this->hiddenNodes], 'float32');
        $this->W2 = CudaArray::rand([$this->hiddenNodes, $this->numClasses], -0.1, 0.1, 'float32');
        $this->b2 = CudaArray::zeros([1, $this->numClasses], 'float32');
    }

    public function loadModel(string $path, string $versionCheck): bool
    {
        if (!file_exists($path)) {
            return false;
        }

        echo CLI::magenta("Found serialized model on disk. Loading weights...\n");
        $startLoad = microtime(true);
        $candidateModel = unserialize(file_get_contents($path));

        if (($candidateModel['version'] ?? null) !== $versionCheck) {
            echo CLI::yellow("Saved model is from an older config/version. Re-training...\n");
            return false;
        }

        $this->W1 = unserialize($candidateModel['W1']);
        $this->b1 = unserialize($candidateModel['b1']);
        $this->W2 = unserialize($candidateModel['W2']);
        $this->b2 = unserialize($candidateModel['b2']);

        echo CLI::green("Model restored to GPU in " . sprintf("%.2f ms", (microtime(true) - $startLoad) * 1000) . "!\n\n");
        return true;
    }

    public function saveModel(string $path, string $version): void
    {
        echo CLI::magenta("Saving (Serializing) trained CudaArrays to disk...\n");
        $serializedModel = serialize([
            'version' => $version,
            'W1' => serialize($this->W1),
            'b1' => serialize($this->b1),
            'W2' => serialize($this->W2),
            'b2' => serialize($this->b2),
        ]);
        file_put_contents($path, $serializedModel);
        echo CLI::green("Model saved to: $path\n\n");
    }

    public function train(DatasetManager $dataset, int $epochs, int $batchSize, float $learningRate): void
    {
        echo CLI::bold(CLI::blue("Starting GPU Training with Fused Kernels ($batchSize) for $epochs epochs...\n"));
        $trainStart = microtime(true);

        $numBatches = (int) ceil($dataset->trainSamples / $batchSize);
        $module = $this->kernels->getModule();

        for ($epoch = 0; $epoch < $epochs; $epoch++) {
            $epochLoss = 0.0;

            for ($b = 0; $b < $numBatches; $b++) {
                $startIdx = $b * $batchSize;
                $currentBatchSize = min($batchSize, $dataset->trainSamples - $startIdx);

                $batchXHost = array_slice($dataset->trainX, $startIdx * $this->inputFeatures, $currentBatchSize * $this->inputFeatures);
                $batchYHost = array_slice($dataset->trainY, $startIdx * $this->numClasses, $currentBatchSize * $this->numClasses);
                $X = (new CudaArray($batchXHost, 'float32'))->reshape([$currentBatchSize, $this->inputFeatures]);
                $Y = (new CudaArray($batchYHost, 'float32'))->reshape([$currentBatchSize, $this->numClasses]);

                $A1 = CudaArray::zeros([$currentBatchSize, $this->hiddenNodes], 'float32');
                $sizeA1 = $currentBatchSize * $this->hiddenNodes;
                $module->launch('linear_forward_relu', config: $module->autoGrid('linear_forward_relu', $sizeA1), args: [$X, $this->W1, $this->b1, $A1, $currentBatchSize, $this->inputFeatures, $this->hiddenNodes]);

                $Z2 = CudaArray::zeros([$currentBatchSize, $this->numClasses], 'float32');
                $sizeZ2 = $currentBatchSize * $this->numClasses;
                $module->launch('linear_forward', config: $module->autoGrid('linear_forward', $sizeZ2), args: [$A1, $this->W2, $this->b2, $Z2, $currentBatchSize, $this->hiddenNodes, $this->numClasses]);

                $probs = CudaArray::zeros([$currentBatchSize, $this->numClasses], 'float32');
                $dZ2 = CudaArray::zeros([$currentBatchSize, $this->numClasses], 'float32');
                $batchLoss = CudaArray::zeros([$currentBatchSize], 'float32');

                $module->launch('softmax_cross_entropy', config: $module->autoGrid('softmax_cross_entropy', $currentBatchSize), args: [$Z2, $Y, $probs, $dZ2, $batchLoss, $currentBatchSize, $this->numClasses]);

                if ($epoch % 50 === 0 || $epoch === $epochs - 1) {
                    $epochLoss += $batchLoss->sum()->toArray()[0];
                }

                $dW2 = $A1->transpose([1, 0])->matmul($dZ2);
                $db2 = $dZ2->sum(0)->reshape([1, $this->numClasses]);
                $dA1 = $dZ2->matmul($this->W2->transpose([1, 0]));

                $dZ1 = CudaArray::zeros($A1->getShape(), 'float32');
                $size = $A1->getSize();
                $module->launchAsync('relu_backward', config: $module->autoGrid('relu_backward', $size), args: [$A1, $dA1, $dZ1, $size]);

                $X_T = $X->transpose([1, 0]);
                $dW1 = $X_T->matmul($dZ1);

                $tmpDb1 = $dZ1->sum(0);
                $db1 = $tmpDb1->reshape([1, $this->hiddenNodes]);

                foreach ([
                    [$this->W1, $dW1],
                    [$this->b1, $db1],
                    [$this->W2, $dW2],
                    [$this->b2, $db2]
                ] as [$w, $g]) {
                    $sz = $w->getSize();
                    $module->launchAsync('update_weights', config: $module->autoGrid('update_weights', $sz), args: [$w, $g, $learningRate, $sz]);
                }
            }

            if ($epoch % 50 === 0 || $epoch === $epochs - 1) {
                $lossVal = $epochLoss / $dataset->trainSamples;
                echo "   " . CLI::cyan(sprintf("Epoch %4d", $epoch)) . " -> Cross-Entropy Loss: " . CLI::yellow(sprintf("%.5f", $lossVal)) . "\n";

                if (!is_finite($lossVal)) {
                    throw new RuntimeException('Training diverged: non-finite loss detected.');
                }
            }
        }

        echo CLI::green("\nTraining completed in " . sprintf("%.2f ms", (microtime(true) - $trainStart) * 1000) . ".\n");
    }

    public function evaluate(DatasetManager $dataset): void
    {
        echo CLI::bold(CLI::blue("🧪 Running Inference on Test Dataset...\n"));

        $xTest = (new CudaArray($dataset->testX, 'float32'))->reshape([$dataset->testSamples, $this->inputFeatures]);
        $module = $this->kernels->getModule();

        $testA1 = CudaArray::zeros([$dataset->testSamples, $this->hiddenNodes], 'float32');
        $sizeA1 = $dataset->testSamples * $this->hiddenNodes;
        $module->launchAsync('linear_forward_relu', config: $module->autoGrid('linear_forward_relu', $sizeA1), args: [$xTest, $this->W1, $this->b1, $testA1, $dataset->testSamples, $this->inputFeatures, $this->hiddenNodes]);

        $testPredictions = CudaArray::zeros([$dataset->testSamples, $this->numClasses], 'float32');
        $sizeZ2 = $dataset->testSamples * $this->numClasses;
        $module->launchAsync('linear_forward', config: $module->autoGrid('linear_forward', $sizeZ2), args: [$testA1, $this->W2, $this->b2, $testPredictions, $dataset->testSamples, $this->hiddenNodes, $this->numClasses]);

        $predictedClasses = $testPredictions->argMax(1)->toArray();

        $correct = 0;
        for ($i = 0; $i < $dataset->testSamples; $i++) {
            if ((int) $predictedClasses[$i] === $dataset->testTargets[$i]) {
                $correct++;
            }
        }

        $accuracy = ($correct / $dataset->testSamples) * 100.0;
        $accColor = $accuracy > 90 ? "\033[32m" : "\033[33m";

        echo CLI::bold("   Test Accuracy: ") . $accColor . sprintf("%.2f%%", $accuracy) . CLI::bold(" ($correct / {$dataset->testSamples})\n\n\033[0m");

        if ($accuracy < 80.0) {
            throw new RuntimeException(sprintf('Accuracy too low after stable training: %.2f%%', $accuracy));
        }

        echo CLI::green("Stable training check passed.\n");
    }
}

echo CLI::bold(CLI::cyan("\n======================================================\n"));
echo CLI::bold(CLI::cyan("  CUDA PHP Neural Network: 0-9 Digit Recognizer \n"));
echo CLI::bold(CLI::cyan("======================================================\n\n"));

$CONFIG = [
    'modelPath' => __DIR__ . '/trained_model_stable.dat',
    'modelVersion' => 'softmax-ce-v1',
    'datasetPath' => __DIR__ . '/optdigits.csv',
    'inputFeatures' => 64,
    'numClasses' => 10,
    'hiddenNodes' => 64,
    'batchSize' => 256,
    'epochs' => 1000,
    'learningRate' => 0.05
];

try {
    $dataset = new DatasetManager($CONFIG['datasetPath'], $CONFIG['numClasses']);
    $dataset->loadDataset();

    $kernels = new CudaKernelProvider();
    $network = new NeuralNetwork($kernels, $CONFIG['inputFeatures'], $CONFIG['hiddenNodes'], $CONFIG['numClasses']);

    if (!$network->loadModel($CONFIG['modelPath'], $CONFIG['modelVersion'])) {
        $network->initWeights();
        $network->train($dataset, $CONFIG['epochs'], $CONFIG['batchSize'], $CONFIG['learningRate']);
        $network->saveModel($CONFIG['modelPath'], $CONFIG['modelVersion']);
    }

    $network->evaluate($dataset);

} catch (Exception $e) {
    echo CLI::red("\nError: " . $e->getMessage() . "\n");
    exit(1);
}