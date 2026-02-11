<?php

use Cuda\CudaArray;
use Cuda\Compiler;
use Cuda\Attr;
use Cuda\CompiledModule;

class kernels
{
    #[Attr\Kernel(name: 'add')]
    public function add(
        #[attr\TensorType(dtype: 'int32')] $tensor,
        #[attr\TensorType(dtype: 'int32')] $secondTensor,
        #[attr\TensorType(dtype: 'int32')] &$result,
        #[attr\IntType] $size
    ): void {
        /** @var Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();
        if ($idx < $size) {
            $result[$idx] = $tensor[$idx] + $secondTensor[$idx];
        }
    }


    #[Attr\Kernel(name: 'div')]
    public function div(
        #[attr\TensorType(dtype: 'int32')] $tensor,
        #[attr\TensorType(dtype: 'int32')] $secondTensor,
        #[attr\TensorType(dtype: 'int32')] &$result,
        #[attr\IntType] $size
    ): void {
        /** @var Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();
        if ($idx < $size) {
            $result[$idx] = $tensor[$idx] / $secondTensor[$idx];
        }
    }

    #[Attr\Kernel(name: 'sub')]
    public function sub(
        #[attr\TensorType(dtype: 'int32')] $tensor,
        #[attr\TensorType(dtype: 'int32')] $secondTensor,
        #[attr\TensorType(dtype: 'int32')] &$result,
        #[attr\IntType] $size
    ): void {
        /** @var Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();
        if ($idx < $size) {
            $result[$idx] = $tensor[$idx] - $secondTensor[$idx];
        }
    }

    #[Attr\Kernel(name: 'mul')]
    public function mul(
        #[attr\TensorType(dtype: 'int32')] $tensor,
        #[attr\TensorType(dtype: 'int32')] $secondTensor,
        #[attr\TensorType(dtype: 'int32')] &$result,
        #[attr\IntType] $size
    ): void {
        /** @var Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();
        if ($idx < $size) {
            $result[$idx] = $tensor[$idx] * $secondTensor[$idx];
        }
    }


    #[Attr\Kernel(name: 'powk')]
    public function pow(
        #[attr\TensorType(dtype: 'int32')] $tensor,
        #[attr\TensorType(dtype: 'int32')] $secondTensor,
        #[attr\TensorType(dtype: 'int32')] &$result,
        #[attr\IntType] $size
    ): void {
        /** @var Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();
        if ($idx < $size) {
            $result[$idx] = $cuda->math->pow($tensor[$idx], $secondTensor[$idx]);
        }
    }

    #[Attr\Kernel(name: 'inc')]
    public function inc(
        #[attr\TensorType(dtype: 'int32')] &$result,
        #[attr\IntType] $size
    ): void {
        /** @var Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();
        if ($idx < $size) {
            $result[$idx]++;
        }
    }

    #[Attr\Kernel(name: 'dec')]
    public function dec(
        #[attr\TensorType(dtype: 'int32')] &$result,
        #[attr\IntType] $size
    ): void {
        /** @var Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();
        if ($idx < $size) {
            $result[$idx]--;
        }
    }
}

class Tensor extends Cuda\Number
{
    private CudaArray $data;
    private static CompiledModule $handler;

    public function __construct(array|CudaArray $data, string $dtype = 'float32')
    {
        $this->data = $data instanceof CudaArray ? $data : new CudaArray($data, $dtype);
    }

    public function data(): CudaArray
    {
        return $this->data;
    }

    public static function init(CompiledModule $handler): void
    {
        self::$handler = $handler;
    }

    public function __inc(): void
    {
        $this->launchUnary('inc', $this->data);
    }

    public function __dec(): void
    {
        $this->launchUnary('dec', $this->data);
    }

    public function __add(mixed $left, mixed $right): static
    {
        return $this->launchBinary('add', $left, $right);
    }

    public function __sub(mixed $left, mixed $right): static
    {
        return $this->launchBinary('asubd', $left, $right);
    }

    public function __mul(mixed $left, mixed $right): static
    {
        return $this->launchBinary('mul', $left, $right);
    }

    public function __div(mixed $left, mixed $right): static
    {
        return $this->launchBinary('div', $left, $right);
    }
    public function __mod(mixed $left, mixed $right): mixed
    {
        throw new RuntimeException("Operation not implemented");
    }

    public function __pow(mixed $left, mixed $right): mixed
    {
        return $this->launchBinary('powk', $left, $right);
    }

    public function getShape(): array
    {
        return $this->data->getShape();
    }

    public function getSize(): int
    {
        return $this->data->getSize();
    }

    public function dtype(): string
    {
        return $this->data->dtype();
    }

    private function launchUnary(string $kernel,  CudaArray $value): static
    {
        self::$handler->launch(
            name: $kernel,
            config: self::$handler->autoGrid($kernel, $value),
            args: [$value, $value->getSize()]
        );

        return new static($value, $this->data->dtype());
    }

    private function launchBinary(string $kernel, Tensor|int|float $first, Tensor|int|float $second): static
    {
        $first = !$first instanceof Tensor
            ? CudaArray::full($second->getShape(), $first,  dtype: $second->dtype())
            : $first->data();

        $second = !$second instanceof Tensor
            ? CudaArray::full($first->getShape(), $second, dtype: $first->dtype())
            :   $second->data();

        if ($second->getShape() != $first->getShape()) {
            throw new \RuntimeException("Invalid shape.");
        }

        $result = CudaArray::zeros($this->data->getShape(), $this->data->dtype());
        self::$handler->launchAsync(
            name: $kernel,
            config: self::$handler->autoGrid($kernel, $first),
            args: [$first, $second, $result, $result->getSize()],
        );

        return new static($result);
    }
}

$compiler = new Compiler();
$kernels = new Kernels();

$ref = new ReflectionClass($kernels);
foreach ($ref->getMethods(ReflectionMethod::IS_PUBLIC) as $method) {
    $compiler->kernel([$kernels, $method->getName()]);
}

$module = $compiler->compile();
Tensor::init($module);

$a = new Tensor([1, 2, 3, 4, 5], dtype: 'int32');
$b = new Tensor([6, 7, 8, 9, 10], dtype: 'int32');

$result = ($b + $a) ** 2;

var_dump($result->data()->toArray());

