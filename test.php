<?php

use Cuda\Number;

class NumberTest extends Number
{
    public function __construct(private int|float $number) {}

    private function getValue(mixed $operand): int|float
    {
        return ($operand instanceof self) ? $operand->number : $operand;
    }

    public function __add(mixed $left, mixed $right): static
    {
        return new static($this->getValue($left) + $this->getValue($right));
    }

    public function __sub(mixed $left, mixed $right): static
    {
        return new static($this->getValue($left) - $this->getValue($right));
    }

    public function __mul(mixed $left, mixed $right): static
    {
        return new static($this->getValue($left) * $this->getValue($right));
    }

    public function __div(mixed $left, mixed $right): static
    {
        $divisor = $this->getValue($right);
        if ($divisor == 0) {
            throw new \DivisionByZeroError("No division by zero!");
        }
        return new static($this->getValue($left) / $divisor);
    }

    public function __mod(mixed $left, mixed $right): static
    {
        return new static($this->getValue($left) % $this->getValue($right));
    }

    public function __pow(mixed $left, mixed $right): static
    {
        return new static($this->getValue($left) ** $this->getValue($right));
    }

    public function __inc(): void
    {
        $this->number++;
    }

    public function __dec(): void
    {
        $this->number--;
    }

    
    public function __toString(): string
    {
        return (string)$this->number;
    }
}

$a = new NumberTest(10);
$b = new NumberTest(5);

echo "Addition: " . ($a + $b) . "\n";       // 15
echo "Scalar Mult: " . ($a * 2) . "\n";     // 20
echo "Pre-Inc: " . (++$a) . "\n";           // 11
echo "Post-Inc: " . ($a++) . "\n";          // 11 (returns old value)
echo "After Post-Inc: " . $a . "\n";        // 12
echo "Complex: " . (($a + $b) * 2) . "\n";  // (12 + 5) * 2 = 34
echo "Complex 2: " . (($a**2 + $b) * 2) . "\n";  // (12**2 + 5) * 2 = 298