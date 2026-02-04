<?php

use Cuda\Attr as K;
use Cuda\Compiler;
use Cuda\CudaArray;

class KernelDefinitions
{
    #[Cuda\Attr\Kernel(name: 'v_scale')]
    public function scale(
        #[K\TensorType] &$data,
        #[K\FloatType(bits: 32)] $factor,
        #[K\IntType] $n
    ): void {
        /** @var \Cuda\Runtime $cuda */
        $idx = $cuda->globalIdx();
        if ($idx < $n) {
            $data[$idx] = $data[$idx] + $factor / 2 ;
        }
    }
}

$compiler = new Compiler(target: 'sm_75');
$module = $compiler->kernel([new KernelDefinitions(), 'scale'])->compile();
$tensor = CudaArray::ones([512, 512, 512], dtype: 'float32');
$factor = 1.6;
$size = $tensor->getSize();
$module->initialize();

$threadsPerBlock = 256; 
$gridSize = (int) ceil($size / $threadsPerBlock);
$MS = hrtime(true);

$module->run(
    'v_scale',
    config: ['block' => [$threadsPerBlock, 1, 1], 'grid' => [$gridSize, 1, 1]],
    args: [$tensor, $factor, $size],
);

$ME = hrtime(true);
$TS = hrtime(true);

($tensor + $factor) / 2;

$TE = hrtime(true);


var_dump(($ME - $MS) / 1e6, ($TE - $TS) / 1e6);
