<?php

use Cuda\CudaArray;
use Cuda\Attr as Attr;

$compiler = new Cuda\Compiler();

#[Attr\Kernel(name: 'conv2d_forward')]
function conv2d_forward(
    #[Attr\Input(dtype: 'float32')] array $input,
    #[Attr\Input(dtype: 'float32')] array $kernel,
    #[Attr\Input(dtype: 'float32')] array $bias,
    #[Attr\Output(dtype: 'float32')] array &$output,
    #[Attr\Input(dtype: 'int32')] int $batch_size,
    #[Attr\Input(dtype: 'int32')] int $in_channels,
    #[Attr\Input(dtype: 'int32')] int $in_height,
    #[Attr\Input(dtype: 'int32')] int $in_width,
    #[Attr\Input(dtype: 'int32')] int $out_channels,
    #[Attr\Input(dtype: 'int32')] int $kernel_h,
    #[Attr\Input(dtype: 'int32')] int $kernel_w,
    #[Attr\Input(dtype: 'int32')] int $pad_h,
    #[Attr\Input(dtype: 'int32')] int $pad_w,
    #[Attr\Input(dtype: 'int32')] int $stride_h,
    #[Attr\Input(dtype: 'int32')] int $stride_w,
    #[Attr\Input(dtype: 'int32')] int $dilation_h,
    #[Attr\Input(dtype: 'int32')] int $dilation_w
): void {
    /** @var \Cuda\Runtime $cuda */
    $out_height = (int)$cuda->math->floor(($in_height + 2 * $pad_h - $dilation_h * ($kernel_h - 1) - 1) / $stride_h) + 1;
    $out_width = (int)$cuda->math->floor(($in_width + 2 * $pad_w - $dilation_w * ($kernel_w - 1) - 1) / $stride_w) + 1;

    $idx = $cuda->blockIdx()->x * ($cuda->blockDim()->x * $cuda->blockDim()->y) +
        $cuda->threadIdx()->y * $cuda->blockDim()->x +
        $cuda->threadIdx()->x;

    $total_elements = $batch_size * $out_channels * $out_height * $out_width;

    if ($idx >= $total_elements) {
        return;
    }

    $batch = (int) ($idx / ($out_channels * $out_height * $out_width));
    $remaining = $idx % ($out_channels * $out_height * $out_width);
    $out_c = (int) ($remaining / ($out_height * $out_width));
    $remaining = $remaining % ($out_height * $out_width);
    $out_h = (int) ($remaining / $out_width);
    $out_w = $remaining % $out_width;

    if ($batch >= $batch_size || $out_c >= $out_channels || $out_h >= $out_height || $out_w >= $out_width) {
        return;
    }

    $sum = 0.0;
    for ($in_c = 0; $in_c < $in_channels; $in_c++) {
        for ($kh = 0; $kh < $kernel_h; $kh++) {
            for ($kw = 0; $kw < $kernel_w; $kw++) {
                $in_h = $out_h * $stride_h - $pad_h + $kh * $dilation_h;
                $in_w = $out_w * $stride_w - $pad_w + $kw * $dilation_w;
                if ($in_h >= 0 && $in_h < $in_height && $in_w >= 0 && $in_w < $in_width) {
                    $input_idx = (($batch * $in_channels + $in_c) * $in_height + $in_h) * $in_width + $in_w;
                    $kernel_idx = ((($out_c * $in_channels + $in_c) * $kernel_h + $kh) * $kernel_w + $kw);
                    $sum += $input[$input_idx] * $kernel[$kernel_idx];
                }
            }
        }
    }

    $sum += $bias[$out_c];
    $outIdx = (int) ((($batch * $out_channels + $out_c) * $out_height + $out_h) * $out_width + $out_w);
    $output[$outIdx] = $sum;
}

$compiler->kernel('conv2d_forward');
$compiled = $compiler->compile();

$batch_size = 2;
$in_channels = 3;
$in_height = 32;
$in_width = 32;
$out_channels = 16;
$kernel_h = 3;
$kernel_w = 3;

$pad_h = 1;
$pad_w = 1;
$stride_h = 1;
$stride_w = 1;
$dilation_h = 1;
$dilation_w = 1;

$out_height = (int) floor(($in_height + 2 * $pad_h - $dilation_h * ($kernel_h - 1) - 1) / $stride_h) + 1;
$out_width = (int) floor(($in_width + 2 * $pad_w - $dilation_w * ($kernel_w - 1) - 1) / $stride_w) + 1;

$input = CudaArray::rand([$batch_size, $in_channels, $in_height, $in_width], 0, 1);
$kernel = CudaArray::rand([$out_channels, $in_channels, $kernel_h, $kernel_w], -1, 1);
$bias = CudaArray::rand([$out_channels], -0.5, 0.5);
$output = CudaArray::zeros([$batch_size, $out_channels, $out_height, $out_width]);

$threads_per_block = 256;
$total_elements = $batch_size * $out_channels * $out_height * $out_width;
$num_blocks = ceil($total_elements / $threads_per_block);

$result = $compiled->run(
    name: 'conv2d_forward',
    config: [
        'grid' => [$num_blocks, 1, 1],
        'block' => [$threads_per_block, 1, 1]
    ],
    args: [
        $input,
        $kernel,
        $bias,
        $output,
        $batch_size,
        $in_channels,
        $in_height,
        $in_width,
        $out_channels,
        $kernel_h,
        $kernel_w,
        $pad_h,
        $pad_w,
        $stride_h,
        $stride_w,
        $dilation_h,
        $dilation_w
    ]
);

var_dump($output[0]->toArray());