<?php

return [
    'iterations' => [
        'quick' => 3,
        'standard' => 5,
        'thorough' => 10,
    ],
    
    'scenarios' => [
        'memory_intensive' => [
            '1D_HUGE' => [10_000_000],
            '2D_SQUARE' => [4000, 4000],
            '2D_WIDE_ROW' => [1, 8_000_000],
            '2D_TALL_COL' => [8_000_000, 1],
        ],
        'tensor_operations' => [
            'SMALL_TENSORS_16x16x16' => [16, 16, 16],
            'MEDIUM_128x128x128' => [128, 128, 128],
            'LARGE_512x512x16' => [512, 512, 16],
        ],
        'kernel_operations' => [
            'VECTOR_1M' => [1024 * 1024],
            'MATRIX_1Kx1K' => [1024, 1024],
        ]
    ],
    
    'output' => [
        'format' => 'both', // 'console', 'html', 'both', 'json'
        'html_template' => __DIR__ . '/../reports/templates/report.html.twig',
        'output_dir' => __DIR__ . '/../reports/generated/',
        'generate_charts' => true,
    ],
    
    'performance' => [
        'warmup_iterations' => 2,
        'gc_enabled' => true,
        'precision' => 4,
    ]
];