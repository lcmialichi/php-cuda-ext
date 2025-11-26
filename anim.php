<?php

define('N_POINTS', 500);
define('WIDTH', 100);
define('HEIGHT', 25);
define('FRAMES', 500);
define('SCALE', 14.0);
define('N_TURNS', 1.5);
define('N_POINTS_PER_STRAND', N_POINTS / 2);
$a = new CudaArray([4, 1, 2, 3]);
$b = new CudaArray([[4, 1, 2, 3], [6, 7, 8, 9]]);
$result = $a / $b;
var_dump($result->toArray());

$a = new CudaArray([[22, 32], [47, 61]]);
$b = new CudaArray([[[4, 1],[ 2, 3]], [[6, 7], [8, 9]]]);
$result = $a / $b;
var_dump($result->toArray());

$result = $a / 11;
var_dump($result->toArray());
exit;

$STRAND_1_CHARS = ['#', '@', 'O', '.'];
$STRAND_2_CHARS = ['*', '&', 'o', ':'];
$t_values_half = [];
for ($i = 0; $i < N_POINTS_PER_STRAND; $i++) {
    $t_values_half[] = floatval($i / N_POINTS_PER_STRAND * 2 * M_PI * N_TURNS);
}

$T_half = CudaArray::ones([FRAMES, count($t_values_half)]) * new CudaArray($t_values_half);

$R = SCALE * 0.5;
$PITCH_SCALE = HEIGHT / (1.5 * M_PI * N_TURNS) * 0.8;

$t_cos = $T_half->cos();
$t_sin = $T_half->sin();

$x = $t_cos * $R;
$z = $t_sin * $R;
$y = $T_half * $PITCH_SCALE - (HEIGHT / 2.0);

$x = $x->concat([$t_cos->neg() * $R], axis: 1);
$y = $y->concat([$y], axis: 1);
$z = $z->concat([$t_sin->neg() * $R], axis: 1);

$angles = (new CudaArray(range(0, FRAMES - 1)))->reshape([FRAMES, 1]) * 0.05;

$cos_a = $angles->cos();
$sin_a = $angles->sin();

$z_min = -SCALE * 1.5;
$z_max = SCALE * 1.5;
$z_range = $z_max - $z_min;

$X_final = ($x * $cos_a) - ($z * $sin_a) + (WIDTH / 2.0);
$Z_final = ($x * $sin_a) - ($z * $cos_a);
$Y_final = $y + (HEIGHT / 2.0);
$Z_Normalized = ($Z_final - $z_min) / $z_range;

$X_final = $X_final->toArray();
$Y_final = $Y_final->toArray();
$Z_final = $Z_final->toArray();
$Z_Normalized = $Z_Normalized->toArray();

for ($t = 0; $t < FRAMES; $t++) {
    $grid = array_fill(0, HEIGHT, array_fill(0, WIDTH, ['char' => ' ', 'depth' => -INF]));

    $X_proj = $X_final[$t];
    $Y_proj = $Y_final[$t];
    $Z_depth = $Z_final[$t];
    $Z_Norm = $Z_Normalized[$t];

    for ($i = 0; $i < N_POINTS; $i++) {
        $x = $X_proj[$i];
        $y = $Y_proj[$i];
        $z = $Z_depth[$i];

        $tx = floor($x);
        $ty = floor($y);

        if ($tx >= 0 && $tx < WIDTH && $ty >= 0 && $ty < HEIGHT) {
            $normalized_z = $Z_Norm[$i];
            $is_strand_1 = ($i < N_POINTS_PER_STRAND);
            $char_set = $is_strand_1 ? $STRAND_1_CHARS : $STRAND_2_CHARS;

            if ($normalized_z > 0.8) {
                $char = $char_set[0];
            } elseif ($normalized_z > 0.6) {
                $char = $char_set[1];
            } elseif ($normalized_z > 0.4) {
                $char = $char_set[2];
            } else {
                $char = $char_set[3];
            }

            $terminal_y = HEIGHT - 1 - $ty;

            if ($z > $grid[$terminal_y][$tx]['depth']) {
                $grid[$terminal_y][$tx]['char'] = $char;
                $grid[$terminal_y][$tx]['depth'] = $z;
            }
        }
    }

    echo "\033[2J\033[H";
    foreach ($grid as $row) {
        $line = '|';
        foreach ($row as $cell) {
            $line .= $cell['char'];
        }
        echo $line . "|\n";
    }
    echo str_repeat('-', WIDTH) . "\n";
    usleep(40000);
}
