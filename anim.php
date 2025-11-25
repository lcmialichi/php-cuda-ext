<?php

define('N_POINTS', 500);
define('WIDTH', 100);
define('HEIGHT', 25);
define('FRAMES', 150);
define('SCALE', 14.0);
define('N_TURNS', 1.5);
define('N_POINTS_PER_STRAND', N_POINTS / 2);

$STRAND_1_CHARS = ['#', '@', 'O', '.'];
$STRAND_2_CHARS = ['*', '&', 'o', ':'];

$t_values_half = [];
for ($i = 0; $i < N_POINTS_PER_STRAND; $i++) {
    $t_values_half[] = floatval($i / N_POINTS_PER_STRAND * 2 * M_PI * N_TURNS);
}
$T_half = new CudaArray($t_values_half);

$R = SCALE * 0.5;
$PITCH_SCALE = HEIGHT / (1.5 * M_PI * N_TURNS) * 0.8;

$X_1_php = $T_half->cos()->multiply($R);
$Z_1_php = $T_half->sin()->multiply($R);
$Y_1_php = $T_half * $PITCH_SCALE  - (HEIGHT / 2.0);

$X_2_php = $T_half->cos()->neg()->multiply($R);
$Z_2_php = $T_half->sin()->neg()->multiply($R);
$Y_2_php = $Y_1_php;

$X_base = $X_1_php->concat([$X_2_php]);
$Y_base = $Y_1_php->concat([$Y_2_php]);
$Z_base = $Z_1_php->concat([$Z_2_php]);

for ($t = 0; $t < FRAMES; $t++) {

    $angle = $t * 0.05;
    $cos_a = cos($angle);
    $sin_a = sin($angle);

    $X_rot_part1 = $X_base * $cos_a;
    $X_rot_part2 = $Z_base * $sin_a;
    $X_final = $X_rot_part1 - $X_rot_part2;

    $Z_rot_part1 = $X_base * $sin_a;
    $Z_rot_part2 = $Z_base * $cos_a;
    $Z_final = $Z_rot_part1 + $Z_rot_part2;

    $X_final = $X_final + (WIDTH / 2.0);
    $Y_final = $Y_base + (HEIGHT / 2.0);
  

    $grid = array_fill(0, HEIGHT, array_fill(0, WIDTH, ['char' => ' ', 'depth' => -INF]));

    $z_min = -SCALE * 1.5;
    $z_max = SCALE * 1.5;
    $z_range = $z_max - $z_min;

    $Z_Normalized = ($Z_final - $z_min) / $z_range;

      /**
     * @var CudaArray $X_final
     * @var CudaArray $Y_final
     * @var CudaArray $Z_final
     * @var CudaArray $Z_Normalized
     */
    $X_proj = $X_final->toArray();
    $Y_proj = $Y_final->toArray();
    $Z_depth = $Z_final->toArray();
    $Z_Normalized = $Z_Normalized->toArray();

    for ($i = 0; $i < N_POINTS; $i++) {
        $x = $X_proj[$i];
        $y = $Y_proj[$i];
        $z = $Z_depth[$i];

        $tx = floor($x);
        $ty = floor($y);

        if ($tx >= 0 && $tx < WIDTH && $ty >= 0 && $ty < HEIGHT) {
            $normalized_z = $Z_Normalized[$i];
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