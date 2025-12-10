<?php

namespace Cuda\Builtins;

/** @return int */
function threadIdx() {}

/** @return int */
function blockIdx() {}

/** @return int */
function blockDim() {}

/** @return void */
function syncthreads() {}

/** @return float|int */
function max($a, $b) {}

/** @return float|int */
function min($a, $b) {}

/** @return float|int */
function abs($v) {}

/** @return float */
function pow($a, $b) {}

/** @return float */
function exp($v) {}

/** @return float */
function log($v) {}
