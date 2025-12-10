#ifndef COMPILER_H
#define COMPILER_H

#include "php.h"
#include "kernel_types.h"

extern zend_class_entry *cuda_compiler_ce;

int compiler_init(void);

ZEND_METHOD(Compiler, __construct);
ZEND_METHOD(Compiler, kernel);
ZEND_METHOD(Compiler, device);
ZEND_METHOD(Compiler, compile);
ZEND_METHOD(Compiler, getKernels);
ZEND_METHOD(Compiler, getDevices);


#endif