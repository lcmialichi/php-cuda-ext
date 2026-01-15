#ifndef COMPILER_H
#define COMPILER_H

#include "php.h"
#include "kernel_types.h"

extern zend_class_entry *cuda_compiler_ce;

#ifndef ZEND_ACC_IS_ARROW_FUNCTION
# define ZEND_ACC_IS_ARROW_FUNCTION (1U << 27)
#endif

int compiler_init(void);

ZEND_METHOD(Compiler, __construct);
ZEND_METHOD(Compiler, kernel);
ZEND_METHOD(Compiler, device);
ZEND_METHOD(Compiler, compile);
ZEND_METHOD(Compiler, getKernels);
ZEND_METHOD(Compiler, getDevices);


#endif