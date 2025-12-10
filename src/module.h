#ifndef MODULE_H
#define MODULE_H

#include "php.h"
#include "kernel_types.h"

extern zend_class_entry *cuda_module_ce;

int module_init(void);

ZEND_METHOD(CompiledModule, run);
ZEND_METHOD(CompiledModule, hasKernel);
ZEND_METHOD(CompiledModule, getKernels);
ZEND_METHOD(CompiledModule, getPtx);
ZEND_METHOD(CompiledModule, save);

#endif