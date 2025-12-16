#ifndef MODULE_ARGINFO_H
#define MODULE_ARGINFO_H

#include "php.h"

#define MODULE_CLASS_NAME "Cuda\\CompiledModule"

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_module_run, 0, 1, _IS_BOOL, 0)
    ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, config, IS_ARRAY, 1, "[]")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, args, IS_ARRAY, 1, "[]")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_has_kernel, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_get_kernels, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_get_ptx, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_module_save, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, filename, IS_STRING, 0)
ZEND_END_ARG_INFO()

static zend_function_entry module_methods[] = {
    ZEND_ME(CompiledModule, run, arginfo_module_run, ZEND_ACC_PUBLIC)
    ZEND_ME(CompiledModule, hasKernel, arginfo_module_has_kernel, ZEND_ACC_PUBLIC)
    ZEND_ME(CompiledModule, getKernels, arginfo_module_get_kernels, ZEND_ACC_PUBLIC)
    ZEND_ME(CompiledModule, getPtx, arginfo_module_get_ptx, ZEND_ACC_PUBLIC)
    ZEND_ME(CompiledModule, save, arginfo_module_save, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

static zend_class_entry *register_module_class(void)
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, MODULE_CLASS_NAME, module_methods);
    zend_class_entry *module_ce = zend_register_internal_class(&ce);
    module_ce->ce_flags |= ZEND_ACC_FINAL;
    return module_ce;
}

#endif