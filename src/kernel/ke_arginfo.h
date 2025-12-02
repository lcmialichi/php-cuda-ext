#ifndef KE_ARGINFO_H
#define KE_ARGINFO_H

#include "php.h"

#define KE_CLASS_NAME "Cuda\\Kernel" 

ZEND_BEGIN_ARG_INFO_EX(arginfo_kernel_fusion, 0, 0, 1)
    ZEND_ARG_CALLABLE_INFO(0, callable, 0)
ZEND_END_ARG_INFO()

static zend_function_entry kernel_methods[] = {
    ZEND_ME(Kernel, fusion, arginfo_kernel_fusion, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_FE_END
};

static zend_class_entry *register_kernel_class(void)
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, KE_CLASS_NAME, kernel_methods);
    kernel_ce = zend_register_internal_class(&ce);

    return kernel_ce;
}

#endif