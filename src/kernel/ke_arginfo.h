#ifndef KERNEL_ARGINFO_H
#define KERNEL_ARGINFO_H

#include "php.h"

#define KE_CLASS_NAME "Cuda\\Kernel"

ZEND_BEGIN_ARG_INFO_EX(arginfo_kernel_construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_kernel_fn, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, closure, IS_CALLABLE, 0)
    ZEND_ARG_TYPE_INFO(0, attributes, IS_ARRAY, 1)
ZEND_END_ARG_INFO()

static zend_function_entry kernel_methods[] = {
    ZEND_ME(Kernel, __construct, arginfo_kernel_construct, ZEND_ACC_PUBLIC)
    ZEND_ME(Kernel, fn, arginfo_kernel_fn, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_FE_END
};

static zend_class_entry *register_kernel_class(void)
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, KE_CLASS_NAME, kernel_methods);
    zend_class_entry *kernel_ce = zend_register_internal_class(&ce);
    kernel_ce->ce_flags |= ZEND_ACC_ABSTRACT;
    return kernel_ce;
}

#endif