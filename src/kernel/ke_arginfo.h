#ifndef KERNEL_ARGINFO_H
#define KERNEL_ARGINFO_H

#include "php.h"

#define KE_CLASS_NAME "Cuda\\Kernel"

ZEND_BEGIN_ARG_INFO_EX(arginfo_kernel_construct, 0, 0, 0)
ZEND_END_ARG_INFO()

static zend_function_entry kernel_methods[] = {
    ZEND_ME(Kernel, __construct, arginfo_kernel_construct, ZEND_ACC_PUBLIC)
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