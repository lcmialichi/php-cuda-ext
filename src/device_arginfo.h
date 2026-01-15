#ifndef DEVICE_ARGINFO_H
#define DEVICE_ARGINFO_H

#include "php.h"

#define DEVICE_CLASS_NAME "Cuda\\Device"

ZEND_BEGIN_ARG_INFO_EX(arginfo_device_construct, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, closure, IS_CALLABLE, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_device_get_name, 0, 0, 0)
ZEND_END_ARG_INFO()


ZEND_BEGIN_ARG_INFO_EX(arginfo_device_fn, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, closure, IS_CALLABLE, 0)
ZEND_END_ARG_INFO()

static zend_function_entry device_methods[] = {
    ZEND_ME(Device, __construct, arginfo_device_construct, ZEND_ACC_PUBLIC)
    ZEND_ME(Device, getName, arginfo_device_get_name, ZEND_ACC_PUBLIC)
    ZEND_ME(Device, fn, arginfo_device_fn, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_FE_END
};

static zend_class_entry *register_device_class(void)
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, DEVICE_CLASS_NAME, device_methods);
    zend_class_entry *device_ce = zend_register_internal_class(&ce);
    device_ce->ce_flags |= ZEND_ACC_FINAL;
    return device_ce;
}

#endif