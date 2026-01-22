#ifndef ARGINFO_HOST_ARRAY_H
#define ARGINFO_HOST_ARRAY_H

#include "php.h"

ZEND_BEGIN_ARG_INFO_EX(arginfo_host_array_construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_host_array_get, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(0, indices, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_host_array_shape, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_host_array_toArray, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_host_array_getNdims, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_host_array_getSize, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_host_array_getDtype, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_host_array_getElementSize, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_host_array_count, 0, 0, 0)
ZEND_END_ARG_INFO();

ZEND_BEGIN_ARG_INFO_EX(arginfo_host_array_getIterator, 0, 0, 0)
ZEND_END_ARG_INFO();

ZEND_BEGIN_ARG_INFO_EX(arginfo_host_array_iterator_rewind, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_host_array_iterator_valid, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_host_array_iterator_current, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_host_array_iterator_key, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_host_array_iterator_next, 0, 0, 0)
ZEND_END_ARG_INFO()

static const zend_function_entry host_array_methods[] = {
    ZEND_ME(HostArray, __construct, arginfo_host_array_construct, ZEND_ACC_PUBLIC)
    ZEND_ME(HostArray, get, arginfo_host_array_get, ZEND_ACC_PUBLIC)
    ZEND_ME(HostArray, getShape, arginfo_host_array_shape, ZEND_ACC_PUBLIC)
    ZEND_ME(HostArray, toArray, arginfo_host_array_toArray, ZEND_ACC_PUBLIC)
    ZEND_ME(HostArray, getNdims, arginfo_host_array_getNdims, ZEND_ACC_PUBLIC)
    ZEND_ME(HostArray, getSize, arginfo_host_array_getSize, ZEND_ACC_PUBLIC)
    ZEND_ME(HostArray, getDtype, arginfo_host_array_getDtype, ZEND_ACC_PUBLIC)
    ZEND_ME(HostArray, getElementSize, arginfo_host_array_getElementSize, ZEND_ACC_PUBLIC)
    ZEND_ME(HostArray, getIterator, arginfo_host_array_getIterator, ZEND_ACC_PUBLIC)
    ZEND_ME(HostArray, count, arginfo_host_array_count, ZEND_ACC_PUBLIC)
    ZEND_FE_END
};

static const zend_function_entry host_array_iterator_methods[] = {
    PHP_ME(HostArrayIterator, rewind, arginfo_host_array_iterator_rewind, ZEND_ACC_PUBLIC)
    PHP_ME(HostArrayIterator, valid, arginfo_host_array_iterator_valid, ZEND_ACC_PUBLIC)
    PHP_ME(HostArrayIterator, current, arginfo_host_array_iterator_current, ZEND_ACC_PUBLIC)
    PHP_ME(HostArrayIterator, key, arginfo_host_array_iterator_key, ZEND_ACC_PUBLIC)
    PHP_ME(HostArrayIterator, next, arginfo_host_array_iterator_next, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

#endif