#ifndef ARGINFO_contiguous_array_H
#define ARGINFO_contiguous_array_H

#include "php.h"

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_get, 0, 0, 1)
    ZEND_ARG_ARRAY_INFO(0, indices, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_at, 0, 0, 1)
    ZEND_ARG_VARIADIC_INFO(0, indices)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_shape, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_toArray, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_getNdims, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_getSize, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_getDtype, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_getElementSize, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_count, 0, 0, 0)
ZEND_END_ARG_INFO();

ZEND_BEGIN_ARG_INFO_EX(arginfo_contiguous_array_toGpu, 0, 0, 0)
ZEND_END_ARG_INFO();

static const zend_function_entry contiguous_array_methods[] = {
    ZEND_ME(ContiguousArray, __construct, arginfo_contiguous_array_construct, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, get, arginfo_contiguous_array_get, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, at, arginfo_contiguous_array_at, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, getShape, arginfo_contiguous_array_shape, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, toArray, arginfo_contiguous_array_toArray, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, getNdims, arginfo_contiguous_array_getNdims, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, getSize, arginfo_contiguous_array_getSize, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, getDtype, arginfo_contiguous_array_getDtype, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, getElementSize, arginfo_contiguous_array_getElementSize, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, count, arginfo_contiguous_array_count, ZEND_ACC_PUBLIC)
    ZEND_ME(ContiguousArray, toGpu, arginfo_contiguous_array_toGpu, ZEND_ACC_PUBLIC)
    ZEND_FE_END
};

#endif