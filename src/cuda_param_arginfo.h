#ifndef CUDA_PARAMS_ARGINFO_H
#define CUDA_PARAMS_ARGINFO_H

#include "php.h"

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_param_attribute_getDtype, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_param_attribute_isList, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_param_attribute_isNullable, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_attr_tensor___construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, dtype, IS_STRING, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, is_list, _IS_BOOL, 1, "true")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, nullable, _IS_BOOL, 1, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_attr_int___construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, bits, IS_LONG, 1, "32")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, nullable, _IS_BOOL, 1, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_attr_float___construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, bits, IS_LONG, 1, "32")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, nullable, _IS_BOOL, 1, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_attr_bool___construct, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, nullable, _IS_BOOL, 1, "false")
ZEND_END_ARG_INFO()


#endif