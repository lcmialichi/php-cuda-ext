#ifndef CUDA_ATTRIBUTES_ARGINFO_H
#define CUDA_ATTRIBUTES_ARGINFO_H

#include "php.h"

ZEND_BEGIN_ARG_INFO_EX(arginfo_cuda_attr_method___construct, 0, 0, 1)
    ZEND_ARG_INFO(0, name)
    ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
ZEND_END_ARG_INFO()

#endif