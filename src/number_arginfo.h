#ifndef CUDA_NUMBER_ARGINFO_H
#define CUDA_NUMBER_ARGINFO_H

#include "php.h"

ZEND_BEGIN_ARG_INFO_EX(arginfo_number_binary, 0, 0, 2)
    ZEND_ARG_INFO(0, left)
    ZEND_ARG_INFO(0, right)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_number_unary, 0, 0, 0)
ZEND_END_ARG_INFO()

static const zend_function_entry number_methods[] = {
    PHP_ABSTRACT_ME(Number, __add, arginfo_number_binary)
    PHP_ABSTRACT_ME(Number, __sub, arginfo_number_binary)
    PHP_ABSTRACT_ME(Number, __mul, arginfo_number_binary)
    PHP_ABSTRACT_ME(Number, __div, arginfo_number_binary)
    PHP_ABSTRACT_ME(Number, __pow, arginfo_number_binary)
    PHP_ABSTRACT_ME(Number, __mod, arginfo_number_binary)
    PHP_ABSTRACT_ME(Number, __inc, arginfo_number_unary)
    PHP_ABSTRACT_ME(Number, __dec, arginfo_number_unary)
    PHP_FE_END
};

#endif