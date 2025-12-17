#ifndef CUDA_PARAM_H
#define CUDA_PARAM_H

#include "php.h"
#include "zend_types.h"

typedef struct _cuda_param_info {
    zend_string *name;
    zend_string *dtype;
    zend_bool is_list;
    zend_bool nullable;
} cuda_param_info;

extern zend_class_entry *cuda_param_attribute_ce;

void cuda_param_info_free(cuda_param_info *info);

ZEND_METHOD(CudaParamAttribute, getDtype);
ZEND_METHOD(CudaParamAttribute, isList);
ZEND_METHOD(CudaParamAttribute, isNullable);

ZEND_METHOD(CudaAttr_Tensor, __construct);
ZEND_METHOD(CudaAttr_Tensor, getDtype);
ZEND_METHOD(CudaAttr_Tensor, isList);

ZEND_METHOD(CudaAttr_Int, __construct);
ZEND_METHOD(CudaAttr_Int, getDtype);
ZEND_METHOD(CudaAttr_Int, isList);

ZEND_METHOD(CudaAttr_Float, __construct);
ZEND_METHOD(CudaAttr_Float, getDtype);
ZEND_METHOD(CudaAttr_Float, isList);

ZEND_METHOD(CudaAttr_Bool, __construct);
ZEND_METHOD(CudaAttr_Bool, getDtype);
ZEND_METHOD(CudaAttr_Bool, isList);

void cuda_param_attribute_init(void);
void cuda_register_attributes(void);

#endif