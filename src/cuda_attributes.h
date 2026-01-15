#ifndef CUDA_ATTRIBUTES_H
#define CUDA_ATTRIBUTES_H

#include "php.h"

extern zend_class_entry *cuda_attr_kernel_ce;
extern zend_class_entry *cuda_attr_device_ce;
extern zend_class_entry *cuda_attr_input_ce;
extern zend_class_entry *cuda_attr_output_ce;

typedef struct {
 	zend_string *name; 
} cuda_method_attribute_args;

typedef struct {
 	zend_string *dtype;
} cuda_param_attribute_args;

ZEND_METHOD(CudaAttr_Method, __construct);

void cuda_attr_init();

#endif