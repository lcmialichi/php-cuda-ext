#ifndef KERNEL_REFLECTION_H
#define KERNEL_REFLECTION_H

#include "zend_types.h"
#include "zend_closures.h"
#include "cuda_attributes.h"
#include "kernel_types.h"

func_parameter_list_t *cuda_extract_parameters(zend_function *fptr);
cuda_method_attribute_args *cuda_extract_method_attribute(
    zend_function *fptr,
    zend_class_entry *ce_attribute);

#endif