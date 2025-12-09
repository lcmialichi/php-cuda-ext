#ifndef KERNEL_REFLECTION_H
#define KERNEL_REFLECTION_H

#include "zend_types.h"
#include "kernel_types.h"
#include "cuda_attributes.h"

cuda_method_attribute_args *cuda_extract_method_attribute(
    zend_function *fptr,
    zend_class_entry *ce_attribute);

func_parameter_list_t *cuda_extract_parameter_list(
    zend_function *fptr,
    zend_class_entry *ce_input_attr,
    zend_class_entry *ce_output_attr);
#endif