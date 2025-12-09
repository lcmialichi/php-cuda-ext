#ifndef KERNEL_REFLECTION_H
#define KERNEL_REFLECTION_H

#include "php.h"
#include "zend_compile.h"
#include "cuda_attributes.h"


cuda_method_attribute_args *cuda_extract_method_attribute(
    zend_function *fptr,
    zend_class_entry *ce_attribute);

#endif