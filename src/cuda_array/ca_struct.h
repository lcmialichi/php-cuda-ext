#ifndef CA_STRUCT_H
#define CA_STRUCT_H

#include "php.h"
#include "tensor.h"

typedef struct cuda_array_obj
{
    tensor_t *tensor_handle;
    zend_array *shape;
    zend_object obj;
} cuda_array_obj;


#define Z_CUDA_ARRAY_P(zv) ((cuda_array_obj *)((char *)Z_OBJ_P(zv) - XtOffsetOf(cuda_array_obj, obj)))
#endif