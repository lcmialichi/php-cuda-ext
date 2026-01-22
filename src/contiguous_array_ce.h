#ifndef CONTIGUOUS_ARRAY_CE_H
#define CONTIGUOUS_ARRAY_CE_H

#include "php.h"
#include "zend_interfaces.h"
#include "tensor.h"

extern zend_class_entry *contiguous_array_ce;

typedef struct _contiguous_array_object {
    zend_object std;
    tensor_t *tensor;
    size_t offset;
    int *shape;
    size_t *strides;
    int ndims;
    zend_bool read_only;
} contiguous_array_object;

typedef struct {
    zend_object_iterator intern;
    zval host_array;
    zend_long current_idx;
    zend_long max_idx;
    zval current;
    size_t *indices;
    int iter_ndims;
} contiguous_array_iterator;

ZEND_METHOD(ContiguousArray, __construct);
ZEND_METHOD(ContiguousArray, get);
ZEND_METHOD(ContiguousArray, getShape);
ZEND_METHOD(ContiguousArray, getNdims);
ZEND_METHOD(ContiguousArray, getSize);
ZEND_METHOD(ContiguousArray, getDtype);
ZEND_METHOD(ContiguousArray, toArray);
ZEND_METHOD(ContiguousArray, getElementSize);
ZEND_METHOD(ContiguousArray, count);

int contiguous_array_init();
zend_object *contiguous_array_create_object(zend_class_entry *ce);
void contiguous_array_free_object(zend_object *object);
zend_object *contiguous_array_from_tensor(tensor_t *tensor);

size_t dtype_to_size(dtype_t dtype);
const char *dtype_to_string(dtype_t dtype);
void get_value_for_dtype(void *data, size_t index, zval *return_value, dtype_t dtype);
void set_value_for_dtype(void *data, size_t index, zval *value, dtype_t dtype);
void *allocate_for_dtype(dtype_t dtype, size_t count);

#endif