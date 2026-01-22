#ifndef HOST_ARRAY_CE_H
#define HOST_ARRAY_CE_H

#include "php.h"
#include "zend_interfaces.h"
#include "tensor.h"

extern zend_class_entry *host_array_ce;
extern zend_class_entry *host_array_iterator_ce;

typedef struct _host_array_object
{
    zend_object std;
    tensor_t *tensor;
    zend_bool is_view;
    zend_bool read_only;
} host_array_object;

typedef struct {
    zend_object std;
    zval host_array_zval;
    zend_long current_idx;  
    zend_long max_idx;
} host_array_iterator_object;

ZEND_METHOD(HostArray, __construct);
ZEND_METHOD(HostArray, get);
ZEND_METHOD(HostArray, getShape);
ZEND_METHOD(HostArray, getNdims);
ZEND_METHOD(HostArray, getSize);
ZEND_METHOD(HostArray, getDtype);
ZEND_METHOD(HostArray, toArray);
ZEND_METHOD(HostArray, getElementSize);
ZEND_METHOD(HostArray, getIterator);
ZEND_METHOD(HostArray, count);

ZEND_METHOD(HostArrayIterator, __construct);
ZEND_METHOD(HostArrayIterator, next);
ZEND_METHOD(HostArrayIterator, key);
ZEND_METHOD(HostArrayIterator, current);
ZEND_METHOD(HostArrayIterator, valid);
ZEND_METHOD(HostArrayIterator, rewind);

int host_array_init();
zend_object *host_array_create_object(zend_class_entry *ce);
void host_array_free_object(zend_object *object);

size_t dtype_to_size(dtype_t dtype);
const char *dtype_to_string(dtype_t dtype);
void get_value_for_dtype(void *data, size_t index, zval *return_value, dtype_t dtype);
void *allocate_for_dtype(dtype_t dtype, size_t count);

#endif