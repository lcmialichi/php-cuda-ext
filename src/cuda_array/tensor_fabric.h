#ifndef TENSOR_FABRIC_H
#define TENSOR_FABRIC_H
#endif

#include "php.h"
#include "tensor.h"
#include "data_types.h"
#include <stdbool.h>

#define DEFINE_FLATTENER(type_name, c_type)                                              \
    static void flatten_php_array_to_##type_name(zval *data, c_type *buffer, int *index) \
    {                                                                                    \
        if (Z_TYPE_P(data) == IS_ARRAY)                                                  \
        {                                                                                \
            HashTable *ht = Z_ARRVAL_P(data);                                            \
            zval *current;                                                               \
            ZEND_HASH_FOREACH_VAL(ht, current)                                           \
            {                                                                            \
                flatten_php_array_to_##type_name(current, buffer, index);                \
            }                                                                            \
            ZEND_HASH_FOREACH_END();                                                     \
            return;                                                                      \
        }                                                                                \
        c_type value;                                                                    \
        if (Z_TYPE_P(data) == IS_DOUBLE)                                                 \
            value = (c_type)Z_DVAL_P(data);                                              \
        else if (Z_TYPE_P(data) == IS_LONG)                                              \
            value = (c_type)Z_LVAL_P(data);                                              \
        else if (Z_TYPE_P(data) == IS_TRUE)                                              \
            value = (c_type)1;                                                           \
        else if (Z_TYPE_P(data) == IS_FALSE)                                             \
            value = (c_type)0;                                                           \
        else                                                                             \
            value = (c_type)0;                                                           \
        buffer[(*index)++] = value;                                                      \
    }

DEFINE_FLATTENER(float32, float)
DEFINE_FLATTENER(float64, double)
DEFINE_FLATTENER(int32, int32_t)
DEFINE_FLATTENER(int8, int8_t)
DEFINE_FLATTENER(int64, int64_t)
DEFINE_FLATTENER(uint64, uint8_t)
DEFINE_FLATTENER(_bool, bool)

tensor_t *tensor_cast_string(tensor_t *tensor, const char *new_dtype_str);
tensor_t *create_tensor_from_php_array(zval *data, dtype_t dtype);

tensor_t *cuda_tensor_create_with_value(int *shape, int ndims, scalar_value_t value, dtype_t dtype);
tensor_t *cuda_tensor_create(const int shape[], int ndims, const void *data, dtype_t dtype);
tensor_t *cuda_tensor_create_on_host(const int shape[], int ndims, void *data, dtype_t dtype);
tensor_t *cuda_tensor_create_float(const int shape[], int ndims, const float data[]);
tensor_t *cuda_tensor_create_int(const int shape[], int ndims, const int data[]);
tensor_t *cuda_tensor_create_rand(
    int *shape,
    int ndims,
    float min_value,
    float max_value,
    dtype_t dtype,
    unsigned long long seed);

tensor_t *cuda_tensor_create_scalar(float value, int *shape, int ndims);
tensor_t *cuda_tensor_create_empty(const int shape[], int ndims);
tensor_t *cuda_tensor_create_empty_dtype(const int shape[], int ndims, dtype_t dtype);
tensor_t *resolve_result_tensor(tensor_t *t);
tensor_t *cuda_tensor_clone(tensor_t *base_tensor);

int cuda_tensor_get_scalar_value(tensor_t *t, float *result_val, int index);