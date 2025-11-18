#ifndef TENSOR_FABRIC_H
#define TENSOR_FABRIC_H
#endif

#include "php.h"
#include "../tensor.h"

tensor_t *create_tensor_from_php_array(zval *data);
tensor_t *cuda_tensor_create_with_value(int *shape, int ndims, float value);
tensor_t *cuda_tensor_create(const int shape[], int ndims, const float data[]);
tensor_t *cuda_tensor_create_scalar(float value, int *shape, int ndims);
tensor_t *cuda_tensor_create_empty(const int shape[], int ndims);
tensor_t *resolve_result_tensor(tensor_t *t);