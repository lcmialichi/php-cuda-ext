#ifndef PHP_HELPERS_H
#define PHP_HELPERS_H
#endif

#include "php.h"
#include "tensor.h"

void extract_shape_from_array(zval *data, int *shape, int *ndims);
void flatten_php_array(zval *data, float *flat_array, int *index);
size_t calculate_total_size(zval *data);
int parse_slice_parameter(zval *param, slice_info_t *slice);