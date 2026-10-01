#ifndef CONCAT_KERNELS_H
#define CONCAT_KERNELS_H

#include "../tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

int launch_concat_kernel_host(tensor_t **input_tensors, int num_tensors,
                              tensor_t *output_tensor, int axis,
                              size_t outer_dims, size_t inner_dims, int output_axis_size);

#ifdef __cplusplus
}
#endif

#endif