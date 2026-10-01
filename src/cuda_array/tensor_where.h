#ifndef CUDA_TENSOR_WHERE_H
#define CUDA_TENSOR_WHERE_H

#include "tensor.h"

tensor_t *cuda_tensor_where(tensor_t *condition, tensor_t *on_true, tensor_t *on_false);

#endif