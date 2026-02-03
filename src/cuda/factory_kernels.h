#ifndef FACTORY_KERNELS_H
#define FACTORY_KERNELS_H
#include "../data_types.h"

#ifdef __cplusplus
extern "C"
{
#endif
    void launch_assign_scalar_val_kernel(
        void *base,
        dtype_t dtype,
        scalar_value_t value,
        size_t total_elements);

#ifdef __cplusplus
}
#endif

#endif