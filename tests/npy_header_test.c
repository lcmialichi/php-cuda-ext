#include "php.h"
#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../src/cuda_array/npy_import.c"

static int accepts(const char *header, int *dimensions, dtype_t *dtype)
{
    int shape[MAX_DIMS];
    size_t elements = 0;
    int valid = npy_parse_header(header, strlen(header), shape, dimensions, &elements, dtype);
    if (valid)
    {
        assert(shape[0] == 2);
        assert(elements == (size_t)(*dimensions == 1 ? 2 : 6));
    }
    return valid;
}

int main(void)
{
    int dimensions = 0;
    dtype_t dtype = DTYPE_UNKNOWN;
    assert(accepts("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }\n", &dimensions, &dtype));
    assert(dimensions == 2 && dtype == DTYPE_FLOAT32);
    assert(accepts("{\"shape\": (2,), \"descr\": '|b1', \"fortran_order\": False}", &dimensions, &dtype));
    assert(dimensions == 1 && dtype == DTYPE_BOOL);
    assert(!accepts("{'descr': '>f4', 'fortran_order': False, 'shape': (2, 3)}", &dimensions, &dtype));
    assert(!accepts("{'descr': '<f4', 'fortran_order': True, 'shape': (2, 3)}", &dimensions, &dtype));
    assert(!accepts("{'descr': '<f4', 'fortran_order': False, 'shape': (0, 3)}", &dimensions, &dtype));
    assert(!accepts("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), 'shape': (2, 3)}", &dimensions, &dtype));
    puts("npy header checks passed");
    return 0;
}