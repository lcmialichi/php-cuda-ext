#include "cuda_array_wrapper.h"
#include "php.h"

int shapes_compatible_elementwise(tensor_t *a, tensor_t *b)
{
    if (a == NULL || b == NULL)
    {
        php_error_docref(NULL, E_WARNING, "Null tensor input");
        return 0;
    }

    if (a->data == NULL || b->data == NULL)
    {
        php_error_docref(NULL, E_WARNING, "Tensor data is NULL");
        return 0;
    }

    if (a->desc == NULL || b->desc == NULL)
    {
        php_error_docref(NULL, E_WARNING, "Tensor descriptor is NULL");
        return 0;
    }

    if (a->ndims != b->ndims)
    {
        php_error_docref(NULL, E_WARNING, "Different number of dimensions: %d vs %d",
                         a->ndims, b->ndims);
        return 0;
    }

    for (int i = 0; i < a->ndims; i++)
    {
        if (a->shape[i] != b->shape[i])
        {
            php_error_docref(NULL, E_WARNING, "Shape mismatch at dimension %d: %d vs %d",
                             i, a->shape[i], b->shape[i]);
            return 0;
        }
    }

    return 1;
}
