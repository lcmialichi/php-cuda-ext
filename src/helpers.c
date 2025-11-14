#include "cuda_array_wrapper.h"
#include "php.h"

char* tensor_shape_as_string(tensor_t *tensor) {
    if (tensor->ndims == 0) {
        char *result = (char*)emalloc(8);
        strcpy(result, "scalar");
        return result;
    }
    
    int buffer_size = tensor->ndims * 12 + 2;
    char *result = (char*)emalloc(buffer_size);
    
    char *ptr = result;
    *ptr++ = '(';
    
    for (int i = 0; i < tensor->ndims; i++) {
        if (i > 0) {
            *ptr++ = ',';
            *ptr++ = ' ';
        }
        ptr += sprintf(ptr, "%d", tensor->shape[i]);
    }
    
    *ptr++ = ')';
    *ptr = '\0';
    
    return result;
}
