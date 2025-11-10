#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "cuda.h"
#include "cuda_array.h"
#include "cuda_array_wrapper.h"
#include "cuda_array_arginfo.h"

zend_class_entry *cuda_array_ce;

typedef struct {
    tensor_t *tensor_handle;
    zend_array *shape;
    zend_object obj;
} cuda_array_obj;

static zend_object_handlers cuda_array_handlers;

static cuda_array_obj* php_cuda_array_fetch_object(zend_object *obj) {
    return (cuda_array_obj*)((char*)obj - XtOffsetOf(cuda_array_obj, obj));
}

static zend_object* cuda_array_create_object(zend_class_entry *class_type) {
    cuda_array_obj *obj = (cuda_array_obj*)ecalloc(1, sizeof(cuda_array_obj));
    
    zend_object_std_init(&obj->obj, class_type);
    object_properties_init(&obj->obj, class_type);
    
    obj->obj.handlers = &cuda_array_handlers;
    obj->tensor_handle = NULL;
    obj->shape = NULL;

    return &obj->obj;
}

static void cuda_array_free_object(zend_object *object) {
    cuda_array_obj *obj = php_cuda_array_fetch_object(object);
    
    if (obj->tensor_handle != NULL) {
        if (obj->tensor_handle->data != NULL) {
            cudaFree(obj->tensor_handle->data);
        }
        cuda_tensor_destroy(obj->tensor_handle);
    }
    
    if (obj->shape != NULL) {
        zend_array_destroy(obj->shape);
    }
    
    zend_object_std_dtor(&obj->obj);
}

static void extract_shape_from_array(zval *data, int *shape, int *ndims) {
    *ndims = 0;
    
    void extract_shape_recursive(zval *arr, int current_depth) {
        if (Z_TYPE_P(arr) != IS_ARRAY) return;
        if (current_depth >= 10) return;
        
        HashTable *arr_ht = Z_ARRVAL_P(arr);
        int count = zend_array_count(arr_ht);
        
        if (count == 0) return;
        
        shape[current_depth] = count;
        if (current_depth >= *ndims) {
            *ndims = current_depth + 1;
        }
        
        if (count > 0) {
            zval *first = zend_hash_index_find(arr_ht, 0);
            if (first != NULL) {
                extract_shape_recursive(first, current_depth + 1);
            }
        }
    }
    
    extract_shape_recursive(data, 0);
}

static void flatten_php_array(zval *data, float *flat_array, int *index) {
    if (Z_TYPE_P(data) != IS_ARRAY) {
        if (Z_TYPE_P(data) == IS_LONG) {
            flat_array[(*index)++] = (float)Z_LVAL_P(data);
        } else if (Z_TYPE_P(data) == IS_DOUBLE) {
            flat_array[(*index)++] = (float)Z_DVAL_P(data);
        } else if (Z_TYPE_P(data) == IS_TRUE) {
            flat_array[(*index)++] = 1.0f;
        } else if (Z_TYPE_P(data) == IS_FALSE) {
            flat_array[(*index)++] = 0.0f;
        }
        return;
    }
    
    HashTable *ht = Z_ARRVAL_P(data);
    zval *current;
    ZEND_HASH_FOREACH_VAL(ht, current) {
        flatten_php_array(current, flat_array, index);
    } ZEND_HASH_FOREACH_END();
}

static size_t calculate_total_size(zval *data) {
    if (Z_TYPE_P(data) != IS_ARRAY) {
        return 1;
    }
    
    size_t total = 1;
    HashTable *ht = Z_ARRVAL_P(data);
    zval *first = zend_hash_index_find(ht, 0);
    
    if (first != NULL) {
        total = zend_array_count(ht) * calculate_total_size(first);
    }
    
    return total;
}

PHP_METHOD(CudaArray, __construct) {
    zval *data;
    
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(data)
    ZEND_PARSE_PARAMETERS_END();
    
    cuda_array_obj *obj = php_cuda_array_fetch_object(Z_OBJ_P(ZEND_THIS));
    
    int shape[10] = {0};
    int ndims = 0;
    extract_shape_from_array(data, shape, &ndims);
    
    if (ndims == 0) {
        zend_throw_error(NULL, "Invalid array: cannot determine dimensions");
        RETURN_NULL();
    }
    
    size_t total_size = calculate_total_size(data);
    
    float *host_data = (float*)emalloc(total_size * sizeof(float));
    int index = 0;
    flatten_php_array(data, host_data, &index);
    
    float *gpu_data = NULL;
    cudaError_t cuda_status = cudaMalloc((void**)&gpu_data, total_size * sizeof(float));
    
    if (cuda_status != cudaSuccess) {
        efree(host_data);
        zend_throw_error(NULL, "Failed to allocate GPU memory: %s", cudaGetErrorString(cuda_status));
        RETURN_NULL();
    }
    
    cuda_status = cudaMemcpy(gpu_data, host_data, total_size * sizeof(float), cudaMemcpyHostToDevice);
    
    if (cuda_status != cudaSuccess) {
        cudaFree(gpu_data);
        efree(host_data);
        zend_throw_error(NULL, "Failed to copy data to GPU: %s", cudaGetErrorString(cuda_status));
        RETURN_NULL();
    }
    
    efree(host_data);
    
    obj->tensor_handle = cuda_tensor_create_with_data(shape, ndims, gpu_data);
    
    if (obj->tensor_handle == NULL) {
        cudaFree(gpu_data);
        zend_throw_error(NULL, "Failed to create tensor on GPU");
        RETURN_NULL();
    }
    
    obj->shape = zend_new_array(ndims);
    for (int i = 0; i < ndims; i++) {
        zval dim;
        ZVAL_LONG(&dim, shape[i]);
        zend_hash_index_update(obj->shape, i, &dim);
    }
}

PHP_METHOD(CudaArray, multiply) {
    zval *other_zv;
    
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(other_zv, cuda_array_ce)
    ZEND_PARSE_PARAMETERS_END();
    
    cuda_array_obj *this_obj = php_cuda_array_fetch_object(Z_OBJ_P(ZEND_THIS));
    cuda_array_obj *other_obj = php_cuda_array_fetch_object(Z_OBJ_P(other_zv));
    
    if (this_obj->tensor_handle == NULL || other_obj->tensor_handle == NULL) {
        zend_throw_error(NULL, "Tensor not initialized");
        RETURN_NULL();
    }
    
    tensor_t *result_tensor = cuda_tensor_multiply(this_obj->tensor_handle, other_obj->tensor_handle);
    
    if (result_tensor == NULL) {
        zend_throw_error(NULL, "Multiplication failed - incompatible shapes");
        RETURN_NULL();
    }
    
    object_init_ex(return_value, cuda_array_ce);
    cuda_array_obj *result_obj = php_cuda_array_fetch_object(Z_OBJ_P(return_value));
    
    result_obj->tensor_handle = result_tensor;
    
    int *result_shape = cuda_tensor_get_shape(result_tensor);
    int result_ndims = result_tensor->ndims;
    
    result_obj->shape = zend_new_array(result_ndims);
    for (int i = 0; i < result_ndims; i++) {
        zval dim;
        ZVAL_LONG(&dim, result_shape[i]);
        zend_hash_index_update(result_obj->shape, i, &dim);
    }
}


PHP_METHOD(CudaArray, matmul) {
    zval *other_zv;
    
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(other_zv, cuda_array_ce)
    ZEND_PARSE_PARAMETERS_END();
    
    cuda_array_obj *this_obj = php_cuda_array_fetch_object(Z_OBJ_P(ZEND_THIS));
    cuda_array_obj *other_obj = php_cuda_array_fetch_object(Z_OBJ_P(other_zv));
    
    if (this_obj->tensor_handle == NULL || other_obj->tensor_handle == NULL) {
        zend_throw_error(NULL, "Tensor not initialized");
        RETURN_NULL();
    }
    
    tensor_t *result_tensor = cuda_tensor_matmul(this_obj->tensor_handle, other_obj->tensor_handle);
    
    if (result_tensor == NULL) {
        zend_throw_error(NULL, "Matmul failed - incompatible shapes");
        RETURN_NULL();
    }
    
    object_init_ex(return_value, cuda_array_ce);
    cuda_array_obj *result_obj = php_cuda_array_fetch_object(Z_OBJ_P(return_value));
    
    result_obj->tensor_handle = result_tensor;
    
    int *result_shape = cuda_tensor_get_shape(result_tensor);
    int result_ndims = result_tensor->ndims;
    
    result_obj->shape = zend_new_array(result_ndims);
    for (int i = 0; i < result_ndims; i++) {
        zval dim;
        ZVAL_LONG(&dim, result_shape[i]);
        zend_hash_index_update(result_obj->shape, i, &dim);
    }
}

PHP_METHOD(CudaArray, transpose) {
    cuda_array_obj *this_obj = php_cuda_array_fetch_object(Z_OBJ_P(ZEND_THIS));
    
    if (this_obj->tensor_handle == NULL) {
        zend_throw_error(NULL, "Tensor not initialized");
        RETURN_NULL();
    }
    
    tensor_t *result_tensor = cuda_tensor_transpose(this_obj->tensor_handle);
    
    if (result_tensor == NULL) {
        zend_throw_error(NULL, "Transpose failed");
        RETURN_NULL();
    }
    
    object_init_ex(return_value, cuda_array_ce);
    cuda_array_obj *result_obj = php_cuda_array_fetch_object(Z_OBJ_P(return_value));
    
    result_obj->tensor_handle = result_tensor;
    
    int *result_shape = cuda_tensor_get_shape(result_tensor);
    int result_ndims = result_tensor->ndims;
    
    result_obj->shape = zend_new_array(result_ndims);
    for (int i = 0; i < result_ndims; i++) {
        zval dim;
        ZVAL_LONG(&dim, result_shape[i]);
        zend_hash_index_update(result_obj->shape, i, &dim);
    }
}

PHP_METHOD(CudaArray, getShape) {
    cuda_array_obj *obj = php_cuda_array_fetch_object(Z_OBJ_P(ZEND_THIS));
    
    if (obj->shape == NULL) {
        RETURN_NULL();
    }
    
    array_init_size(return_value, zend_array_count(obj->shape));
    
    zval *current;
    ZEND_HASH_FOREACH_VAL(obj->shape, current) {
        zval copy;
        ZVAL_COPY(&copy, current);
        zend_hash_next_index_insert(Z_ARRVAL_P(return_value), &copy);
    } ZEND_HASH_FOREACH_END();
}

PHP_METHOD(CudaArray, toArray) {
    cuda_array_obj *obj = php_cuda_array_fetch_object(Z_OBJ_P(ZEND_THIS));
    
    if (obj->tensor_handle == NULL) {
        RETURN_NULL();
    }
    
    size_t total_size = 1;
    for (int i = 0; i < obj->tensor_handle->ndims; i++) {
        total_size *= obj->tensor_handle->shape[i];
    }
    
    float *host_data = (float*)emalloc(total_size * sizeof(float));
    
    cudaError_t cuda_status = cudaMemcpy(
        host_data, 
        obj->tensor_handle->data, 
        total_size * sizeof(float), 
        cudaMemcpyDeviceToHost
    );
    
    if (cuda_status != cudaSuccess) {
        efree(host_data);
        zend_throw_error(NULL, "Failed to copy data from GPU: %s", cudaGetErrorString(cuda_status));
        RETURN_NULL();
    }
    
    void build_array_from_flat(zval *result, float *data, int *dims, int current_dim, size_t *offset, int total_dims) {
        int size = dims[current_dim];
        array_init_size(result, size);
        
        if (current_dim == total_dims - 1) {
            for (int i = 0; i < size; i++) {
                zval element;
                ZVAL_DOUBLE(&element, data[*offset]);
                zend_hash_index_update(Z_ARRVAL_P(result), i, &element);
                (*offset)++;
            }
        } else {
            for (int i = 0; i < size; i++) {
                zval subarray;
                build_array_from_flat(&subarray, data, dims, current_dim + 1, offset, total_dims);
                zend_hash_index_update(Z_ARRVAL_P(result), i, &subarray);
            }
        }
    }
    
    size_t offset = 0;
    build_array_from_flat(return_value, host_data, obj->tensor_handle->shape, 0, &offset, obj->tensor_handle->ndims);
    
    efree(host_data);
}

static zend_function_entry cuda_array_methods[] = {
    PHP_ME(CudaArray, __construct, arginfo_cuda_array_construct, ZEND_ACC_PUBLIC | ZEND_ACC_CTOR)
    PHP_ME(CudaArray, multiply, arginfo_cuda_array_multiply, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, matmul, arginfo_cuda_array_matmul, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, transpose, arginfo_cuda_array_transpose, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, getShape, arginfo_cuda_array_getShape, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, toArray, arginfo_cuda_array_toArray, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

void cuda_array_init() {
    zend_class_entry ce;
    
    INIT_CLASS_ENTRY(ce, "CudaArray", cuda_array_methods);
    cuda_array_ce = zend_register_internal_class(&ce);
    
    cuda_array_ce->create_object = cuda_array_create_object;
    
    memcpy(&cuda_array_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    cuda_array_handlers.offset = XtOffsetOf(cuda_array_obj, obj);
    cuda_array_handlers.free_obj = cuda_array_free_object;
}