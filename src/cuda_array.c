#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "cuda.h"
#include "cuda_array.h"
#include "cuda_array_wrapper.h"
#include "cuda_array_arginfo.h"
#include "tensor.h"

zend_class_entry *cuda_array_ce;

typedef struct {
    tensor_t *tensor_handle;
    zend_array *shape;
    zend_object obj;
} cuda_array_obj;

static zend_object_handlers cuda_array_handlers;

static cuda_array_obj *php_cuda_array_fetch_object(zend_object *obj) {
    return (cuda_array_obj*)((char*)obj - XtOffsetOf(cuda_array_obj, obj));
}

static zend_object *cuda_array_create_object(zend_class_entry *class_type) {
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
        cuda_tensor_destroy(obj->tensor_handle);
        obj->tensor_handle = NULL;
    }
    
    if (obj->shape != NULL) {
        zend_array_destroy(obj->shape);
        obj->shape = NULL;
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

typedef tensor_t* (*tensor_operation_func)(tensor_t*, tensor_t*);
typedef tensor_t* (*scalar_operation_func)(tensor_t*, float);
typedef tensor_t* (*self_operation_func)(tensor_t*);

static void create_result_object(zval *return_value, tensor_t *result_tensor) {
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

static tensor_t* get_second_tensor(zval *other_zv, cuda_array_obj *this_obj) {
    if (Z_TYPE_P(other_zv) == IS_OBJECT && instanceof_function(Z_OBJCE_P(other_zv), cuda_array_ce)) {
        cuda_array_obj *other_obj = php_cuda_array_fetch_object(Z_OBJ_P(other_zv));
        
        if (other_obj->tensor_handle == NULL) {
            zend_throw_error(NULL, "Other tensor not initialized");
            return NULL;
        }
        
        return other_obj->tensor_handle;
    }
    else if (Z_TYPE_P(other_zv) == IS_DOUBLE || Z_TYPE_P(other_zv) == IS_LONG) {
        double scalar_value = (Z_TYPE_P(other_zv) == IS_DOUBLE) ? Z_DVAL_P(other_zv) : (double)Z_LVAL_P(other_zv);
        return cuda_tensor_create_scalar((float)scalar_value, this_obj->tensor_handle->shape, this_obj->tensor_handle->ndims);
    }
    else {
        zend_throw_error(NULL, "Parameter must be CudaArray or number");
        return NULL;
    }
}

static void self_operation_handler(INTERNAL_FUNCTION_PARAMETERS, 
                                   const char* operation_name, 
                                   self_operation_func tensor_func){

    cuda_array_obj *this_obj = php_cuda_array_fetch_object(Z_OBJ_P(ZEND_THIS));
    
    if (this_obj->tensor_handle == NULL) {
        zend_throw_error(NULL, "Tensor not initialized");
        RETURN_NULL();
    }
    
    tensor_t *result_tensor = tensor_func(this_obj->tensor_handle);
    
    if (result_tensor == NULL) {
        zend_throw_error(NULL, "%s failed", operation_name);
        RETURN_NULL();
    }
    
    create_result_object(return_value, result_tensor);

}

static void binary_operation_handler(INTERNAL_FUNCTION_PARAMETERS, 
                                   const char* operation_name, 
                                   tensor_operation_func tensor_func,
                                   scalar_operation_func scalar_func) {
    zval *other_zv;
    
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other_zv)
    ZEND_PARSE_PARAMETERS_END();
    
    cuda_array_obj *this_obj = php_cuda_array_fetch_object(Z_OBJ_P(ZEND_THIS));
    
    if (this_obj->tensor_handle == NULL) {
        zend_throw_error(NULL, "Tensor not initialized");
        RETURN_NULL();
    }
    
    tensor_t *result_tensor = NULL;
    
    if (Z_TYPE_P(other_zv) == IS_OBJECT && instanceof_function(Z_OBJCE_P(other_zv), cuda_array_ce)) {
        cuda_array_obj *other_obj = php_cuda_array_fetch_object(Z_OBJ_P(other_zv));
        
        if (other_obj->tensor_handle == NULL) {
            zend_throw_error(NULL, "Other tensor not initialized");
            RETURN_NULL();
        }
        
        result_tensor = tensor_func(this_obj->tensor_handle, other_obj->tensor_handle);
        
    } else if (Z_TYPE_P(other_zv) == IS_DOUBLE || Z_TYPE_P(other_zv) == IS_LONG) {
        float scalar_value = (Z_TYPE_P(other_zv) == IS_DOUBLE) ? 
                            (float)Z_DVAL_P(other_zv) : (float)Z_LVAL_P(other_zv);
        result_tensor = scalar_func(this_obj->tensor_handle, scalar_value);
        
    } else {
        zend_throw_error(NULL, "Operation requires CudaArray or numeric value");
        RETURN_NULL();
    }
    
    if (result_tensor == NULL) {
        zend_throw_error(NULL, "%s failed", operation_name);
        RETURN_NULL();
    }
    
    create_result_object(return_value, result_tensor);
}

static void static_tensor_creator(INTERNAL_FUNCTION_PARAMETERS, const char* method_name, float value) {
    zval *shape_array;
    
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(shape_array)
    ZEND_PARSE_PARAMETERS_END();
    
    int shape[10] = {0};
    int ndims = 0;
    
    zval *dim;
    int i = 0;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(shape_array), dim) {
        if (i < 10 && Z_TYPE_P(dim) == IS_LONG) {
            shape[i++] = Z_LVAL_P(dim);
        }
    } ZEND_HASH_FOREACH_END();
    ndims = i;
    
    if (ndims == 0) {
        zend_throw_error(NULL, "Invalid shape: must provide dimensions");
        RETURN_NULL();
    }
    
    tensor_t *tensor = cuda_tensor_create_with_value(shape, ndims, value);
    if (!tensor) {
        zend_throw_error(NULL, "Failed to create %s tensor", method_name);
        RETURN_NULL();
    }
    
    create_result_object(return_value, tensor);
}

PHP_METHOD(CudaArray, multiply) {
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Multiplication", cuda_tensor_multiply, cuda_tensor_multiply_scalar);
}

PHP_METHOD(CudaArray, divide) {
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Division", cuda_tensor_divide, cuda_tensor_divide_scalar);
}

PHP_METHOD(CudaArray, add) {
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Addition", cuda_tensor_add, cuda_tensor_add_scalar);
}

PHP_METHOD(CudaArray, subtract) {
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Subtraction", cuda_tensor_subtract, cuda_tensor_subtract_scalar);
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
    
    obj->tensor_handle = cuda_tensor_create_empty(shape, ndims);
    if (!obj->tensor_handle) {
        zend_throw_error(NULL, "Failed to create empty tensor");
        RETURN_NULL();
    }
    
    int index = 0;
    cudaError_t cuda_status = cuda_flatten_php_array_to_gpu(
        data, 
        obj->tensor_handle->data, 
        &index, 
        total_size
    );
    
    if (cuda_status != cudaSuccess) {
        cuda_tensor_destroy(obj->tensor_handle);
        obj->tensor_handle = NULL;
        zend_throw_error(NULL, "Failed to copy data to GPU: %s", cudaGetErrorString(cuda_status));
        RETURN_NULL();
    }
    
    obj->shape = zend_new_array(ndims);
    for (int i = 0; i < ndims; i++) {
        zval dim;
        ZVAL_LONG(&dim, shape[i]);
        zend_hash_index_update(obj->shape, i, &dim);
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
    
    create_result_object(return_value, result_tensor);
}

PHP_METHOD(CudaArray, transpose) {
    self_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Transpose", cuda_tensor_transpose);
}

PHP_METHOD(CudaArray, power) {
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Power", cuda_tensor_power, cuda_tensor_power_scalar);
}

PHP_METHOD(CudaArray, sqrt) {
    self_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Sqrt", cuda_tensor_sqrt);
}

PHP_METHOD(CudaArray, exp) {
    self_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Exp", cuda_tensor_exp);
}

PHP_METHOD(CudaArray, log) {
    self_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Log", cuda_tensor_log);
}

PHP_METHOD(CudaArray, sin) {
    self_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Sin", cuda_tensor_sin);
}

PHP_METHOD(CudaArray, cos) {
    self_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Cos", cuda_tensor_cos);
}

PHP_METHOD(CudaArray, greater) {
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Greater", cuda_tensor_greater, cuda_tensor_greater_scalar);
}

PHP_METHOD(CudaArray, less) {
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Less", cuda_tensor_less, cuda_tensor_less_scalar);
}

PHP_METHOD(CudaArray, equal) {
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Equal", cuda_tensor_equal, cuda_tensor_equal_scalar);
}

PHP_METHOD(CudaArray, notEqual) {
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "NotEqual", cuda_tensor_not_equal, cuda_tensor_not_equal_scalar);
}

PHP_METHOD(CudaArray, greaterEqual) {
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "GreaterEqual", cuda_tensor_greater_equal, cuda_tensor_greater_equal_scalar);
}

PHP_METHOD(CudaArray, lessEqual) {
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "LessEqual", cuda_tensor_less_equal, cuda_tensor_less_equal_scalar);
}

PHP_METHOD(CudaArray, zeros) {
    static_tensor_creator(INTERNAL_FUNCTION_PARAM_PASSTHRU, "zeros", 0.0f);
}

PHP_METHOD(CudaArray, ones) {
    static_tensor_creator(INTERNAL_FUNCTION_PARAM_PASSTHRU, "ones", 1.0f);
}

PHP_METHOD(CudaArray, full) {
    zval *shape_array;
    double value;
    
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY(shape_array)
        Z_PARAM_DOUBLE(value)
    ZEND_PARSE_PARAMETERS_END();
    
    int shape[10] = {0};
    int ndims = 0;
    
    zval *dim;
    int i = 0;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(shape_array), dim) {
        if (i < 10 && Z_TYPE_P(dim) == IS_LONG) {
            shape[i++] = Z_LVAL_P(dim);
        }
    } ZEND_HASH_FOREACH_END();
    ndims = i;
    
    if (ndims == 0) {
        zend_throw_error(NULL, "Invalid shape: must provide dimensions");
        RETURN_NULL();
    }
    
    tensor_t *tensor = cuda_tensor_create_with_value(shape, ndims, (float)value);
    if (!tensor) {
        zend_throw_error(NULL, "Failed to create full tensor");
        RETURN_NULL();
    }
    
    create_result_object(return_value, tensor);
}

PHP_METHOD(CudaArray, reshape) {
    zval *new_shape_array;
    
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(new_shape_array)
    ZEND_PARSE_PARAMETERS_END();
    
    cuda_array_obj *this_obj = php_cuda_array_fetch_object(Z_OBJ_P(ZEND_THIS));
    
    if (this_obj->tensor_handle == NULL) {
        zend_throw_error(NULL, "Tensor not initialized");
        RETURN_NULL();
    }
    int new_shape[10] = {0};
    int new_ndims = 0;
    
    zval *dim_val;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(new_shape_array), dim_val) {
        if (new_ndims >= 10) {
            zend_throw_error(NULL, "Too many dimensions: maximum 10 supported");
            RETURN_NULL();
        }
        
        if (Z_TYPE_P(dim_val) == IS_LONG) {
            new_shape[new_ndims++] = Z_LVAL_P(dim_val);
        } else {
            zend_throw_error(NULL, "Shape dimensions must be integers");
            RETURN_NULL();
        }
    } ZEND_HASH_FOREACH_END();
    
    if (new_ndims == 0) {
        zend_throw_error(NULL, "Invalid shape: must provide at least one dimension");
        RETURN_NULL();
    }
    
    size_t new_total_size = 1;
    for (int i = 0; i < new_ndims; i++) {
        if (new_shape[i] <= 0) {
            zend_throw_error(NULL, "Invalid dimension size: %d", new_shape[i]);
            RETURN_NULL();
        }
        new_total_size *= new_shape[i];
    }
    
    size_t current_total_size = 1;
    for (int i = 0; i < this_obj->tensor_handle->ndims; i++) {
        current_total_size *= this_obj->tensor_handle->shape[i];
    }
    
    if (new_total_size != current_total_size) {
        zend_throw_error(NULL, 
            "Cannot reshape array of size %zu into shape [%d", 
            current_total_size, new_shape[0]);
        
        for (int i = 1; i < new_ndims; i++) {
            zend_error(E_WARNING, ", %d", new_shape[i]);
        }
        zend_error(E_WARNING, "]");
        RETURN_NULL();
    }
    
    tensor_t *reshaped_tensor = cuda_tensor_reshape(this_obj->tensor_handle, new_shape, new_ndims);
    
    if (reshaped_tensor == NULL) {
        zend_throw_error(NULL, "Reshape operation failed");
        RETURN_NULL();
    }
    
    create_result_object(return_value, reshaped_tensor);
}

PHP_METHOD(CudaArray, flatten) {
    cuda_array_obj *this_obj = php_cuda_array_fetch_object(Z_OBJ_P(ZEND_THIS));
    
    if (this_obj->tensor_handle == NULL) {
        zend_throw_error(NULL, "Tensor not initialized");
        RETURN_NULL();
    }
    
    size_t total_size = 1;
    for (int i = 0; i < this_obj->tensor_handle->ndims; i++) {
        total_size *= this_obj->tensor_handle->shape[i];
    }
    
    int flat_shape[] = { (int)total_size };
    
    tensor_t *flat_tensor = cuda_tensor_reshape(this_obj->tensor_handle, flat_shape, 1);
    
    if (flat_tensor == NULL) {
        zend_throw_error(NULL, "Flatten operation failed");
        RETURN_NULL();
    }
    
    create_result_object(return_value, flat_tensor);
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
    PHP_ME(CudaArray, divide, arginfo_cuda_array_divide, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, add, arginfo_cuda_array_add, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, subtract, arginfo_cuda_array_subtract, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, matmul, arginfo_cuda_array_matmul, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, transpose, arginfo_cuda_array_transpose, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, power, arginfo_cuda_array_binary, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, sqrt, arginfo_cuda_array_unary, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, exp, arginfo_cuda_array_unary, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, log, arginfo_cuda_array_unary, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, sin, arginfo_cuda_array_unary, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, cos, arginfo_cuda_array_unary, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, greater, arginfo_cuda_array_binary, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, less, arginfo_cuda_array_binary, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, equal, arginfo_cuda_array_binary, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, notEqual, arginfo_cuda_array_binary, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, greaterEqual, arginfo_cuda_array_binary, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, lessEqual, arginfo_cuda_array_binary, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, getShape, arginfo_cuda_array_getShape, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, toArray, arginfo_cuda_array_toArray, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, reshape, arginfo_cuda_array_reshape, ZEND_ACC_PUBLIC)
    PHP_ME(CudaArray, zeros, arginfo_cuda_array_zeros, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(CudaArray, ones, arginfo_cuda_array_ones, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    PHP_ME(CudaArray, full, arginfo_cuda_array_full, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
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