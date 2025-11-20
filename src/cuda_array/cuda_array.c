#include "php.h"
#include "cuda_array.h"
#include "ca_private.h"
#include "ca_arginfo.h"
#include "operations.h"
#include "tensor_fabric.h"
#include "memory_pool.h"    
#include "cuda.h"

zend_class_entry *cuda_array_ce;
static zend_object_handlers cuda_array_handlers;

typedef tensor_t *(*tensor_operation_func)(tensor_t *, tensor_t *);
typedef tensor_t *(*scalar_operation_func)(tensor_t *, float);
typedef tensor_t *(*self_operation_func)(tensor_t *);

static cuda_array_obj *php_cuda_array_fetch_object(zend_object *obj);
static cuda_array_obj *php_cuda_array_fetch_valid_object(zend_object *obj);
static zend_object *cuda_array_create_object(zend_class_entry *class_type);
static void cuda_array_free_object(zend_object *object);
static void create_result_object(zval *return_value, tensor_t *result_tensor);
static tensor_t *get_second_tensor(zval *other_zv, cuda_array_obj *this_obj);
static int parse_slice_parameter(zval *param, slice_info_t *slice);

static void static_tensor_creator(INTERNAL_FUNCTION_PARAMETERS, const char *method_name, float value);

static void unary_operation_handler(INTERNAL_FUNCTION_PARAMETERS,
                                    const char *operation_name,
                                    int operation_type);

static void self_operation_handler(INTERNAL_FUNCTION_PARAMETERS,
                                   const char *operation_name,
                                   self_operation_func tensor_func);

static void binary_operation_handler(INTERNAL_FUNCTION_PARAMETERS, const char *operation_name, int operation_type);

static void sync_php_object_shape(cuda_array_obj *obj, tensor_t *tensor);

ZEND_METHOD(CudaArray, __construct)
{
    zval *data;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ARRAY(data)
    ZEND_PARSE_PARAMETERS_END();

    cuda_array_obj *obj = php_cuda_array_fetch_object(Z_OBJ_P(ZEND_THIS));
    tensor_t *tensor = create_tensor_from_php_array(data);

    if (!tensor)
    {
        RETURN_NULL();
    }

    obj->tensor_handle = tensor;
    sync_php_object_shape(obj, tensor);
}

ZEND_METHOD(CudaArray, multiply)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Multiplication", OP_MUL);
}

ZEND_METHOD(CudaArray, divide)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Division", OP_DIV);
}

ZEND_METHOD(CudaArray, add)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Addition", OP_ADD);
}

ZEND_METHOD(CudaArray, subtract)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Subtraction", OP_SUB);
}

ZEND_METHOD(CudaArray, power)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Power", OP_POW);
}

ZEND_METHOD(CudaArray, greater)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Greater", OP_GT);
}

ZEND_METHOD(CudaArray, less)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Less", OP_LT);
}

ZEND_METHOD(CudaArray, equal)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Equal", OP_EQ);
}

ZEND_METHOD(CudaArray, notEqual)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "NotEqual", OP_NE);
}

ZEND_METHOD(CudaArray, greaterEqual)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "GreaterEqual", OP_GE);
}

ZEND_METHOD(CudaArray, lessEqual)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "LessEqual", OP_LE);
}

ZEND_METHOD(CudaArray, zeros)
{
    static_tensor_creator(INTERNAL_FUNCTION_PARAM_PASSTHRU, "zeros", 0.0f);
}

ZEND_METHOD(CudaArray, ones)
{
    static_tensor_creator(INTERNAL_FUNCTION_PARAM_PASSTHRU, "ones", 1.0f);
}

ZEND_METHOD(CudaArray, transpose)
{
    self_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Transpose", cuda_tensor_transpose);
}

ZEND_METHOD(CudaArray, sqrt)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Sqrt", OP_SQRT);
}

ZEND_METHOD(CudaArray, exp)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Exp", OP_EXP);
}

ZEND_METHOD(CudaArray, log)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Log", OP_LOG);
}

ZEND_METHOD(CudaArray, sin)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Sin", OP_SIN);
}

ZEND_METHOD(CudaArray, cos)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Cos", OP_COS);
}

ZEND_METHOD(CudaArray, tan)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Tan", OP_TAN);
}

ZEND_METHOD(CudaArray, abs)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Abs", OP_ABS);
}

ZEND_METHOD(CudaArray, neg)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Neg", OP_NEG);
}

ZEND_METHOD(CudaArray, matmul)
{
    zval *other_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_OBJECT_OF_CLASS(other_zv, cuda_array_ce)
    ZEND_PARSE_PARAMETERS_END();

    cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));
    cuda_array_obj *other_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(other_zv));

    if (this_obj->tensor_handle == NULL || other_obj->tensor_handle == NULL)
    {
        zend_throw_error(NULL, "Tensor not initialized");
        RETURN_NULL();
    }

    tensor_t *result_tensor = cuda_tensor_matmul(this_obj->tensor_handle, other_obj->tensor_handle);

    if (result_tensor == NULL)
    {
        zend_throw_error(NULL, "Matmul failed - incompatible shapes");
        RETURN_NULL();
    }

    create_result_object(return_value, result_tensor);
}

ZEND_METHOD(CudaArray, full)
{
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
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(shape_array), dim)
    {
        if (i < 10 && Z_TYPE_P(dim) == IS_LONG)
        {
            shape[i++] = Z_LVAL_P(dim);
        }
    }
    ZEND_HASH_FOREACH_END();
    ndims = i;

    if (ndims == 0)
    {
        zend_throw_error(NULL, "Invalid shape: must provide dimensions");
        RETURN_NULL();
    }

    tensor_t *tensor = cuda_tensor_create_with_value(shape, ndims, (float)value);
    if (!tensor)
    {
        zend_throw_error(NULL, "Failed to create full tensor");
        RETURN_NULL();
    }

    create_result_object(return_value, tensor);
}

ZEND_METHOD(CudaArray, reshape)
{
    zval *new_shape_array;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ARRAY(new_shape_array)
    ZEND_PARSE_PARAMETERS_END();

    cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));

    if (this_obj->tensor_handle == NULL)
    {
        zend_throw_error(NULL, "Tensor not initialized");
        RETURN_NULL();
    }
    int new_shape[10] = {0};
    int new_ndims = 0;

    zval *dim_val;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(new_shape_array), dim_val)
    {
        if (new_ndims >= 10)
        {
            zend_throw_error(NULL, "Too many dimensions: maximum 10 supported");
            RETURN_NULL();
        }

        if (Z_TYPE_P(dim_val) == IS_LONG)
        {
            new_shape[new_ndims++] = Z_LVAL_P(dim_val);
        }
        else
        {
            zend_throw_error(NULL, "Shape dimensions must be integers");
            RETURN_NULL();
        }
    }
    ZEND_HASH_FOREACH_END();

    if (new_ndims == 0)
    {
        zend_throw_error(NULL, "Invalid shape: must provide at least one dimension");
        RETURN_NULL();
    }

    size_t new_total_size = 1;
    for (int i = 0; i < new_ndims; i++)
    {
        if (new_shape[i] <= 0)
        {
            zend_throw_error(NULL, "Invalid dimension size: %d", new_shape[i]);
            RETURN_NULL();
        }
        new_total_size *= new_shape[i];
    }

    size_t current_total_size = 1;
    for (int i = 0; i < this_obj->tensor_handle->ndims; i++)
    {
        current_total_size *= this_obj->tensor_handle->shape[i];
    }

    if (new_total_size != current_total_size)
    {
        zend_throw_error(NULL,
                         "Cannot reshape array of size %zu into shape [%d",
                         current_total_size, new_shape[0]);

        for (int i = 1; i < new_ndims; i++)
        {
            zend_error(E_WARNING, ", %d", new_shape[i]);
        }
        zend_error(E_WARNING, "]");
        RETURN_NULL();
    }

    tensor_t *reshaped_tensor = cuda_tensor_reshape(this_obj->tensor_handle, new_shape, new_ndims);

    if (reshaped_tensor == NULL)
    {
        zend_throw_error(NULL, "Reshape operation failed");
        RETURN_NULL();
    }

    create_result_object(return_value, reshaped_tensor);
}

ZEND_METHOD(CudaArray, flatten)
{
    cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));

    if (this_obj->tensor_handle == NULL)
    {
        zend_throw_error(NULL, "Tensor not initialized");
        RETURN_NULL();
    }

    size_t total_size = 1;
    for (int i = 0; i < this_obj->tensor_handle->ndims; i++)
    {
        total_size *= this_obj->tensor_handle->shape[i];
    }

    int flat_shape[] = {(int)total_size};

    tensor_t *flat_tensor = cuda_tensor_reshape(this_obj->tensor_handle, flat_shape, 1);

    if (flat_tensor == NULL)
    {
        zend_throw_error(NULL, "Flatten operation failed");
        RETURN_NULL();
    }

    create_result_object(return_value, flat_tensor);
}

ZEND_METHOD(CudaArray, getShape)
{
    cuda_array_obj *obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));

    if (obj->shape == NULL)
    {
        RETURN_NULL();
    }

    array_init_size(return_value, zend_array_count(obj->shape));

    zval *current;
    ZEND_HASH_FOREACH_VAL(obj->shape, current)
    {
        zval copy;
        ZVAL_COPY(&copy, current);
        zend_hash_next_index_insert(Z_ARRVAL_P(return_value), &copy);
    }
    ZEND_HASH_FOREACH_END();
}

ZEND_METHOD(CudaArray, getStrides)
{
    cuda_array_obj *obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));
    tensor_t *t = obj->tensor_handle;

    if (!t->strides)
    {
        RETURN_NULL();
    }

    array_init_size(return_value, t->ndims);

    for (int i = 0; i < t->ndims; i++)
    {
        add_next_index_long(return_value, t->strides[i]);
    }
}

ZEND_METHOD(CudaArray, toArray)
{
    cuda_array_obj *obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));

    if (!obj->tensor_handle)
    {
        RETURN_NULL();
    }

    tensor_t *tensor = obj->tensor_handle;

    tensor_t *base = tensor->is_view ? tensor->base_tensor : tensor;
    size_t base_total = base->total_size;

    float *host_data = emalloc(base_total * sizeof(float));

    cudaError_t status = cudaMemcpy(
        host_data,
        base->data,
        base_total * sizeof(float),
        cudaMemcpyDeviceToHost);

    if (status != cudaSuccess)
    {
        efree(host_data);
        zend_throw_error(NULL, "Failed to copy data from GPU: %s", cudaGetErrorString(status));
        RETURN_NULL();
    }

    void build_recursive(
        zval * result,
        float *data,
        int dim,
        tensor_t *t,
        size_t current_offset)
    {
        array_init(result);

        if (dim == t->ndims)
        {
            zval v;
            ZVAL_DOUBLE(&v, data[current_offset]);
            RETVAL_ZVAL(result, 0, 0);
            return;
        }

        int size = t->shape[dim];
        size_t stride = t->strides[dim];

        for (int i = 0; i < size; i++)
        {

            size_t child_offset = current_offset + i * stride;

            if (dim == t->ndims - 1)
            {
                zval val;
                ZVAL_DOUBLE(&val, data[child_offset]);
                zend_hash_index_update(Z_ARRVAL_P(result), i, &val);
            }
            else
            {
                zval sub;
                build_recursive(&sub, data, dim + 1, t, child_offset);
                zend_hash_index_update(Z_ARRVAL_P(result), i, &sub);
            }
        }
    }

    size_t offset_elements = tensor->gpu_offset / sizeof(float);
    build_recursive(return_value, host_data, 0, tensor, offset_elements);

    efree(host_data);
}

ZEND_METHOD(CudaArray, __invoke)
{
    zval *slices;
    int slice_count;

    ZEND_PARSE_PARAMETERS_START(0, -1)
    Z_PARAM_VARIADIC('*', slices, slice_count)
    ZEND_PARSE_PARAMETERS_END();

    cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));

    if (this_obj->tensor_handle == NULL)
    {
        zend_throw_error(NULL, "Tensor not initialized");
        RETURN_NULL();
    }

    int ndim = this_obj->tensor_handle->ndims;
    slice_info_t *slice_info = (slice_info_t *)emalloc(ndim * sizeof(slice_info_t));

    if (slice_count == 0)
    {
        int ndims = this_obj->tensor_handle->ndims;
        slice_info_t *slice_info = (slice_info_t *)emalloc(ndims * sizeof(slice_info_t));

        for (int i = 0; i < ndims; i++)
        {
            slice_info[i].type = SLICE_ALL;
        }
    }

    for (int i = 0; i < ndim; i++)
    {
        if (i < slice_count)
        {
            if (!parse_slice_parameter(&slices[i], &slice_info[i]))
            {
                efree(slice_info);
                zend_throw_error(NULL, "Invalid slice parameter at dimension %d", i + 1);
                RETURN_NULL();
            }
        }
        else
        {
            memset(&slice_info[i], 0, sizeof(slice_info_t));
            slice_info[i].type = SLICE_ALL;
        }
    }

    tensor_t *view_tensor = cuda_tensor_create_sliced_view(this_obj->tensor_handle, slice_info, ndim);
    efree(slice_info);

    if (!view_tensor)
    {
        zend_throw_error(NULL, "Failed to create tensor view");
        RETURN_NULL();
    }

    object_init_ex(return_value, cuda_array_ce);
    cuda_array_obj *new_obj = php_cuda_array_fetch_object(Z_OBJ_P(return_value));
    new_obj->tensor_handle = view_tensor;

    new_obj->shape = zend_new_array(view_tensor->ndims);
    for (int i = 0; i < view_tensor->ndims; i++)
    {
        zval dim;
        ZVAL_LONG(&dim, view_tensor->shape[i]);
        zend_hash_index_update(new_obj->shape, i, &dim);
    }
}

static void sync_php_object_shape(cuda_array_obj *obj, tensor_t *tensor)
{
    if (obj->shape)
    {
        zend_array_destroy(obj->shape);
    }

    obj->shape = zend_new_array(tensor->ndims);

    for (int i = 0; i < tensor->ndims; i++)
    {
        zval dim;
        ZVAL_LONG(&dim, tensor->shape[i]);
        zend_hash_index_update(obj->shape, i, &dim);
    }
}

int cuda_array_init(size_t mb)
{
    if (!tensor_mem_init(mb)) { 
        php_error_docref(NULL, E_WARNING, 
                         "Failed to initialize CUDA memory pool with %ld MB.", 
                         mb);
        return 0;
    }

    zend_class_entry *cuda_array_ce = register_cuda_array_class();

    cuda_array_ce->create_object = cuda_array_create_object;

    memcpy(&cuda_array_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    cuda_array_handlers.offset = XtOffsetOf(cuda_array_obj, obj);
    cuda_array_handlers.free_obj = cuda_array_free_object;

    return 1;
}

void cuda_array_shutdown()
{
    tensor_mem_destroy();
}

static int parse_slice_parameter(zval *param, slice_info_t *slice)
{
    memset(slice, 0, sizeof(slice_info_t));

    if (Z_TYPE_P(param) == IS_NULL)
    {
        slice->type = SLICE_ALL;
        return 1;
    }

    if (Z_TYPE_P(param) == IS_LONG)
    {
        slice->type = SLICE_INDEX;
        slice->data.index = Z_LVAL_P(param);
        return 1;
    }

    if (Z_TYPE_P(param) == IS_ARRAY)
    {
        HashTable *ht = Z_ARRVAL_P(param);
        if (zend_array_count(ht) == 2)
        {
            zval *start_val = zend_hash_index_find(ht, 0);
            zval *end_val = zend_hash_index_find(ht, 1);

            if (start_val && end_val &&
                Z_TYPE_P(start_val) == IS_LONG &&
                Z_TYPE_P(end_val) == IS_LONG)
            {

                slice->type = SLICE_RANGE;
                slice->data.range.start = Z_LVAL_P(start_val);
                slice->data.range.end = Z_LVAL_P(end_val);
                return 1;
            }
        }
    }

    return 0;
}

static cuda_array_obj *php_cuda_array_fetch_object(zend_object *obj)
{
    return (cuda_array_obj *)((char *)obj - XtOffsetOf(cuda_array_obj, obj));
}

static cuda_array_obj *php_cuda_array_fetch_valid_object(zend_object *obj)
{
    cuda_array_obj *this_obj = (cuda_array_obj *)((char *)obj - XtOffsetOf(cuda_array_obj, obj));
    if (this_obj->tensor_handle == NULL)
    {
        zend_error(E_ERROR, "Attempting to access uninitialized tensor!");
        return NULL;
    }

    if (this_obj->tensor_handle->is_view && !this_obj->tensor_handle->base_tensor)
    {
        zend_error(E_ERROR, "Attempting to access a view with no base tensor!");
        return NULL;
    }

    return this_obj;
}

static zend_object *cuda_array_create_object(zend_class_entry *class_type)
{
    cuda_array_obj *obj = (cuda_array_obj *)ecalloc(1, sizeof(cuda_array_obj));

    zend_object_std_init(&obj->obj, class_type);
    object_properties_init(&obj->obj, class_type);

    obj->obj.handlers = &cuda_array_handlers;
    obj->tensor_handle = NULL;
    obj->shape = NULL;

    return &obj->obj;
}

static void cuda_array_free_object(zend_object *object)
{
    cuda_array_obj *obj = php_cuda_array_fetch_object(object);

    if (obj->tensor_handle != NULL)
    {
        cuda_tensor_destroy(obj->tensor_handle);
        obj->tensor_handle = NULL;
    }

    if (obj->shape != NULL)
    {
        zend_array_destroy(obj->shape);
        obj->shape = NULL;
    }

    zend_object_std_dtor(&obj->obj);
}

static void create_result_object(zval *return_value, tensor_t *result_tensor)
{
    object_init_ex(return_value, cuda_array_ce);
    cuda_array_obj *result_obj = php_cuda_array_fetch_object(Z_OBJ_P(return_value));

    result_obj->tensor_handle = result_tensor;

    int *result_shape = result_tensor->shape;
    int result_ndims = result_tensor->ndims;

    result_obj->shape = zend_new_array(result_ndims);
    for (int i = 0; i < result_ndims; i++)
    {
        zval dim;
        ZVAL_LONG(&dim, result_shape[i]);
        zend_hash_index_update(result_obj->shape, i, &dim);
    }
}

static tensor_t *get_second_tensor(zval *other_zv, cuda_array_obj *this_obj)
{
    if (Z_TYPE_P(other_zv) == IS_OBJECT && instanceof_function(Z_OBJCE_P(other_zv), cuda_array_ce))
    {
        cuda_array_obj *other_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(other_zv));

        if (other_obj->tensor_handle == NULL)
        {
            zend_throw_error(NULL, "Other tensor not initialized");
            return NULL;
        }

        return other_obj->tensor_handle;
    }
    else if (Z_TYPE_P(other_zv) == IS_DOUBLE || Z_TYPE_P(other_zv) == IS_LONG)
    {
        double scalar_value = (Z_TYPE_P(other_zv) == IS_DOUBLE) ? Z_DVAL_P(other_zv) : (double)Z_LVAL_P(other_zv);
        return cuda_tensor_create_scalar((float)scalar_value, this_obj->tensor_handle->shape, this_obj->tensor_handle->ndims);
    }
    else
    {
        zend_throw_error(NULL, "Parameter must be CudaArray or number");
        return NULL;
    }
}

static void self_operation_handler(INTERNAL_FUNCTION_PARAMETERS,
                                   const char *operation_name,
                                   self_operation_func tensor_func)
{

    cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));

    if (this_obj->tensor_handle == NULL)
    {
        zend_throw_error(NULL, "Tensor not initialized");
        RETURN_NULL();
    }

    tensor_t *result_tensor = tensor_func(this_obj->tensor_handle);

    if (result_tensor == NULL)
    {
        zend_throw_error(NULL, "%s failed", operation_name);
        RETURN_NULL();
    }

    create_result_object(return_value, result_tensor);
}

static void unary_operation_handler(INTERNAL_FUNCTION_PARAMETERS,
                                    const char *operation_name,
                                    int operation_type)
{
    cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));

    if (this_obj->tensor_handle == NULL)
    {
        zend_throw_error(NULL, "Tensor not initialized");
        RETURN_NULL();
    }

    tensor_t *result_tensor = cuda_unary_op(this_obj->tensor_handle, operation_type);

    if (result_tensor == NULL)
    {
        zend_throw_error(NULL, "%s failed", operation_name);
        RETURN_NULL();
    }

    create_result_object(return_value, result_tensor);
}

static void binary_operation_handler(INTERNAL_FUNCTION_PARAMETERS, const char *operation_name, int operation_type)
{
    zval *other_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(other_zv)
    ZEND_PARSE_PARAMETERS_END();

    cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));

    if (this_obj->tensor_handle == NULL)
    {
        zend_throw_error(NULL, "Tensor not initialized");
        RETURN_NULL();
    }

    tensor_t *result_tensor = NULL;

    if (Z_TYPE_P(other_zv) == IS_OBJECT && instanceof_function(Z_OBJCE_P(other_zv), cuda_array_ce))
    {
        cuda_array_obj *other_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(other_zv));

        if (other_obj->tensor_handle == NULL)
        {
            zend_throw_error(NULL, "Other tensor not initialized");
            RETURN_NULL();
        }

        result_tensor = cuda_tensor_op(this_obj->tensor_handle, other_obj->tensor_handle, operation_type);
    }
    else if (Z_TYPE_P(other_zv) == IS_DOUBLE || Z_TYPE_P(other_zv) == IS_LONG)
    {
        float scalar_value = (Z_TYPE_P(other_zv) == IS_DOUBLE) ? (float)Z_DVAL_P(other_zv) : (float)Z_LVAL_P(other_zv);
        result_tensor = cuda_scalar_op(this_obj->tensor_handle, scalar_value, operation_type);
    }
    else
    {
        zend_throw_error(NULL, "Operation requires CudaArray or numeric value");
        RETURN_NULL();
    }

    if (result_tensor == NULL)
    {
        zend_throw_error(NULL, "%s failed", operation_name);
        RETURN_NULL();
    }

    create_result_object(return_value, result_tensor);
}

static void static_tensor_creator(INTERNAL_FUNCTION_PARAMETERS, const char *method_name, float value)
{
    zval *shape_array;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ARRAY(shape_array)
    ZEND_PARSE_PARAMETERS_END();

    int shape[10] = {0};
    int ndims = 0;

    zval *dim;
    int i = 0;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(shape_array), dim)
    {
        if (i < 10 && Z_TYPE_P(dim) == IS_LONG)
        {
            shape[i++] = Z_LVAL_P(dim);
        }
    }
    ZEND_HASH_FOREACH_END();
    ndims = i;

    if (ndims == 0)
    {
        zend_throw_error(NULL, "Invalid shape: must provide dimensions");
        RETURN_NULL();
    }

    tensor_t *tensor = cuda_tensor_create_with_value(shape, ndims, value);
    if (!tensor)
    {
        zend_throw_error(NULL, "Failed to create %s tensor", method_name);
        RETURN_NULL();
    }

    create_result_object(return_value, tensor);
}
