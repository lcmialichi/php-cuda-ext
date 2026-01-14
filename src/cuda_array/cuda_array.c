#include "php.h"
#include "cuda_array.h"
#include "ca_private.h"
#include "ca_arginfo.h"
#include "operations.h"
#include "tensor_fabric.h"
#include "memory_pool.h"
#include "cuda.h"
#include "zend_smart_str.h"
#include "data_types.h"

zend_class_entry *cuda_array_ce;
static zend_object_handlers cuda_array_handlers;

static cuda_array_obj *php_cuda_array_fetch_object(zend_object *obj);
static cuda_array_obj *php_cuda_array_fetch_valid_object(zend_object *obj);
static zend_object *cuda_array_create_object(zend_class_entry *class_type);
static void cuda_array_free_object(zend_object *object);
static void create_result_object(zval *return_value, tensor_t *result_tensor);
static zend_object *cuda_array_clone_obj(zend_object *old_object);
static int parse_slice_parameter(zval *param, slice_info_t *slice);
static zend_result cuda_array_do_operation(zend_uchar opcode, zval *result, zval *op1, zval *op2);
static zval *cuda_array_read_dimension(zend_object *object, zval *offset, int type, zval *rv);
static void cuda_array_write_dimension(zend_object *object, zval *offset, zval *value);
static tensor_t *cuda_tensor_concat(zval *tensors_array, int axis);
static void static_tensor_creator(INTERNAL_FUNCTION_PARAMETERS, const char *method_name, float value);
static void rand_tensor_creator(INTERNAL_FUNCTION_PARAMETERS, unsigned long long seed);

static void reduction_operation_handler(INTERNAL_FUNCTION_PARAMETERS, const char *operation_name, operation_type_t operation_type, int return_arg);
static void unary_operation_handler(INTERNAL_FUNCTION_PARAMETERS, const char *operation_name, operation_type_t operation_type);
static void binary_operation_handler(INTERNAL_FUNCTION_PARAMETERS, const char *operation_name, operation_type_t operation_type);

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

ZEND_METHOD(CudaArray, gt)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Greater", OP_GT);
}

ZEND_METHOD(CudaArray, lt)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Less", OP_LT);
}

ZEND_METHOD(CudaArray, eq)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Equal", OP_EQ);
}

ZEND_METHOD(CudaArray, ne)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "NotEqual", OP_NE);
}

ZEND_METHOD(CudaArray, ge)
{
    binary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "GreaterEqual", OP_GE);
}

ZEND_METHOD(CudaArray, le)
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

ZEND_METHOD(CudaArray, rand)
{
    rand_tensor_creator(INTERNAL_FUNCTION_PARAM_PASSTHRU, 4242424242424242ULL);
}

ZEND_METHOD(CudaArray, transpose)
{
    cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));
    tensor_t *tensor = this_obj->tensor_handle;

    zval *dims_array = NULL;

    ZEND_PARSE_PARAMETERS_START(0, 1)
    Z_PARAM_OPTIONAL
    Z_PARAM_ARRAY_OR_NULL(dims_array)
    ZEND_PARSE_PARAMETERS_END();

    if (dims_array == NULL)
    {
        int default_axis[MAX_DIMS];
        for (int i = 0; i < tensor->ndims; i++)
        {
            default_axis[i] = tensor->ndims - 1 - i;
        }

        tensor_t *result_tensor = cuda_tensor_transpose(tensor, default_axis, tensor->ndims);
        if (result_tensor == NULL)
        {
            zend_throw_error(NULL, "transpose failed");
            RETURN_NULL();
        }
        create_result_object(return_value, result_tensor);
        return;
    }

    int axis[MAX_DIMS] = {0};
    int naxis = tensor->ndims;
    int i = 0;

    for (i = 0; i < naxis; i++)
    {
        axis[i] = i;
    }

    zval *dim;
    int j = 0;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(dims_array), dim)
    {
        if (j >= MAX_DIMS)
        {
            zend_throw_error(NULL, "too many dimensions in transpose argument (max %d)", MAX_DIMS);
            RETURN_NULL();
        }

        if (Z_TYPE_P(dim) != IS_LONG)
        {
            zend_throw_error(NULL, "invalid argument for 'transpose' - expected integer dimensions");
            RETURN_NULL();
        }

        axis[j++] = Z_LVAL_P(dim);
    }
    ZEND_HASH_FOREACH_END();

    if (tensor->ndims != j)
    {
        zend_throw_error(NULL, "transpose expects %d dimensions, got %d", tensor->ndims, j);
        RETURN_NULL();
    }

    bool axis_used[MAX_DIMS] = {false};
    for (i = 0; i < naxis; i++)
    {
        if (axis[i] < 0 || axis[i] >= naxis)
        {
            zend_throw_error(NULL, "invalid axis %d for tensor with %d dimensions", axis[i], naxis);
            RETURN_NULL();
        }
        if (axis_used[axis[i]])
        {
            zend_throw_error(NULL, "duplicate axis %d in transpose", axis[i]);
            RETURN_NULL();
        }
        axis_used[axis[i]] = true;
    }

    tensor_t *result_tensor = cuda_tensor_transpose(tensor, axis, naxis);
    if (result_tensor == NULL)
    {
        zend_throw_error(NULL, "transpose operation failed");
        RETURN_NULL();
    }

    create_result_object(return_value, result_tensor);
}

ZEND_METHOD(CudaArray, matmul)
{
    cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));
    tensor_t *tensor_a = this_obj->tensor_handle;

    zval *other_array = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_OBJECT(other_array)
    ZEND_PARSE_PARAMETERS_END();

    cuda_array_obj *other_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(other_array));
    if (!other_obj)
    {
        zend_throw_error(NULL, "Invalid tensor object for matrix multiplication");
        RETURN_NULL();
    }

    tensor_t *tensor_b = other_obj->tensor_handle;

    if (tensor_a == NULL || tensor_b == NULL)
    {
        zend_throw_error(NULL, "Both operands must be valid tensor objects.");
        RETURN_NULL();
    }

    tensor_t *result_tensor = cuda_tensor_matmul(tensor_a, tensor_b);
    if (result_tensor == NULL)
    {
        zend_throw_error(NULL, "Matrix multiplication failed - incompatible dimensions");
        RETURN_NULL();
    }

    create_result_object(return_value, result_tensor);
}

ZEND_METHOD(CudaArray, sqrt)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Sqrt", OP_SQRT);
}

ZEND_METHOD(CudaArray, floor)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Floor", OP_FLOOR);
}

ZEND_METHOD(CudaArray, ceil)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Ceil", OP_CEIL);
}

ZEND_METHOD(CudaArray, round)
{
    unary_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Round", OP_ROUND);
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

ZEND_METHOD(CudaArray, sum)
{
    reduction_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Sum Reduction", OP_REDUCE_SUM, 0);
}

ZEND_METHOD(CudaArray, mean)
{
    reduction_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Mean Reduction", OP_REDUCE_MEAN, 0);
}

ZEND_METHOD(CudaArray, max)
{
    reduction_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Max Reduction", OP_REDUCE_MAX, 0);
}

ZEND_METHOD(CudaArray, min)
{
    reduction_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Min Reduction", OP_REDUCE_MIN, 0);
}

ZEND_METHOD(CudaArray, prod)
{
    reduction_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "Product Reduction", OP_REDUCE_PROD, 0);
}

ZEND_METHOD(CudaArray, argMax)
{
    reduction_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "ArgMax Reduction", OP_ARG_MAX, 1);
}

ZEND_METHOD(CudaArray, argMin)
{
    reduction_operation_handler(INTERNAL_FUNCTION_PARAM_PASSTHRU, "ArgMin Reduction", OP_ARG_MIN, 1);
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

ZEND_METHOD(CudaArray, getNdims)
{
    cuda_array_obj *obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));
    tensor_t *t = obj->tensor_handle;
    if (!t->ndims)
    {
        RETURN_NULL();
    }

    RETURN_LONG(t->ndims);
}

ZEND_METHOD(CudaArray, getSize)
{
    cuda_array_obj *obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));
    tensor_t *t = obj->tensor_handle;
    if (!t->total_size)
    {
        RETURN_LONG(0);
    }

    RETURN_LONG((int)t->total_size);
}

ZEND_METHOD(CudaArray, concat)
{
    zval *this_ptr = ZEND_THIS;
    zval *input_tensors_array;
    zend_long axis_long = 0;

    ZEND_PARSE_PARAMETERS_START(1, 2)
    Z_PARAM_ARRAY(input_tensors_array)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(axis_long)
    ZEND_PARSE_PARAMETERS_END();

    int axis = (int)axis_long;

    zval full_tensors_list;
    array_init(&full_tensors_list);

    zval temp_zval;
    ZVAL_COPY(&temp_zval, this_ptr);
    zend_hash_next_index_insert(Z_ARRVAL(full_tensors_list), &temp_zval);

    HashTable *input_ht = Z_ARRVAL_P(input_tensors_array);
    zval *pzval;

    ZEND_HASH_FOREACH_VAL(input_ht, pzval)
    {
        zval temp_zval_arg;
        ZVAL_COPY(&temp_zval_arg, pzval);
        zend_hash_next_index_insert(Z_ARRVAL(full_tensors_list), &temp_zval_arg);
    }
    ZEND_HASH_FOREACH_END();

    tensor_t *new_tensor = cuda_tensor_concat(&full_tensors_list, axis);
    zend_array_destroy(Z_ARRVAL(full_tensors_list));

    if (!new_tensor)
    {
        RETURN_THROWS();
    }

    create_result_object(return_value, new_tensor);
}

ZEND_METHOD(CudaArray, toArray)
{
    cuda_array_obj *obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));
    tensor_t *tensor = obj->tensor_handle;

    tensor_t *base = tensor->is_view ? tensor->base_tensor : tensor;
    size_t base_total = base->total_size;

    float *host_data = emalloc(base_total * tensor->element_size);

    cudaError_t status = cudaMemcpy(
        host_data,
        base->data,
        base_total * tensor->element_size,
        cudaMemcpyDeviceToHost);

    if (status != cudaSuccess)
    {
        efree(host_data);
        zend_throw_error(NULL, "Failed to copy data from GPU: %s", cudaGetErrorString(status));
        RETURN_NULL();
    }

    void build_recursive(
        zval * result,
        void *data,
        int dim,
        tensor_t *t,
        size_t current_offset,
        dtype_t dtype)
    {
        array_init(result);

        int size = t->shape[dim];
        size_t stride = t->strides[dim];

        for (int i = 0; i < size; i++)
        {
            size_t child_offset = current_offset + i * stride;

            if (dim == t->ndims - 1)
            {
                zval val;

                if (dtype == DTYPE_FLOAT32)
                {
                    float *float_data = (float *)data;
                    ZVAL_DOUBLE(&val, (double)float_data[child_offset]);
                }
                else if (dtype == DTYPE_FLOAT64)
                {
                    double *double_data = (double *)data;
                    ZVAL_DOUBLE(&val, double_data[child_offset]);
                }
                else if (dtype == DTYPE_INT32)
                {
                    int32_t *int_data = (int32_t *)data;
                    ZVAL_LONG(&val, (zend_long)int_data[child_offset]);
                }
                else if (dtype == DTYPE_INT64)
                {
                    int64_t *int64_data = (int64_t *)data;
                    ZVAL_LONG(&val, (zend_long)int64_data[child_offset]);
                }

                zend_hash_index_update(Z_ARRVAL_P(result), i, &val);
            }
            else
            {
                zval sub;
                build_recursive(&sub, data, dim + 1, t, child_offset, dtype);
                zend_hash_index_update(Z_ARRVAL_P(result), i, &sub);
            }
        }
    }

    size_t offset_elements = tensor->gpu_offset / sizeof(float);
    build_recursive(return_value, host_data, 0, tensor, offset_elements, tensor->dtype);
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

    create_result_object(return_value, view_tensor);
}

ZEND_METHOD(CudaArray, __debugInfo)
{
    cuda_array_obj *obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));
    tensor_t *tensor = obj->tensor_handle;
    array_init(return_value);

    if (!tensor || tensor->ndims <= 0)
    {
        add_assoc_string(return_value, "Error", "CudaArray handle is NULL or has zero dimensions");
        return;
    }

    zval shape_array;
    array_init(&shape_array);

    for (int i = 0; i < tensor->ndims; i++)
    {
        add_next_index_long(&shape_array, (zend_long)tensor->shape[i]);
    }
    add_assoc_zval(return_value, "Shape", &shape_array);

    const char *dtype_str;
    size_t element_size;

    if (tensor->dtype == DTYPE_FLOAT32)
    {
        dtype_str = "float32";
        element_size = sizeof(float);
    }
    else if (tensor->dtype == DTYPE_INT32)
    {
        dtype_str = "int32";
        element_size = sizeof(int);
    }
    else
    {
        dtype_str = "unknown";
        element_size = 0;
    }

    add_assoc_string(return_value, "Dtype", (char *)dtype_str);
    add_assoc_long(return_value, "Elements", (zend_long)tensor->total_size);
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
    if (!tensor_mem_init(mb))
    {
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
    cuda_array_handlers.clone_obj = cuda_array_clone_obj;
    cuda_array_handlers.do_operation = cuda_array_do_operation;
    cuda_array_handlers.read_dimension = cuda_array_read_dimension;
    cuda_array_handlers.write_dimension = cuda_array_write_dimension;
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

    if (!this_obj || this_obj->tensor_handle == NULL)
    {
        zend_error(E_ERROR, "Attempting to access uninitialized tensor!");
        return NULL;
    }

    if (this_obj->shape == NULL)
    {
        zend_error(E_ERROR, "Attempting to access tensor with no shape!");
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

static zend_object *cuda_array_clone_obj(zend_object *old_object)
{
    cuda_array_obj *old_ca = php_cuda_array_fetch_valid_object(old_object);
    zend_object *new_object = cuda_array_create_object(cuda_array_ce);

    cuda_array_obj *new_ca = php_cuda_array_fetch_object(new_object);
    if (!new_ca)
    {
        zend_throw_error(NULL, "Internal error during object cloning: cannot fetch object data.");
        zend_object_release(new_object);
        return NULL;
    }

    zend_objects_clone_members(new_object, old_object);
    tensor_t *new_tensor = cuda_tensor_clone(old_ca->tensor_handle);
    if (new_tensor == NULL)
    {
        zend_throw_error(NULL, "Failed to clone CUDA tensor data during object cloning.");
        zend_object_std_dtor(new_object);
        zend_object_release(new_object);
        return NULL;
    }

    new_ca->tensor_handle = new_tensor;
    sync_php_object_shape(new_ca, new_ca->tensor_handle);

    return new_object;
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

    sync_php_object_shape(result_obj, result_tensor);
}

static void unary_operation_handler(INTERNAL_FUNCTION_PARAMETERS,
                                    const char *operation_name,
                                    operation_type_t operation_type)
{
    cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));

    if (this_obj->tensor_handle == NULL)
    {
        zend_throw_error(NULL, "CudaArray not initialized");
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

static void reduction_operation_handler(INTERNAL_FUNCTION_PARAMETERS, const char *operation_name, operation_type_t operation_type, int return_arg)
{
    zend_long axis_zv = REDUCE_GLOBAL_FLAG;

    ZEND_PARSE_PARAMETERS_START(0, 1)
    Z_PARAM_OPTIONAL
    Z_PARAM_LONG(axis_zv)
    ZEND_PARSE_PARAMETERS_END();

    cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));

    if (this_obj->tensor_handle == NULL)
    {
        zend_throw_error(NULL, "Input tensor not initialized.");
        RETURN_NULL();
    }

    tensor_t *input_tensor = this_obj->tensor_handle;
    int axis = (int)axis_zv;

    if (axis == REDUCE_GLOBAL_FLAG)
    {
        size_t total_size = 1;
        for (int i = 0; i < this_obj->tensor_handle->ndims; i++)
        {
            total_size *= this_obj->tensor_handle->shape[i];
        }

        int flat_shape[] = {(int)total_size};
        input_tensor = cuda_tensor_reshape(this_obj->tensor_handle, flat_shape, 1);
        axis = 0;
    }

    axis = axis >= 0 ? axis : input_tensor->ndims + axis;

    if ((axis < 0 || axis >= input_tensor->ndims) && axis != REDUCE_GLOBAL_FLAG)
    {
        zend_throw_error(NULL, "Axis %d out of bounds for tensor with %d dimensions.", axis, input_tensor->ndims);
        RETURN_NULL();
    }

    tensor_t *result_tensor = (return_arg == 1)
                                  ? cuda_tensor_reduce_arg(input_tensor, axis, operation_type)
                                  : cuda_tensor_reduce(input_tensor, axis, operation_type);

    if (result_tensor == NULL)
    {
        RETURN_NULL();
    }

    create_result_object(return_value, result_tensor);
}

static void binary_operation_handler(INTERNAL_FUNCTION_PARAMETERS, const char *operation_name, operation_type_t operation_type)
{
    zval *other_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ZVAL(other_zv)
    ZEND_PARSE_PARAMETERS_END();

    cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(ZEND_THIS));

    if (this_obj->tensor_handle == NULL)
    {
        zend_throw_error(NULL, "CudaArray not initialized");
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

static zend_result cuda_array_do_operation(zend_uchar opcode, zval *result, zval *op1, zval *op2)
{
    zend_bool define_value = 0;
    float op_value = 0.0f;
    const char *operation_name = NULL;
    int operation_type = 0;

    switch (opcode)
    {
    case ZEND_ADD:
        operation_name = "Addition (+)";
        operation_type = OP_ADD;
        break;
    case ZEND_SUB:
        operation_name = "Subtraction (-)";
        operation_type = OP_SUB;
        break;
    case ZEND_MUL:
        operation_name = "Multiplication (*)";
        operation_type = OP_MUL;
        break;
    case ZEND_DIV:
        operation_name = "Division (/)";
        operation_type = OP_DIV;
        break;
    case ZEND_POW:
        operation_name = "Power (**)";
        operation_type = OP_POW;
        break;
    case ZEND_PRE_INC:
    case ZEND_POST_INC:
        operation_name = "Increment (++)";
        operation_type = OP_ADD;
        break;
    case ZEND_PRE_DEC:
    case ZEND_POST_DEC:
        operation_name = "Decrement (--)";
        operation_type = OP_SUB;
        break;
    default:
        return FAILURE;
    }

    tensor_t *result_tensor = NULL;

    if (Z_TYPE_P(op1) == IS_OBJECT && Z_OBJCE_P(op1) == cuda_array_ce)
    {
        cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(op1));
        if (!this_obj || this_obj->tensor_handle == NULL)
        {
            return FAILURE;
        }

        if (Z_TYPE_P(op2) == IS_OBJECT && instanceof_function(Z_OBJCE_P(op2), cuda_array_ce))
        {
            cuda_array_obj *other_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(op2));
            result_tensor = cuda_tensor_op(this_obj->tensor_handle, other_obj->tensor_handle, operation_type);
        }
        else if (Z_TYPE_P(op2) == IS_DOUBLE || Z_TYPE_P(op2) == IS_LONG)
        {
            float scalar_value = (Z_TYPE_P(op2) == IS_DOUBLE) ? (float)Z_DVAL_P(op2) : (float)Z_LVAL_P(op2);
            result_tensor = cuda_scalar_op(this_obj->tensor_handle, scalar_value, operation_type);
        }
        else
        {
            return FAILURE;
        }
    }
    else if (Z_TYPE_P(op2) == IS_OBJECT && Z_OBJCE_P(op2) == cuda_array_ce)
    {
        cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(op2));
        if (!this_obj || this_obj->tensor_handle == NULL)
        {
            return FAILURE;
        }

        if (Z_TYPE_P(op1) == IS_DOUBLE || Z_TYPE_P(op1) == IS_LONG)
        {
            float scalar_value = (Z_TYPE_P(op1) == IS_DOUBLE) ? (float)Z_DVAL_P(op1) : (float)Z_LVAL_P(op1);
            result_tensor = cuda_inv_scalar_op(this_obj->tensor_handle, scalar_value, operation_type);
        }
        else
        {
            return FAILURE;
        }
    }
    else
    {
        return FAILURE;
    }

    if (result_tensor == NULL)
    {
        zend_throw_error(NULL, "CudaArray operation %s failed (incompatible shapes or internal error)", operation_name);
        return FAILURE;
    }

    create_result_object(result, result_tensor);
    return SUCCESS;
}

static zval *cuda_array_read_dimension(zend_object *object, zval *offset, int type, zval *rv)
{
    cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(object);
    tensor_t *base_tensor = this_obj->tensor_handle;
    int ndim = base_tensor->ndims;

    if (ndim == 0)
    {
        zend_throw_error(NULL, "Cannot slice a zero-dimensional tensor (scalar).");
        return &EG(uninitialized_zval);
    }

    slice_info_t slice_info_array[MAX_DIMS];

    if (!parse_slice_parameter(offset, &slice_info_array[0]))
    {
        zend_throw_error(NULL, "Invalid dimension access: key must be NULL, integer, or [start, end] array.");
        return &EG(uninitialized_zval);
    }

    if (base_tensor->ndims == 1 && slice_info_array[0].type == SLICE_INDEX)
    {
        float result_val;
        if (cuda_tensor_get_scalar_value(base_tensor, &result_val, slice_info_array[0].data.index) != SUCCESS)
        {
            zend_throw_error(NULL, "Failed to extract scalar value from GPU.");
            return &EG(uninitialized_zval);
        }

        ZVAL_DOUBLE(rv, (double)result_val);
        return rv;
    }

    for (int i = 1; i < ndim; i++)
    {
        slice_info_array[i].type = SLICE_ALL;
    }

    tensor_t *view_tensor = cuda_tensor_create_dim_view(
        base_tensor,
        slice_info_array,
        ndim);

    if (view_tensor == NULL)
    {
        zend_throw_error(NULL, "Failed to create tensor view during array access.");
        return &EG(uninitialized_zval);
    }

    create_result_object(rv, view_tensor);
    return rv;
}

static void cuda_array_write_dimension(zend_object *object, zval *offset, zval *value)
{
    cuda_array_obj *this_obj = php_cuda_array_fetch_valid_object(object);
    if (offset == NULL)
    {
        zend_throw_error(NULL, "It is not permitted to append (operator []) to a CudaArray. ");
        return;
    }

    slice_info_t slice_info;
    if (!parse_slice_parameter(offset, &slice_info))
    {
        zend_throw_error(NULL, "Invalid tensor index parameter.");
        return;
    }

    tensor_t *base_tensor = this_obj->tensor_handle;

    if (slice_info.type == SLICE_INDEX)
    {
        int index = slice_info.data.index;
        size_t element_offset = (size_t)index * base_tensor->strides[0];

        if (base_tensor->ndims == 0 || index < 0 || index >= base_tensor->shape[0])
        {
            zend_throw_error(NULL, "Index out of bounds for write operation (Dim 0).");
            return;
        }

        if (Z_TYPE_P(value) == IS_DOUBLE || Z_TYPE_P(value) == IS_LONG)
        {
            float scalar_value = (Z_TYPE_P(value) == IS_DOUBLE) ? (float)Z_DVAL_P(value) : (float)Z_LVAL_P(value);
            if (cuda_tensor_set_scalar(base_tensor, element_offset, scalar_value) != SUCCESS)
            {
                zend_throw_error(NULL, "Failed to write scalar value to GPU memory.");
            }
        }
        else if (Z_TYPE_P(value) == IS_OBJECT && instanceof_function(Z_OBJCE_P(value), cuda_array_ce))
        {
            cuda_array_obj *src_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(value));
            tensor_t *src_tensor = src_obj->tensor_handle;
            int dest_ndims = base_tensor->ndims - 1;

            if (src_tensor->ndims != dest_ndims)
            {
                zend_throw_error(NULL, "CudaArray assignment requires a source with %d dimensions, but %d given.", dest_ndims, src_tensor->ndims);
                return;
            }
            for (int i = 0; i < dest_ndims; i++)
            {
                if (src_tensor->shape[i] != base_tensor->shape[i + 1])
                {
                    zend_throw_error(NULL, "Shape mismatch for tensor assignment at dimension %d.", i + 1);
                    return;
                }
            }

            if (cuda_tensor_set_tensor(base_tensor, element_offset, src_tensor) != SUCCESS)
            {
                zend_throw_error(NULL, "Failed GPU memory copy during tensor assignment");
            }
        }
        else
        {
            zend_throw_error(NULL, "Only scalar or CudaArray assignment is supported for single index.");
        }
    }
    else if (slice_info.type == SLICE_RANGE)
    {
        zend_throw_error(NULL, "SLICE_RANGE not implemented yet.");
    }
    else
    {
        zend_throw_error(NULL, "Only single index, array index list, or range slice assignment is supported in this context for now.");
    }
}
static void rand_tensor_creator(INTERNAL_FUNCTION_PARAMETERS, unsigned long long seed)
{
    zval *shape_array;
    double min = 0;
    double max = 100;

    ZEND_PARSE_PARAMETERS_START_EX(ZPP_ERROR_FAILURE, 1, 3)
    Z_PARAM_ARRAY(shape_array)
    Z_PARAM_OPTIONAL
    Z_PARAM_DOUBLE(min)
    Z_PARAM_DOUBLE(max)
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

    tensor_t *tensor = cuda_tensor_create_rand(shape, ndims, (float)min, (float)max, seed);

    if (!tensor)
    {
        zend_throw_error(NULL, "Failed to create random tensor");
        RETURN_NULL();
    }

    create_result_object(return_value, tensor);
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

static tensor_t *cuda_tensor_concat(zval *tensors_array, int axis)
{
    HashTable *ht = Z_ARRVAL_P(tensors_array);
    zval *pzval;
    int i = 0;

    int list_count = zend_hash_num_elements(ht);

    if (list_count == 0)
    {
        zend_throw_error(NULL, "Concat requires at least one tensor.");
        return NULL;
    }

    if (list_count > MAX_CONCAT_TENSORS)
    {
        zend_throw_error(NULL, "Too many tensors to concatenate. Maximum is %d.", MAX_CONCAT_TENSORS);
        return NULL;
    }

    tensor_t **tensor_list = (tensor_t **)emalloc(sizeof(tensor_t *) * list_count);
    size_t total_length_on_axis = 0;
    int first_ndims = -1;

    ZEND_HASH_FOREACH_VAL(ht, pzval)
    {
        if (Z_TYPE_P(pzval) != IS_OBJECT || !instanceof_function(Z_OBJCE_P(pzval), cuda_array_ce))
        {
            zend_throw_error(NULL, "All elements must be CudaArray objects.");
            efree(tensor_list);
            return NULL;
        }

        cuda_array_obj *other_obj = php_cuda_array_fetch_valid_object(Z_OBJ_P(pzval));
        tensor_t *current_tensor = other_obj->tensor_handle;

        tensor_list[i] = current_tensor;

        if (i == 0)
        {
            first_ndims = current_tensor->ndims;
            if (axis < 0 || axis >= first_ndims)
            {
                zend_throw_error(NULL, "Axis %d is out of bounds for the first tensor (dims: %d).", axis, first_ndims);
                efree(tensor_list);
                return NULL;
            }
        }
        else
        {
            if (current_tensor->ndims != first_ndims)
            {
                zend_throw_error(NULL, "All tensors must have the same number of dimensions (%d != %d).",
                                 current_tensor->ndims, first_ndims);
                efree(tensor_list);
                return NULL;
            }
            for (int d = 0; d < first_ndims; d++)
            {
                if (d != axis && current_tensor->shape[d] != tensor_list[0]->shape[d])
                {
                    zend_throw_error(NULL, "Shapes must match along non-concatenated axis %d.", d);
                    efree(tensor_list);
                    return NULL;
                }
            }
        }

        total_length_on_axis += current_tensor->shape[axis];
        i++;
    }
    ZEND_HASH_FOREACH_END();

    int *new_shape = (int *)emalloc(sizeof(int) * first_ndims);
    memcpy(new_shape, tensor_list[0]->shape, sizeof(int) * first_ndims);
    new_shape[axis] = (int)total_length_on_axis;
    tensor_t *new_tensor = cuda_tensor_create_empty(new_shape, first_ndims);

    efree(new_shape);

    if (!new_tensor)
    {
        zend_throw_error(NULL, "Failed to allocate memory for concatenated tensor.");
        efree(tensor_list);
        return NULL;
    }

    void *input_ptrs[MAX_CONCAT_TENSORS];
    int input_axis_sizes[MAX_CONCAT_TENSORS];
    size_t input_strides_axis[MAX_CONCAT_TENSORS];
    size_t input_axis_offsets[MAX_CONCAT_TENSORS];

    size_t current_offset = 0;
    size_t output_stride_axis = 1;

    for (int d = axis + 1; d < new_tensor->ndims; d++)
    {
        output_stride_axis *= new_tensor->shape[d];
    }
    for (i = 0; i < list_count; i++)
    {
        tensor_t *current = tensor_list[i];
        input_ptrs[i] = current->data;
        input_axis_sizes[i] = current->shape[axis];
        input_axis_offsets[i] = current_offset;

        size_t input_stride_axis = 1;
        for (int d = axis + 1; d < current->ndims; d++)
        {
            input_stride_axis *= current->shape[d];
        }
        input_strides_axis[i] = input_stride_axis;
        current_offset += input_axis_sizes[i];
    }

    size_t outer_dims = 1;
    for (int d = 0; d < axis; d++)
    {
        outer_dims *= new_tensor->shape[d];
    }

    size_t inner_dims = 1;
    for (int d = axis + 1; d < new_tensor->ndims; d++)
    {
        inner_dims *= new_tensor->shape[d];
    }

    int result = launch_concat_kernel_host(
        tensor_list,
        list_count,
        new_tensor,
        axis,
        input_axis_offsets,
        input_strides_axis,
        output_stride_axis,
        outer_dims,
        inner_dims,
        (int)total_length_on_axis);

    efree(tensor_list);

    if (result != SUCCESS)
    {
        zend_throw_error(NULL, "CUDA concat kernel failed.");
        return NULL;
    }

    return new_tensor;
}