#include "php.h"
#include "host_array_ce.h"
#include "host_array_arginfo.h"
#include <string.h>
#include <math.h>

static zend_object_handlers host_array_handlers;
static zend_object_handlers host_array_iterator_handlers;
zend_class_entry *host_array_ce;
zend_class_entry *host_array_iterator_ce;

static void host_array_offset_set(zend_object *object, zval *offset, zval *value);
static void host_array_offset_unset(zend_object *object, zval *offset);
static int host_array_count_elements(zend_object *object, zend_long *count);

static host_array_object *host_array_from_obj(zend_object *zobj)
{
    return (host_array_object *)((char *)zobj - host_array_handlers.offset);
}

static host_array_iterator_object *host_array_iterator_from_obj(zend_object *zobj)
{
    return (host_array_iterator_object *)((char *)zobj - host_array_iterator_handlers.offset);
}

size_t dtype_to_size(dtype_t dtype)
{
    switch (dtype)
    {
    case DTYPE_FLOAT32:
        return sizeof(float);
    case DTYPE_FLOAT64:
        return sizeof(double);
    case DTYPE_INT8:
        return sizeof(int8_t);
    case DTYPE_INT16:
        return sizeof(int16_t);
    case DTYPE_INT32:
        return sizeof(int32_t);
    case DTYPE_INT64:
        return sizeof(int64_t);
    case DTYPE_UINT8:
        return sizeof(uint8_t);
    case DTYPE_UINT16:
        return sizeof(uint16_t);
    case DTYPE_UINT32:
        return sizeof(uint32_t);
    case DTYPE_UINT64:
        return sizeof(uint64_t);
    case DTYPE_BOOL:
        return sizeof(uint8_t);
    default:
        return sizeof(double);
    }
}

const char *dtype_to_string(dtype_t dtype)
{
    switch (dtype)
    {
    case DTYPE_FLOAT32:
        return "float32";
    case DTYPE_FLOAT64:
        return "float64";
    case DTYPE_INT8:
        return "int8";
    case DTYPE_INT16:
        return "int16";
    case DTYPE_INT32:
        return "int32";
    case DTYPE_INT64:
        return "int64";
    case DTYPE_UINT8:
        return "uint8";
    case DTYPE_UINT16:
        return "uint16";
    case DTYPE_UINT32:
        return "uint32";
    case DTYPE_UINT64:
        return "uint64";
    case DTYPE_BOOL:
        return "bool";
    default:
        return "unknown";
    }
}

// this function should be elsewhere
void *allocate_for_dtype(dtype_t dtype, size_t count)
{
    size_t size = dtype_to_size(dtype);
    void *ptr = ecalloc(count, size);
    return ptr;
}

void set_value_for_dtype(void *data, size_t index, zval *value, dtype_t dtype)
{
    size_t size = dtype_to_size(dtype);
    void *ptr = (char *)data + index * size;

    switch (dtype)
    {
    case DTYPE_FLOAT32:
    {
        float *fptr = (float *)ptr;
        if (Z_TYPE_P(value) == IS_DOUBLE)
        {
            *fptr = (float)Z_DVAL_P(value);
        }
        else if (Z_TYPE_P(value) == IS_LONG)
        {
            *fptr = (float)Z_LVAL_P(value);
        }
        break;
    }
    case DTYPE_FLOAT64:
    {
        double *dptr = (double *)ptr;
        if (Z_TYPE_P(value) == IS_DOUBLE)
        {
            *dptr = Z_DVAL_P(value);
        }
        else if (Z_TYPE_P(value) == IS_LONG)
        {
            *dptr = (double)Z_LVAL_P(value);
        }
        break;
    }
    case DTYPE_INT8:
    {
        int8_t *iptr = (int8_t *)ptr;
        if (Z_TYPE_P(value) == IS_LONG)
        {
            *iptr = (int8_t)Z_LVAL_P(value);
        }
        else if (Z_TYPE_P(value) == IS_DOUBLE)
        {
            *iptr = (int8_t)Z_DVAL_P(value);
        }
        break;
    }
    case DTYPE_INT16:
    {
        int16_t *iptr = (int16_t *)ptr;
        if (Z_TYPE_P(value) == IS_LONG)
        {
            *iptr = (int16_t)Z_LVAL_P(value);
        }
        else if (Z_TYPE_P(value) == IS_DOUBLE)
        {
            *iptr = (int16_t)Z_DVAL_P(value);
        }
        break;
    }
    case DTYPE_INT32:
    {
        int32_t *iptr = (int32_t *)ptr;
        if (Z_TYPE_P(value) == IS_LONG)
        {
            *iptr = (int32_t)Z_LVAL_P(value);
        }
        else if (Z_TYPE_P(value) == IS_DOUBLE)
        {
            *iptr = (int32_t)Z_DVAL_P(value);
        }
        break;
    }
    case DTYPE_INT64:
    {
        int64_t *iptr = (int64_t *)ptr;
        if (Z_TYPE_P(value) == IS_LONG)
        {
            *iptr = (int64_t)Z_LVAL_P(value);
        }
        else if (Z_TYPE_P(value) == IS_DOUBLE)
        {
            *iptr = (int64_t)Z_DVAL_P(value);
        }
        break;
    }
    case DTYPE_UINT8:
    {
        uint8_t *uptr = (uint8_t *)ptr;
        if (Z_TYPE_P(value) == IS_LONG)
        {
            *uptr = (uint8_t)Z_LVAL_P(value);
        }
        else if (Z_TYPE_P(value) == IS_DOUBLE)
        {
            *uptr = (uint8_t)Z_DVAL_P(value);
        }
        break;
    }
    case DTYPE_UINT16:
    {
        uint16_t *uptr = (uint16_t *)ptr;
        if (Z_TYPE_P(value) == IS_LONG)
        {
            *uptr = (uint16_t)Z_LVAL_P(value);
        }
        else if (Z_TYPE_P(value) == IS_DOUBLE)
        {
            *uptr = (uint16_t)Z_DVAL_P(value);
        }
        break;
    }
    case DTYPE_UINT32:
    {
        uint32_t *uptr = (uint32_t *)ptr;
        if (Z_TYPE_P(value) == IS_LONG)
        {
            *uptr = (uint32_t)Z_LVAL_P(value);
        }
        else if (Z_TYPE_P(value) == IS_DOUBLE)
        {
            *uptr = (uint32_t)Z_DVAL_P(value);
        }
        break;
    }
    case DTYPE_UINT64:
    {
        uint64_t *uptr = (uint64_t *)ptr;
        if (Z_TYPE_P(value) == IS_LONG)
        {
            *uptr = (uint64_t)Z_LVAL_P(value);
        }
        else if (Z_TYPE_P(value) == IS_DOUBLE)
        {
            *uptr = (uint64_t)Z_DVAL_P(value);
        }
        break;
    }
    case DTYPE_BOOL:
    {
        uint8_t *bptr = (uint8_t *)ptr;
        if (Z_TYPE_P(value) == IS_TRUE)
        {
            *bptr = 1;
        }
        else if (Z_TYPE_P(value) == IS_FALSE)
        {
            *bptr = 0;
        }
        else if (Z_TYPE_P(value) == IS_LONG)
        {
            *bptr = Z_LVAL_P(value) != 0;
        }
        else if (Z_TYPE_P(value) == IS_DOUBLE)
        {
            *bptr = Z_DVAL_P(value) != 0.0;
        }
        break;
    }
    default:
        break;
    }
}

void get_value_for_dtype(void *data, size_t index, zval *return_value, dtype_t dtype)
{
    size_t size = dtype_to_size(dtype);
    void *ptr = (char *)data + index * size;

    switch (dtype)
    {
    case DTYPE_FLOAT32:
    {
        float *fptr = (float *)ptr;
        ZVAL_DOUBLE(return_value, (double)*fptr);
        break;
    }
    case DTYPE_FLOAT64:
    {
        double *dptr = (double *)ptr;
        ZVAL_DOUBLE(return_value, *dptr);
        break;
    }
    case DTYPE_INT8:
    {
        int8_t *iptr = (int8_t *)ptr;
        ZVAL_LONG(return_value, (zend_long)*iptr);
        break;
    }
    case DTYPE_INT16:
    {
        int16_t *iptr = (int16_t *)ptr;
        ZVAL_LONG(return_value, (zend_long)*iptr);
        break;
    }
    case DTYPE_INT32:
    {
        int32_t *iptr = (int32_t *)ptr;
        ZVAL_LONG(return_value, (zend_long)*iptr);
        break;
    }
    case DTYPE_INT64:
    {
        int64_t *iptr = (int64_t *)ptr;
        ZVAL_LONG(return_value, (zend_long)*iptr);
        break;
    }
    case DTYPE_UINT8:
    {
        uint8_t *uptr = (uint8_t *)ptr;
        ZVAL_LONG(return_value, (zend_long)*uptr);
        break;
    }
    case DTYPE_UINT16:
    {
        uint16_t *uptr = (uint16_t *)ptr;
        ZVAL_LONG(return_value, (zend_long)*uptr);
        break;
    }
    case DTYPE_UINT32:
    {
        uint32_t *uptr = (uint32_t *)ptr;
        ZVAL_LONG(return_value, (zend_long)*uptr);
        break;
    }
    case DTYPE_UINT64:
    {
        uint64_t *uptr = (uint64_t *)ptr;
        ZVAL_LONG(return_value, (zend_long)*uptr);
        break;
    }
    case DTYPE_BOOL:
    {
        uint8_t *bptr = (uint8_t *)ptr;
        ZVAL_BOOL(return_value, *bptr != 0);
        break;
    }
    default:
    {
        ZVAL_NULL(return_value);
        break;
    }
    }
}

static size_t calculate_linear_offset(tensor_t *tensor, size_t *indices)
{
    size_t offset = tensor->offset;
    for (int i = 0; i < tensor->ndims; i++)
    {
        offset += indices[i] * tensor->strides[i];
    }
    return offset;
}

static zend_array *build_array_recursive(tensor_t *tensor, int dim, size_t *indices)
{
    zend_array *arr = NULL;

    if (dim == tensor->ndims - 1)
    {
        arr = zend_new_array(tensor->shape[dim]);
        for (int i = 0; i < tensor->shape[dim]; i++)
        {
            indices[dim] = i;
            size_t offset = calculate_linear_offset(tensor, indices);

            zval val;
            get_value_for_dtype(tensor->data, offset, &val, tensor->dtype);
            zend_hash_index_update(arr, i, &val);
        }
    }
    else
    {
        arr = zend_new_array(tensor->shape[dim]);
        for (int i = 0; i < tensor->shape[dim]; i++)
        {
            indices[dim] = i;
            zend_array *subarr = build_array_recursive(tensor, dim + 1, indices);
            if (subarr)
            {
                zval val;
                ZVAL_ARR(&val, subarr);
                zend_hash_index_update(arr, i, &val);
            }
        }
    }
    return arr;
}

ZEND_METHOD(HostArray, __construct)
{
    zend_throw_error(NULL, "HostArray cannot be constructed directly. Use CudaArray::toHost()");
    RETURN_NULL();
}

ZEND_METHOD(HostArray, get)
{
    zend_object *zobj = Z_OBJ_P(getThis());
    host_array_object *obj = host_array_from_obj(zobj);

    if (!obj || !obj->tensor)
    {
        zend_throw_error(NULL, "HostArray is not properly initialized");
        RETURN_NULL();
    }

    tensor_t *tensor = obj->tensor;
    zval *index_array;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ARRAY(index_array)
    ZEND_PARSE_PARAMETERS_END();

    if (!tensor->data || tensor->ndims == 0)
    {
        zend_throw_error(NULL, "Array not initialized or empty");
        return;
    }

    size_t indices[tensor->ndims];
    zend_long idx_count = zend_array_count(Z_ARRVAL_P(index_array));

    if (idx_count != tensor->ndims)
    {
        zend_throw_error(NULL, "Incorrect number of indices: expected %d, got %ld",
                         tensor->ndims, idx_count);
        return;
    }

    int i = 0;
    zval *val;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(index_array), val)
    {
        zend_long idx = zval_get_long(val);
        if (idx < 0 || idx >= tensor->shape[i])
        {
            zend_throw_error(NULL, "Index %ld out of bounds for dimension %d (size %d)",
                             idx, i, tensor->shape[i]);
            return;
        }
        indices[i] = (size_t)idx;
        i++;
    }
    ZEND_HASH_FOREACH_END();

    size_t offset = calculate_linear_offset(tensor, indices);
    get_value_for_dtype(tensor->data, offset, return_value, tensor->dtype);
}

ZEND_METHOD(HostArray, getShape)
{
    zend_object *zobj = Z_OBJ_P(getThis());
    host_array_object *obj = host_array_from_obj(zobj);

    if (!obj || !obj->tensor)
    {
        zend_throw_error(NULL, "HostArray is not properly initialized");
        RETURN_NULL();
    }

    tensor_t *tensor = obj->tensor;
    array_init(return_value);

    if (!tensor || tensor->ndims == 0)
    {
        return;
    }

    for (int i = 0; i < tensor->ndims; i++)
    {
        add_next_index_long(return_value, tensor->shape[i]);
    }
}

ZEND_METHOD(HostArray, toArray)
{
    zend_object *zobj = Z_OBJ_P(getThis());
    host_array_object *obj = host_array_from_obj(zobj);

    if (!obj || !obj->tensor)
    {
        zend_throw_error(NULL, "HostArray is not properly initialized");
        RETURN_NULL();
    }

    tensor_t *tensor = obj->tensor;

    if (!tensor->data || tensor->ndims == 0)
    {
        RETURN_NULL();
    }

    size_t indices[tensor->ndims];
    memset(indices, 0, sizeof(size_t) * tensor->ndims);
    zend_array *result = build_array_recursive(tensor, 0, indices);

    if (result)
    {
        ZVAL_ARR(return_value, result);
    }
    else
    {
        RETURN_NULL();
    }
}

ZEND_METHOD(HostArray, getNdims)
{
    zend_object *zobj = Z_OBJ_P(getThis());
    host_array_object *obj = host_array_from_obj(zobj);

    if (!obj || !obj->tensor)
    {
        RETURN_LONG(0);
    }

    RETURN_LONG(obj->tensor->ndims);
}

ZEND_METHOD(HostArray, getSize)
{
    zend_object *zobj = Z_OBJ_P(getThis());
    host_array_object *obj = host_array_from_obj(zobj);

    if (!obj || !obj->tensor)
    {
        RETURN_LONG(0);
    }

    RETURN_LONG(obj->tensor->total_size);
}

ZEND_METHOD(HostArray, getDtype)
{
    zend_object *zobj = Z_OBJ_P(getThis());
    host_array_object *obj = host_array_from_obj(zobj);

    if (!obj || !obj->tensor)
    {
        RETURN_STRING("unknown");
    }

    const char *dtype_str = dtype_to_string(obj->tensor->dtype);
    RETURN_STRING(dtype_str);
}

ZEND_METHOD(HostArray, getElementSize)
{
    zend_object *zobj = Z_OBJ_P(getThis());
    host_array_object *obj = host_array_from_obj(zobj);

    if (!obj || !obj->tensor)
    {
        RETURN_LONG(0);
    }

    RETURN_LONG(dtype_to_size(obj->tensor->dtype));
}

ZEND_METHOD(HostArray, count)
{
    zend_object *zobj = Z_OBJ_P(getThis());
    host_array_object *obj = host_array_from_obj(zobj);

    if (!obj || !obj->tensor || obj->tensor->ndims == 0)
    {
        RETURN_LONG(0);
    }

    RETURN_LONG(obj->tensor->shape[0]);
}

ZEND_METHOD(HostArray, getIterator)
{
    host_array_object *obj = host_array_from_obj(Z_OBJ_P(getThis()));

    if (!obj || !obj->tensor)
    {
        zend_throw_error(NULL, "HostArray is not properly initialized");
        RETURN_NULL();
    }

    zend_object *iterator_obj = zend_objects_new(host_array_iterator_ce);
    host_array_iterator_object *iterator = host_array_iterator_from_obj(iterator_obj);

    zend_object *host_copy_obj = host_array_create_object(host_array_ce);
    host_array_object *host_copy = host_array_from_obj(host_copy_obj);

    tensor_t *tensor_copy = (tensor_t *)emalloc(sizeof(tensor_t));
    if (!tensor_copy) {
        zend_object_std_dtor(host_copy_obj);
        zend_object_std_dtor(iterator_obj);
        zend_throw_error(NULL, "Failed to allocate tensor copy");
        RETURN_NULL();
    }
    memset(tensor_copy, 0, sizeof(tensor_t));

    tensor_t *original = obj->tensor;
    tensor_copy->dtype = original->dtype;
    tensor_copy->element_size = original->element_size;
    tensor_copy->is_view = original->is_view;
    tensor_copy->is_on_gpu = original->is_on_gpu;
    tensor_copy->offset = original->offset;

    tensor_copy->ndims = original->ndims;
    if (original->ndims > 0) {
        tensor_copy->shape = (int *)emalloc(sizeof(int) * original->ndims);
        if (!tensor_copy->shape) {
            efree(tensor_copy);
            zend_object_std_dtor(host_copy_obj);
            zend_object_std_dtor(iterator_obj);
            zend_throw_error(NULL, "Failed to allocate shape copy");
            RETURN_NULL();
        }
        memcpy(tensor_copy->shape, original->shape, sizeof(int) * original->ndims);

        tensor_copy->strides = (size_t *)emalloc(sizeof(size_t) * original->ndims);
        if (!tensor_copy->strides) {
            efree(tensor_copy->shape);
            efree(tensor_copy);
            zend_object_std_dtor(host_copy_obj);
            zend_object_std_dtor(iterator_obj);
            zend_throw_error(NULL, "Failed to allocate strides copy");
            RETURN_NULL();
        }
        memcpy(tensor_copy->strides, original->strides, sizeof(size_t) * original->ndims);
    } else {
        tensor_copy->shape = NULL;
        tensor_copy->strides = NULL;
    }

    tensor_copy->total_size = original->total_size;
    tensor_copy->allocated_size = original->allocated_size;
    
    if (original->data && tensor_copy->total_size > 0) {
        tensor_copy->data = allocate_for_dtype(tensor_copy->dtype, tensor_copy->total_size);
        if (!tensor_copy->data) {
            if (tensor_copy->shape) efree(tensor_copy->shape);
            if (tensor_copy->strides) efree(tensor_copy->strides);
            efree(tensor_copy);
            zend_object_std_dtor(host_copy_obj);
            zend_object_std_dtor(iterator_obj);
            zend_throw_error(NULL, "Failed to allocate data copy");
            RETURN_NULL();
        }
        
        size_t total_bytes = tensor_copy->total_size * tensor_copy->element_size;
        memcpy(tensor_copy->data, original->data, total_bytes);
    } else {
        tensor_copy->data = NULL;
    }

    tensor_copy->base_tensor = original->is_view ? original->base_tensor : original;
    tensor_copy->ref_count = 1;
    
    if (original->is_view && original->base_tensor) {
        original->base_tensor->ref_count++;
    }

    host_copy->tensor = tensor_copy;
    host_copy->is_view = 0;
    host_copy->read_only = obj->read_only;

    ZVAL_OBJ(&iterator->host_array_zval, host_copy_obj);
    
    iterator->current_idx = 0;
    iterator->max_idx = tensor_copy->ndims > 0 ? tensor_copy->shape[0] : 0;

    ZVAL_OBJ(return_value, iterator_obj);
}

zend_object *host_array_create_object(zend_class_entry *ce)
{
    host_array_object *obj = (host_array_object *)ecalloc(1, sizeof(host_array_object));

    zend_object_std_init(&obj->std, ce);
    object_properties_init(&obj->std, ce);

    obj->tensor = NULL;
    obj->is_view = 0;
    obj->read_only = 1;

    obj->std.handlers = &host_array_handlers;

    return &obj->std;
}

void host_array_iterator_free_object(zend_object *object)
{
    host_array_iterator_object *iterator = host_array_iterator_from_obj(object);

    if (Z_TYPE(iterator->host_array_zval) != IS_UNDEF)
    {
        host_array_object *host_copy = host_array_from_obj(Z_OBJ(iterator->host_array_zval));
        
        if (host_copy && host_copy->tensor) {
            cuda_tensor_destroy(host_copy->tensor);
        }
        
        zval_ptr_dtor(&iterator->host_array_zval);
        ZVAL_UNDEF(&iterator->host_array_zval);
    }

    zend_object_std_dtor(&iterator->std);
}

static int host_array_offset_exists(zend_object *object, zval *offset, int type)
{
    host_array_object *obj = host_array_from_obj(object);

    if (!obj || !obj->tensor || obj->tensor->ndims == 0 || !obj->tensor->shape)
    {
        return 0;
    }

    zend_long idx = zval_get_long(offset);
    return (idx >= 0 && idx < obj->tensor->shape[0]);
}

static zval *host_array_offset_get(zend_object *object, zval *offset, int type, zval *rv)
{
    host_array_object *obj = host_array_from_obj(object);

    if (!obj || !obj->tensor || obj->tensor->ndims == 0)
    {
        ZVAL_NULL(rv);
        return rv;
    }

    tensor_t *tensor = obj->tensor;
    zend_long idx = zval_get_long(offset);

    if (idx < 0 || idx >= tensor->shape[0])
    {
        ZVAL_NULL(rv);
        return rv;
    }

    if (tensor->ndims == 1)
    {
        size_t offset_idx = tensor->offset + idx * tensor->strides[0];
        get_value_for_dtype(tensor->data, offset_idx, rv, tensor->dtype);
        return rv;
    }
    else
    {
        zend_object *zobj = host_array_create_object(host_array_ce);
        host_array_object *slice = host_array_from_obj(zobj);

        tensor_t *slice_tensor = (tensor_t *)emalloc(sizeof(tensor_t));
        memset(slice_tensor, 0, sizeof(tensor_t));

        slice_tensor->dtype = tensor->dtype;
        slice_tensor->element_size = tensor->element_size;
        slice_tensor->is_view = 1;
        slice_tensor->is_on_gpu = 0;

        slice_tensor->ndims = tensor->ndims - 1;
        slice_tensor->shape = (int *)emalloc(sizeof(int) * slice_tensor->ndims);
        slice_tensor->strides = (size_t *)emalloc(sizeof(size_t) * slice_tensor->ndims);

        for (int i = 0; i < slice_tensor->ndims; i++)
        {
            slice_tensor->shape[i] = tensor->shape[i + 1];
            slice_tensor->strides[i] = tensor->strides[i + 1];
        }

        slice_tensor->offset = tensor->offset + idx * tensor->strides[0];
        slice_tensor->data = tensor->data;

        slice_tensor->total_size = 1;
        for (int i = 0; i < slice_tensor->ndims; i++)
        {
            slice_tensor->total_size *= slice_tensor->shape[i];
        }

        slice_tensor->allocated_size = tensor->allocated_size;

        slice_tensor->base_tensor = tensor->is_view ? tensor->base_tensor : tensor;
        slice_tensor->ref_count = 1;

        tensor_t *base_tensor = tensor->is_view ? tensor->base_tensor : tensor;
        if (base_tensor)
        {
            base_tensor->ref_count++;
        }

        slice->tensor = slice_tensor;
        slice->is_view = 1;
        slice->read_only = 1;

        ZVAL_OBJ(rv, zobj);
        return rv;
    }
}

static void host_array_offset_set(zend_object *object, zval *offset, zval *value)
{
    zend_throw_error(NULL, "HostArray is read-only");
}

static void host_array_offset_unset(zend_object *object, zval *offset)
{
    zend_throw_error(NULL, "HostArray elements cannot be unset");
}

ZEND_METHOD(HostArrayIterator, rewind)
{
    host_array_iterator_object *iterator = host_array_iterator_from_obj(Z_OBJ_P(getThis()));
    iterator->current_idx = 0;
}

ZEND_METHOD(HostArrayIterator, valid)
{
    host_array_iterator_object *iterator = host_array_iterator_from_obj(Z_OBJ_P(getThis()));

    if (Z_TYPE(iterator->host_array_zval) == IS_UNDEF)
    {
        RETURN_BOOL(0);
    }

    host_array_object *obj = host_array_from_obj(Z_OBJ(iterator->host_array_zval));

    if (!obj || !obj->tensor)
    {
        RETURN_BOOL(0);
    }

    RETURN_BOOL(iterator->current_idx >= 0 && iterator->current_idx < iterator->max_idx);
}

ZEND_METHOD(HostArrayIterator, key)
{
    host_array_iterator_object *iterator = host_array_iterator_from_obj(Z_OBJ_P(getThis()));
    RETURN_LONG(iterator->current_idx);
}

ZEND_METHOD(HostArrayIterator, next)
{
    host_array_iterator_object *iterator = host_array_iterator_from_obj(Z_OBJ_P(getThis()));
    iterator->current_idx++;
}

ZEND_METHOD(HostArrayIterator, current)
{
    host_array_iterator_object *iterator = host_array_iterator_from_obj(Z_OBJ_P(getThis()));

    if (Z_TYPE(iterator->host_array_zval) == IS_UNDEF)
    {
        RETURN_NULL();
    }

    host_array_object *obj = host_array_from_obj(Z_OBJ(iterator->host_array_zval));

    if (!obj || !obj->tensor)
    {
        RETURN_NULL();
    }

    if (iterator->current_idx < 0 || iterator->current_idx >= iterator->max_idx)
    {
        RETURN_NULL();
    }

    zval index;
    ZVAL_LONG(&index, iterator->current_idx);

    host_array_offset_get(&obj->std, &index, BP_VAR_R, return_value);
}

ZEND_METHOD(HostArrayIterator, __construct)
{
    zend_throw_error(NULL, "HostArrayIterator cannot be constructed directly");
    RETURN_NULL();
}

zend_object *host_array_iterator_create_object(zend_class_entry *ce)
{
    host_array_iterator_object *iterator = (host_array_iterator_object *)ecalloc(1,
                                                                                 sizeof(host_array_iterator_object));

    zend_object_std_init(&iterator->std, ce);
    object_properties_init(&iterator->std, ce);

    ZVAL_UNDEF(&iterator->host_array_zval);
    iterator->current_idx = 0;
    iterator->max_idx = 0;

    iterator->std.handlers = &host_array_iterator_handlers;

    return &iterator->std;
}

void host_array_iterator_free_object(zend_object *object)
{
    host_array_iterator_object *iterator = host_array_iterator_from_obj(object);
    zend_object_std_dtor(&iterator->std);
}

static int host_array_count_elements(zend_object *object, zend_long *count)
{
    host_array_object *obj = host_array_from_obj(object);
    if (!obj || !obj->tensor || obj->tensor->ndims == 0)
    {
        *count = 0;
        return SUCCESS;
    }

    *count = obj->tensor->shape[0];
    return SUCCESS;
}

int host_array_init()
{
    zend_class_entry ce;
    zend_class_entry ce_iterator;

    INIT_CLASS_ENTRY(ce_iterator, "Cuda\\HostArrayIterator", host_array_iterator_methods);
    host_array_iterator_ce = zend_register_internal_class(&ce_iterator);
    host_array_iterator_ce->create_object = host_array_iterator_create_object;
    host_array_iterator_ce->ce_flags |= ZEND_ACC_FINAL;

    zend_class_implements(host_array_iterator_ce, 1, zend_ce_iterator);

    memcpy(&host_array_iterator_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    host_array_iterator_handlers.offset = XtOffsetOf(host_array_iterator_object, std);
    host_array_iterator_handlers.free_obj = host_array_iterator_free_object;

    INIT_CLASS_ENTRY(ce, "Cuda\\HostArray", host_array_methods);
    host_array_ce = zend_register_internal_class(&ce);

    host_array_ce->create_object = host_array_create_object;
    host_array_ce->ce_flags |= ZEND_ACC_FINAL;

    zend_class_implements(host_array_ce, 2,
                          zend_ce_aggregate,
                          zend_ce_countable);

    memcpy(&host_array_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    host_array_handlers.offset = XtOffsetOf(host_array_object, std);
    host_array_handlers.free_obj = host_array_free_object;

    host_array_handlers.read_dimension = host_array_offset_get;
    host_array_handlers.has_dimension = host_array_offset_exists;
    host_array_handlers.write_dimension = host_array_offset_set;
    host_array_handlers.unset_dimension = host_array_offset_unset;

    host_array_handlers.count_elements = host_array_count_elements;

    return SUCCESS;
}