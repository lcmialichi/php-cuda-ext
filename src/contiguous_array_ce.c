#include "php.h"
#include "contiguous_array_ce.h"
#include "contiguous_array_arginfo.h"
#include <string.h>
#include <math.h>

static zend_object_handlers contiguous_array_handlers;
zend_class_entry *contiguous_array_ce;

static void contiguous_array_offset_set(zend_object *object, zval *offset, zval *value);
static void contiguous_array_offset_unset(zend_object *object, zval *offset);
static int contiguous_array_count_elements(zend_object *object, zend_long *count);

static contiguous_array_object *contiguous_array_from_obj(zend_object *zobj)
{
    return (contiguous_array_object *)((char *)zobj - contiguous_array_handlers.offset);
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
            *fptr = (float)Z_DVAL_P(value);
        else if (Z_TYPE_P(value) == IS_LONG)
            *fptr = (float)Z_LVAL_P(value);
        break;
    }
    case DTYPE_FLOAT64:
    {
        double *dptr = (double *)ptr;
        if (Z_TYPE_P(value) == IS_DOUBLE)
            *dptr = Z_DVAL_P(value);
        else if (Z_TYPE_P(value) == IS_LONG)
            *dptr = (double)Z_LVAL_P(value);
        break;
    }
    case DTYPE_INT8:
    {
        int8_t *iptr = (int8_t *)ptr;
        if (Z_TYPE_P(value) == IS_LONG)
            *iptr = (int8_t)Z_LVAL_P(value);
        else if (Z_TYPE_P(value) == IS_DOUBLE)
            *iptr = (int8_t)Z_DVAL_P(value);
        break;
    }
    case DTYPE_INT16:
    {
        int16_t *iptr = (int16_t *)ptr;
        if (Z_TYPE_P(value) == IS_LONG)
            *iptr = (int16_t)Z_LVAL_P(value);
        else if (Z_TYPE_P(value) == IS_DOUBLE)
            *iptr = (int16_t)Z_DVAL_P(value);
        break;
    }
    case DTYPE_INT32:
    {
        int32_t *iptr = (int32_t *)ptr;
        if (Z_TYPE_P(value) == IS_LONG)
            *iptr = (int32_t)Z_LVAL_P(value);
        else if (Z_TYPE_P(value) == IS_DOUBLE)
            *iptr = (int32_t)Z_DVAL_P(value);
        break;
    }
    case DTYPE_INT64:
    {
        int64_t *iptr = (int64_t *)ptr;
        if (Z_TYPE_P(value) == IS_LONG)
            *iptr = (int64_t)Z_LVAL_P(value);
        else if (Z_TYPE_P(value) == IS_DOUBLE)
            *iptr = (int64_t)Z_DVAL_P(value);
        break;
    }
    case DTYPE_UINT8:
    {
        uint8_t *uptr = (uint8_t *)ptr;
        if (Z_TYPE_P(value) == IS_LONG)
            *uptr = (uint8_t)Z_LVAL_P(value);
        else if (Z_TYPE_P(value) == IS_DOUBLE)
            *uptr = (uint8_t)Z_DVAL_P(value);
        break;
    }
    case DTYPE_UINT16:
    {
        uint16_t *uptr = (uint16_t *)ptr;
        if (Z_TYPE_P(value) == IS_LONG)
            *uptr = (uint16_t)Z_LVAL_P(value);
        else if (Z_TYPE_P(value) == IS_DOUBLE)
            *uptr = (uint16_t)Z_DVAL_P(value);
        break;
    }
    case DTYPE_UINT32:
    {
        uint32_t *uptr = (uint32_t *)ptr;
        if (Z_TYPE_P(value) == IS_LONG)
            *uptr = (uint32_t)Z_LVAL_P(value);
        else if (Z_TYPE_P(value) == IS_DOUBLE)
            *uptr = (uint32_t)Z_DVAL_P(value);
        break;
    }
    case DTYPE_UINT64:
    {
        uint64_t *uptr = (uint64_t *)ptr;
        if (Z_TYPE_P(value) == IS_LONG)
            *uptr = (uint64_t)Z_LVAL_P(value);
        else if (Z_TYPE_P(value) == IS_DOUBLE)
            *uptr = (uint64_t)Z_DVAL_P(value);
        break;
    }
    case DTYPE_BOOL:
    {
        uint8_t *bptr = (uint8_t *)ptr;
        if (Z_TYPE_P(value) == IS_TRUE)
            *bptr = 1;
        else if (Z_TYPE_P(value) == IS_FALSE)
            *bptr = 0;
        else if (Z_TYPE_P(value) == IS_LONG)
            *bptr = Z_LVAL_P(value) != 0;
        else if (Z_TYPE_P(value) == IS_DOUBLE)
            *bptr = Z_DVAL_P(value) != 0.0;
        break;
    }
    default:
        break;
    }
}

void *allocate_for_dtype(dtype_t dtype, size_t count)
{
    size_t size = dtype_to_size(dtype);
    void *ptr = ecalloc(count, size);
    return ptr;
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

static size_t calculate_linear_offset(contiguous_array_object *obj, size_t *indices)
{
    size_t offset = obj->offset;
    for (int i = 0; i < obj->ndims; i++)
    {
        offset += indices[i] * obj->strides[i];
    }
    return offset;
}

static zend_object *contiguous_array_create_slice(contiguous_array_object *parent, size_t index)
{
    zend_object *zobj = contiguous_array_create_object(contiguous_array_ce);
    contiguous_array_object *slice = contiguous_array_from_obj(zobj);

    parent->tensor->ref_count++;
    slice->tensor = parent->tensor;

    slice->offset = parent->offset + index * parent->strides[0];
    slice->shape = parent->shape + 1;
    slice->strides = parent->strides + 1;
    slice->ndims = parent->ndims - 1;
    slice->read_only = 1;

    return zobj;
}

ZEND_METHOD(ContiguousArray, __construct)
{
    zend_throw_error(NULL, "ContiguousArray cannot be constructed directly. Use CudaArray::toHost()");
    RETURN_NULL();
}

ZEND_METHOD(ContiguousArray, get)
{
    zend_object *zobj = Z_OBJ_P(getThis());
    contiguous_array_object *obj = contiguous_array_from_obj(zobj);

    if (!obj || !obj->tensor)
    {
        zend_throw_error(NULL, "ContiguousArray is not properly initialized");
        RETURN_NULL();
    }

    zval *index_array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_ARRAY(index_array)
    ZEND_PARSE_PARAMETERS_END();

    if (!obj->tensor->data || obj->ndims == 0)
    {
        zend_throw_error(NULL, "Array not initialized or empty");
        return;
    }

    zend_long idx_count = zend_array_count(Z_ARRVAL_P(index_array));
    if (idx_count != obj->ndims)
    {
        zend_throw_error(NULL, "Incorrect number of indices: expected %d, got %ld",
                         obj->ndims, idx_count);
        return;
    }

    size_t indices[obj->ndims];
    int i = 0;
    zval *val;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(index_array), val)
    {
        zend_long idx = zval_get_long(val);
        if (idx < 0 || idx >= obj->shape[i])
        {
            zend_throw_error(NULL, "Index %ld out of bounds for dimension %d (size %d)",
                             idx, i, obj->shape[i]);
            return;
        }
        indices[i] = (size_t)idx;
        i++;
    }
    ZEND_HASH_FOREACH_END();

    size_t offset = calculate_linear_offset(obj, indices);
    get_value_for_dtype(obj->tensor->data, offset, return_value, obj->tensor->dtype);
}

ZEND_METHOD(ContiguousArray, getShape)
{
    zend_object *zobj = Z_OBJ_P(getThis());
    contiguous_array_object *obj = contiguous_array_from_obj(zobj);

    if (!obj || !obj->tensor)
    {
        zend_throw_error(NULL, "ContiguousArray is not properly initialized");
        RETURN_NULL();
    }

    array_init(return_value);
    for (int i = 0; i < obj->ndims; i++)
    {
        add_next_index_long(return_value, obj->shape[i]);
    }
}

ZEND_METHOD(ContiguousArray, toArray)
{
    zend_object *zobj = Z_OBJ_P(getThis());
    contiguous_array_object *obj = contiguous_array_from_obj(zobj);

    if (!obj || !obj->tensor)
    {
        zend_throw_error(NULL, "ContiguousArray is not properly initialized");
        RETURN_NULL();
    }

    if (!obj->tensor->data || obj->ndims == 0)
    {
        RETURN_NULL();
    }

    size_t total_elements = 1;
    for (int i = 0; i < obj->ndims; i++)
    {
        total_elements *= obj->shape[i];
    }

    if (obj->ndims == 1)
    {
        array_init_size(return_value, total_elements);
        for (size_t i = 0; i < total_elements; i++)
        {
            zval element;
            size_t offset = obj->offset + i * obj->strides[0];
            get_value_for_dtype(obj->tensor->data, offset, &element, obj->tensor->dtype);
            add_next_index_zval(return_value, &element);
        }
    }
    else
    {
        size_t indices[obj->ndims];
        memset(indices, 0, sizeof(size_t) * obj->ndims);

        zval build_array(contiguous_array_object * obj, int dim)
        {
            zval arr;
            array_init_size(&arr, obj->shape[dim]);

            if (dim == obj->ndims - 1)
            {
                for (int i = 0; i < obj->shape[dim]; i++)
                {
                    indices[dim] = i;
                    size_t offset = calculate_linear_offset(obj, indices);
                    zval element;
                    get_value_for_dtype(obj->tensor->data, offset, &element, obj->tensor->dtype);
                    add_next_index_zval(&arr, &element);
                }
            }
            else
            {
                for (int i = 0; i < obj->shape[dim]; i++)
                {
                    indices[dim] = i;
                    zval subarray = build_array(obj, dim + 1);
                    add_next_index_zval(&arr, &subarray);
                }
            }
            return arr;
        }

        *return_value = build_array(obj, 0);
    }
}

ZEND_METHOD(ContiguousArray, getNdims)
{
    zend_object *zobj = Z_OBJ_P(getThis());
    contiguous_array_object *obj = contiguous_array_from_obj(zobj);

    if (!obj || !obj->tensor)
    {
        RETURN_LONG(0);
    }

    RETURN_LONG(obj->ndims);
}

ZEND_METHOD(ContiguousArray, getSize)
{
    zend_object *zobj = Z_OBJ_P(getThis());
    contiguous_array_object *obj = contiguous_array_from_obj(zobj);

    if (!obj || !obj->tensor)
    {
        RETURN_LONG(0);
    }

    size_t total = 1;
    for (int i = 0; i < obj->ndims; i++)
    {
        total *= obj->shape[i];
    }
    RETURN_LONG(total);
}

ZEND_METHOD(ContiguousArray, getDtype)
{
    zend_object *zobj = Z_OBJ_P(getThis());
    contiguous_array_object *obj = contiguous_array_from_obj(zobj);

    if (!obj || !obj->tensor)
    {
        RETURN_STRING("unknown");
    }

    const char *dtype_str = dtype_to_string(obj->tensor->dtype);
    RETURN_STRING(dtype_str);
}

ZEND_METHOD(ContiguousArray, getElementSize)
{
    zend_object *zobj = Z_OBJ_P(getThis());
    contiguous_array_object *obj = contiguous_array_from_obj(zobj);

    if (!obj || !obj->tensor)
    {
        RETURN_LONG(0);
    }

    RETURN_LONG(dtype_to_size(obj->tensor->dtype));
}

ZEND_METHOD(ContiguousArray, count)
{
    zend_object *zobj = Z_OBJ_P(getThis());
    contiguous_array_object *obj = contiguous_array_from_obj(zobj);

    if (!obj || !obj->tensor || obj->ndims == 0)
    {
        RETURN_LONG(0);
    }

    RETURN_LONG(obj->shape[0]);
}

zend_object *contiguous_array_create_object(zend_class_entry *ce)
{
    contiguous_array_object *obj = (contiguous_array_object *)ecalloc(1, sizeof(contiguous_array_object));

    zend_object_std_init(&obj->std, ce);
    object_properties_init(&obj->std, ce);

    obj->tensor = NULL;
    obj->offset = 0;
    obj->shape = NULL;
    obj->strides = NULL;
    obj->ndims = 0;
    obj->read_only = 1;

    obj->std.handlers = &contiguous_array_handlers;

    return &obj->std;
}

zend_object *contiguous_array_from_tensor(tensor_t *tensor)
{
    if (!tensor)
    {
        return NULL;
    }

    zend_object *zobj = contiguous_array_create_object(contiguous_array_ce);
    contiguous_array_object *obj = contiguous_array_from_obj(zobj);

    if (tensor->is_on_gpu)
    {
        // may convert to host if is in GPU
    }

    tensor->ref_count++;
    obj->tensor = tensor;
    obj->offset = tensor->offset;
    obj->shape = tensor->shape;
    obj->strides = tensor->strides;
    obj->ndims = tensor->ndims;
    obj->read_only = 1;

    return zobj;
}

void contiguous_array_free_object(zend_object *object)
{
    contiguous_array_object *obj = contiguous_array_from_obj(object);

    if (obj->tensor != NULL)
    {
        cuda_tensor_destroy(obj->tensor);
        obj->tensor = NULL;
    }

    zend_object_std_dtor(&obj->std);
}

static int contiguous_array_offset_exists(zend_object *object, zval *offset, int type)
{
    contiguous_array_object *obj = contiguous_array_from_obj(object);

    if (!obj || !obj->tensor || obj->ndims == 0 || !obj->shape)
    {
        return 0;
    }

    zend_long idx = zval_get_long(offset);
    return (idx >= 0 && idx < obj->shape[0]);
}

static zval *contiguous_array_offset_get(zend_object *object, zval *offset, int type, zval *rv)
{
    contiguous_array_object *obj = contiguous_array_from_obj(object);

    if (!obj || !obj->tensor || obj->ndims <= 0 || !obj->tensor->data)
    {
        ZVAL_NULL(rv);
        return rv;
    }

    zend_long idx = zval_get_long(offset);
    if (idx < 0 || idx >= obj->shape[0])
    {
        ZVAL_NULL(rv);
        return rv;
    }

    if (obj->ndims == 1)
    {
        size_t linear_offset = obj->offset + idx * obj->strides[0];
        get_value_for_dtype(obj->tensor->data, linear_offset, rv, obj->tensor->dtype);
    }
    else
    {
        zend_object *slice_obj = contiguous_array_create_slice(obj, idx);
        ZVAL_OBJ(rv, slice_obj);
    }

    return rv;
}

static void contiguous_array_offset_set(zend_object *object, zval *offset, zval *value)
{
    zend_throw_error(NULL, "ContiguousArray is read-only");
}

static void contiguous_array_offset_unset(zend_object *object, zval *offset)
{
    zend_throw_error(NULL, "ContiguousArray elements cannot be unset");
}

static void contiguous_array_iterator_dtor(zend_object_iterator *iter)
{
    contiguous_array_iterator *iterator = (contiguous_array_iterator *)iter;
    zval_ptr_dtor(&iterator->host_array);
    if (iterator->indices)
    {
        efree(iterator->indices);
    }
    if (Z_TYPE(iterator->current) != IS_UNDEF)
    {
        zval_ptr_dtor(&iterator->current);
    }

}

static void contiguous_array_iterator_invalidate_current(zend_object_iterator *iter)
{
    contiguous_array_iterator *iterator = (contiguous_array_iterator *)iter;
    
    if (Z_TYPE(iterator->current) != IS_UNDEF) {
        zval_ptr_dtor(&iterator->current);
        ZVAL_UNDEF(&iterator->current);
    }
}

static int contiguous_array_iterator_valid(zend_object_iterator *iter)
{
    contiguous_array_iterator *iterator = (contiguous_array_iterator *)iter;
    
    if (Z_TYPE(iterator->host_array) != IS_OBJECT) {
        return FAILURE;
    }
    
    contiguous_array_object *obj = contiguous_array_from_obj(Z_OBJ(iterator->host_array));
    if (!obj || !obj->tensor) {
        return FAILURE;
    }
    
    return (iterator->current_idx < iterator->max_idx) ? SUCCESS : FAILURE;
}

static zval *contiguous_array_iterator_get_current_data(zend_object_iterator *iter)
{
    contiguous_array_iterator *iterator = (contiguous_array_iterator *)iter;
    
    if (Z_TYPE(iterator->current) != IS_UNDEF) {
        zval_ptr_dtor(&iterator->current);
        ZVAL_UNDEF(&iterator->current);
    }
    
    contiguous_array_object *obj = contiguous_array_from_obj(Z_OBJ(iterator->host_array));
    
    if (obj->ndims == 1) {
        size_t offset = obj->offset + iterator->current_idx * obj->strides[0];
        get_value_for_dtype(obj->tensor->data, offset, &iterator->current, obj->tensor->dtype);
    } else {
        zend_object *slice_obj = contiguous_array_create_slice(obj, iterator->current_idx);
        ZVAL_OBJ(&iterator->current, slice_obj);
    }
    
    return &iterator->current;
}

static void contiguous_array_iterator_get_current_key(zend_object_iterator *iter, zval *key)
{
    contiguous_array_iterator *iterator = (contiguous_array_iterator *)iter;
    ZVAL_LONG(key, iterator->current_idx);
}

static void contiguous_array_iterator_move_forward(zend_object_iterator *iter)
{
    contiguous_array_iterator *iterator = (contiguous_array_iterator *)iter;
    if (Z_TYPE(iterator->current) != IS_UNDEF) {
        zval_ptr_dtor(&iterator->current);
        ZVAL_UNDEF(&iterator->current);
    }
    
    iterator->current_idx++;
}

static void contiguous_array_iterator_rewind(zend_object_iterator *iter)
{
    contiguous_array_iterator *iterator = (contiguous_array_iterator *)iter;
    
    if (Z_TYPE(iterator->current) != IS_UNDEF) {
        zval_ptr_dtor(&iterator->current);
        ZVAL_UNDEF(&iterator->current);
    }
    
    iterator->current_idx = 0;
}

static const zend_object_iterator_funcs contiguous_array_iterator_funcs = {
    contiguous_array_iterator_dtor,
    contiguous_array_iterator_valid,
    contiguous_array_iterator_get_current_data,
    contiguous_array_iterator_get_current_key,
    contiguous_array_iterator_move_forward,
    contiguous_array_iterator_rewind,
    contiguous_array_iterator_invalidate_current,
    NULL};

static zend_object_iterator *contiguous_array_get_iterator(zend_class_entry *ce, zval *object, int by_ref)
{
    if (by_ref)
    {
        zend_throw_error(NULL, "An iterator cannot be used with foreach by reference");
        return NULL;
    }

    contiguous_array_object *obj = contiguous_array_from_obj(Z_OBJ_P(object));
    if (!obj || !obj->tensor)
    {
        return NULL;
    }

    contiguous_array_iterator *iterator = emalloc(sizeof(contiguous_array_iterator));
    zend_iterator_init((zend_object_iterator *)iterator);

    ZVAL_COPY(&iterator->host_array, object);
    iterator->current_idx = 0;
    iterator->max_idx = obj->ndims > 0 ? obj->shape[0] : 0;
    ZVAL_UNDEF(&iterator->current);
    iterator->indices = NULL;
    iterator->iter_ndims = 0;

    iterator->intern.funcs = &contiguous_array_iterator_funcs;

    return (zend_object_iterator *)iterator;
}

static int contiguous_array_count_elements(zend_object *object, zend_long *count)
{
    contiguous_array_object *obj = contiguous_array_from_obj(object);
    if (!obj || !obj->tensor || obj->ndims == 0)
    {
        *count = 0;
        return SUCCESS;
    }

    *count = obj->shape[0];
    return SUCCESS;
}

int contiguous_array_init()
{
    zend_class_entry ce;

    INIT_CLASS_ENTRY(ce, "Cuda\\ContiguousArray", contiguous_array_methods);
    contiguous_array_ce = zend_register_internal_class(&ce);

    contiguous_array_ce->create_object = contiguous_array_create_object;
    contiguous_array_ce->ce_flags |= ZEND_ACC_FINAL;
    contiguous_array_ce->get_iterator = contiguous_array_get_iterator;

    zend_class_implements(contiguous_array_ce, 1, zend_ce_aggregate);

    memcpy(&contiguous_array_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    contiguous_array_handlers.offset = XtOffsetOf(contiguous_array_object, std);
    contiguous_array_handlers.free_obj = contiguous_array_free_object;

    contiguous_array_handlers.read_dimension = contiguous_array_offset_get;
    contiguous_array_handlers.has_dimension = contiguous_array_offset_exists;
    contiguous_array_handlers.write_dimension = contiguous_array_offset_set;
    contiguous_array_handlers.unset_dimension = contiguous_array_offset_unset;
    contiguous_array_handlers.count_elements = contiguous_array_count_elements;

    return SUCCESS;
}