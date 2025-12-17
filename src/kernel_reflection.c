#include "zend_compile.h"
#include "zend.h"
#include "zend_hash.h"
#include "zend_attributes.h"
#include "cuda_attributes.h"
#include "kernel_reflection.h"
#include "kernel_types.h"
#include "data_types.h"
#include "cuda_param.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define ZEND_TYPE_IS_ARRAY(type) \
    (ZEND_TYPE_IS_SET(type) && (ZEND_TYPE_PURE_MASK(type) & MAY_BE_ARRAY))

static dtype_t map_dtype_string_to_int(zend_string *dtype_str)
{
    if (!dtype_str)
        return DTYPE_UNKNOWN;

    char *str = ZSTR_VAL(dtype_str);
    size_t len = ZSTR_LEN(dtype_str);

    char *lower = estrndup(str, len);
    for (size_t i = 0; i < len; i++)
    {
        lower[i] = tolower((unsigned char)str[i]);
    }

    dtype_t result = DTYPE_UNKNOWN;

    if (strcasecmp(lower, "float32") == 0 || strcasecmp(lower, "float") == 0)
    {
        result = FLOAT32;
    }
    else if (strcasecmp(lower, "float64") == 0 || strcasecmp(lower, "double") == 0)
    {
        result = FLOAT64;
    }
    else if (strcasecmp(lower, "int32") == 0 || strcasecmp(lower, "int") == 0)
    {
        result = INT32;
    }
    else if (strcasecmp(lower, "int64") == 0 || strcasecmp(lower, "long") == 0)
    {
        result = INT64;
    }
    else if (strcasecmp(lower, "int8") == 0)
    {
        result = INT8;
    }
    else if (strcasecmp(lower, "int16") == 0)
    {
        result = INT16;
    }
    else if (strcasecmp(lower, "uint8") == 0)
    {
        result = UINT8;
    }
    else if (strcasecmp(lower, "uint16") == 0)
    {
        result = UINT16;
    }
    else if (strcasecmp(lower, "uint32") == 0)
    {
        result = UINT32;
    }
    else if (strcasecmp(lower, "uint64") == 0)
    {
        result = UINT64;
    }
    else if (strcasecmp(lower, "bool") == 0 || strcasecmp(lower, "boolean") == 0)
    {
        result = BOOL;
    }

    efree(lower);
    return result;
}

static int call_method_with_0_params(zend_object *obj, zend_class_entry *ce,
                                     const char *method_name, zval *retval)
{
    if (!obj || !ce || !method_name || !retval)
    {
        return FAILURE;
    }

    zend_string *method_name_str = zend_string_init(method_name, strlen(method_name), 0);
    zend_function *method = zend_hash_find_ptr(&ce->function_table, method_name_str);
    zend_string_release(method_name_str);

    if (!method)
    {
        return FAILURE;
    }

    zend_fcall_info fci;
    zend_fcall_info_cache fcc;

    memset(&fci, 0, sizeof(zend_fcall_info));
    memset(&fcc, 0, sizeof(zend_fcall_info_cache));

    fci.size = sizeof(zend_fcall_info);
    fci.object = obj;
    fci.retval = retval;
    fci.param_count = 0;

    fcc.function_handler = method;
    fcc.called_scope = ce;
    fcc.object = obj;

    return zend_call_function(&fci, &fcc);
}

static void set_properties_from_attr_args(zend_class_entry *ce, zval *obj,
                                          zend_attribute *attr)
{
    for (uint32_t i = 0; i < attr->argc; i++)
    {
        zend_attribute_arg *arg = &attr->args[i];
        if (arg->name)
        {
            zend_update_property(ce, Z_OBJ_P(obj),
                                 ZSTR_VAL(arg->name), ZSTR_LEN(arg->name),
                                 &arg->value);
        }
    }
}

static zend_function *find_method(zend_class_entry *ce, const char *method_name)
{
    size_t len = strlen(method_name);
    return zend_hash_str_find_ptr(&ce->function_table, method_name, len);
}

static zval *call_attribute_method(zend_class_entry *ce, zval *obj,
                                   const char *method_name)
{
    static zval result;
    ZVAL_NULL(&result);

    zend_function *method = find_method(ce, method_name);
    if (!method)
    {
        return &result;
    }

    if (call_method_with_0_params(Z_OBJ_P(obj), ce, method_name, &result) == SUCCESS)
    {
        return &result;
    }

    return &result;
}

cuda_param_info *cuda_param_info_create(zend_string *name)
{
    cuda_param_info *info = emalloc(sizeof(cuda_param_info));
    if (!info)
    {
        return NULL;
    }
    memset(info, 0, sizeof(cuda_param_info));
    info->name = zend_string_copy(name);
    return info;
}

void cuda_param_info_free(cuda_param_info *info)
{
    if (!info)
        return;

    if (info->name)
    {
        zend_string_release(info->name);
    }
    if (info->dtype)
    {
        zend_string_release(info->dtype);
    }
    efree(info);
}

zend_string *infer_dtype_from_php_type(zend_arg_info *arg_info)
{
    if (!arg_info || !ZEND_TYPE_IS_SET(arg_info->type))
    {
        return zend_string_init("float32", strlen("float32"), 0);
    }

    zend_type type = arg_info->type;
    uint32_t type_mask = ZEND_TYPE_PURE_MASK(type);

    if (type_mask & MAY_BE_LONG)
    {
        return zend_string_init("int32", strlen("int32"), 0);
    }
    else if (type_mask & MAY_BE_DOUBLE)
    {
        return zend_string_init("float64", strlen("float64"), 0);
    }
    else if (type_mask & MAY_BE_BOOL)
    {
        return zend_string_init("bool", strlen("bool"), 0);
    }
    else if (type_mask & MAY_BE_STRING)
    {
        return NULL;
    }
    else if (type_mask & MAY_BE_ARRAY)
    {
        return zend_string_init("float32", strlen("float32"), 0);
    }
    else if (ZEND_TYPE_NAME(type) != NULL)
    {
        return zend_string_init("float32", strlen("float32"), 0);
    }

    return zend_string_init("float32", strlen("float32"), 0);
}

cuda_param_info *extract_param_info(zend_attribute *attr,
                                    zend_string *param_name,
                                    zend_arg_info *arg_info)
{
    if (!attr || !param_name)
    {
        return NULL;
    }

    cuda_param_info *info = cuda_param_info_create(param_name);
    if (!info)
    {
        return NULL;
    }

    zend_class_entry *attr_ce = zend_hash_find_ptr(CG(class_table), attr->name);
    if (!attr_ce)
    {
        cuda_param_info_free(info);
        return NULL;
    }

    zval attr_obj;
    object_init_ex(&attr_obj, attr_ce);

    set_properties_from_attr_args(attr_ce, &attr_obj, attr);

    zval *result;

    result = call_attribute_method(attr_ce, &attr_obj, "getDtype");
    if (Z_TYPE_P(result) == IS_STRING)
    {
        info->dtype = zend_string_copy(Z_STR_P(result));
    }

    result = call_attribute_method(attr_ce, &attr_obj, "isList");
    info->is_list = Z_TYPE_P(result) == IS_TRUE;

    result = call_attribute_method(attr_ce, &attr_obj, "isNullable");
    info->nullable = Z_TYPE_P(result) == IS_TRUE;

    zval_ptr_dtor(&attr_obj);

    if (!info->dtype)
    {
        zend_string *inferred = infer_dtype_from_php_type(arg_info);
        info->dtype = inferred ? inferred : zend_string_init("float32", strlen("float32"), 0);
    }

    if (!info->is_list)
    {
        info->is_list = 1;
    }

    if (!info->nullable)
    {
        info->nullable = 1;
    }

    return info;
}

void convert_param_info_to_func_parameter(cuda_param_info *info, func_parameter *param)
{
    if (!info || !param)
    {
        return;
    }

    size_t name_len = ZSTR_LEN(info->name);
    size_t copy_len = name_len < 31 ? name_len : 31;
    memcpy(param->name, ZSTR_VAL(info->name), copy_len);
    param->name[copy_len] = '\0';

    param->dtype = info->dtype ? map_dtype_string_to_int(info->dtype) : DTYPE_UNKNOWN;

    if (info->is_list)
    {
        param->second_dtype = param->dtype;
        param->dtype = LIST;
    }
    else
    {
        param->second_dtype = DTYPE_UNKNOWN;
    }

    param->type = INPUT;
    if (info->nullable)
    {
        param->type = OUTPUT;
    }
}

void add_param_info_to_list(func_parameter_list_t *list, cuda_param_info *info)
{
    if (!list || !info)
    {
        return;
    }

    list->total++;
    list->parameters = (func_parameter **)erealloc(
        list->parameters,
        list->total * sizeof(func_parameter *));

    func_parameter *param = (func_parameter *)emalloc(sizeof(func_parameter));
    memset(param, 0, sizeof(func_parameter));

    convert_param_info_to_func_parameter(info, param);

    list->parameters[list->total - 1] = param;
}

func_parameter_list_t *cuda_extract_parameters(zend_function *fptr)
{
    if (!fptr || !fptr->common.arg_info)
    {
        return NULL;
    }

    func_parameter_list_t *param_list = emalloc(sizeof(func_parameter_list_t));
    memset(param_list, 0, sizeof(func_parameter_list_t));

    uint32_t num_args = fptr->common.num_args;
    if (num_args == 0)
    {
        return param_list;
    }

    HashTable *attributes = fptr->common.attributes;
    if (!attributes)
    {
        return param_list;
    }

    zend_attribute *attr;

    ZEND_HASH_FOREACH_PTR(attributes, attr)
    {
        if (!attr || !attr->target || !(attr->target & ZEND_ATTRIBUTE_TARGET_PARAMETER))
        {
            continue;
        }

        uint32_t param_index = attr->offset;

        if (param_index >= num_args)
        {
            continue;
        }

        zend_arg_info *arg = &fptr->common.arg_info[param_index];
        zend_string *param_name = arg->name;

        if (!param_name)
        {
            continue;
        }

        if (instanceof_function(attr->ce, cuda_param_attribute_ce))
        {
            cuda_param_info *info = extract_param_info(attr, param_name, arg);
            if (info)
            {
                add_param_info_to_list(param_list, info);
                cuda_param_info_free(info);
            }
        }
    }
    ZEND_HASH_FOREACH_END();

    return param_list;
}

cuda_method_attribute_args *cuda_extract_method_attribute(
    zend_function *fptr,
    zend_class_entry *ce_attribute)
{
    if (!fptr || !ce_attribute || !fptr->common.attributes)
    {
        return NULL;
    }

    HashTable *attrs = fptr->common.attributes;
    zend_attribute *matched = NULL;
    zend_attribute *attr;

    ZEND_HASH_FOREACH_PTR(attrs, attr)
    {
        if (zend_string_equals(attr->name, ce_attribute->name))
        {
            matched = attr;
            break;
        }
    }
    ZEND_HASH_FOREACH_END();

    if (!matched)
    {
        return NULL;
    }

    cuda_method_attribute_args *args =
        (cuda_method_attribute_args *)emalloc(sizeof(cuda_method_attribute_args));

    args->name = zend_string_copy(fptr->common.function_name);
    args->target = zend_string_init("sm_60", strlen("sm_60"), 0);

    for (uint32_t i = 0; i < matched->argc; i++)
    {
        zend_attribute_arg *a = &matched->args[i];
        if (!a->name)
            continue;

        if (zend_string_equals_literal(a->name, "name") &&
            Z_TYPE(a->value) == IS_STRING)
        {
            zend_string_release(args->name);
            args->name = zend_string_copy(Z_STR(a->value));
        }
        else if (zend_string_equals_literal(a->name, "target") &&
                 Z_TYPE(a->value) == IS_STRING)
        {
            zend_string_release(args->target);
            args->target = zend_string_copy(Z_STR(a->value));
        }
    }

    return args;
}