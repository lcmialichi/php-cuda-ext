#include "zend_compile.h"
#include "zend.h"
#include "zend_hash.h"
#include "zend_attributes.h"
#include "cuda_attributes.h"
#include "kernel_reflection.h"
#include "kernel_types.h"
#include "data_types.h"

dtype_t map_dtype_string_to_int(zend_string *dtype_str)
{
    if (!dtype_str)
        return DTYPE_UNKNOWN;

    if (zend_string_equals_literal_ci(dtype_str, "float"))
    {
        return FLOAT32;
    }
    if (zend_string_equals_literal_ci(dtype_str, "int"))
    {
        return INT32;
    }
    if (zend_string_equals_literal_ci(dtype_str, "double"))
    {
        return FLOAT64;
    }
    if (zend_string_equals_literal_ci(dtype_str, "bool"))
    {
        return BOOL;
    }
    return DTYPE_UNKNOWN;
}

void add_parameter_to_list(func_parameter_list_t *list, parameter_type_t type, const char *name, dtype_t dtype)
{

    list->total++;
    list->parameters = (func_parameter **)erealloc(
        list->parameters,
        list->total * sizeof(func_parameter *));

    func_parameter *param = (func_parameter *)emalloc(sizeof(func_parameter));

    param->type = type;
    strncpy(param->name, name, 31);
    param->name[31] = '\0';
    param->dtype = dtype;

    list->parameters[list->total - 1] = param;
}

func_parameter_list_t *cuda_extract_parameter_list(zend_function *fptr,
                                                 zend_class_entry *ce_input_attr,
                                                 zend_class_entry *ce_output_attr)
{
    if (!fptr || !fptr->common.arg_info)
    {
        return NULL;
    }

    func_parameter_list_t *param_list = (func_parameter_list_t *)emalloc(sizeof(func_parameter_list_t));
    param_list->total = 0;
    param_list->parameters = NULL;

    uint32_t num_args = fptr->common.num_args;
    if (num_args == 0)
    {
        return param_list;
    }

    HashTable *attributes = fptr->common.attributes;

    if (!attributes)
    {
        efree(param_list);
        return NULL;
    }

    zend_string *input_lcname = zend_string_tolower(ce_input_attr->name);
    zend_string *output_lcname = zend_string_tolower(ce_output_attr->name);

    for (uint32_t i = 0; i < num_args; i++)
    {
        zend_arg_info *arg = &fptr->common.arg_info[i];
        zend_string *var_name = arg->name;

        if (!var_name)
        {
            continue;
        }

        uint32_t offset = i;

        zend_attribute *matched_attr = NULL;
        parameter_type_t current_type = DTYPE_UNKNOWN;
        const char *type_name = NULL;

        matched_attr = zend_get_parameter_attribute(attributes, input_lcname, offset);
        if (matched_attr)
        {
            current_type = INPUT;
            type_name = "INPUT";
        }
        else
        {
            matched_attr = zend_get_parameter_attribute(attributes, output_lcname, offset);
            if (matched_attr)
            {
                current_type = OUTPUT;
                type_name = "OUTPUT";
            }
        }

        if (matched_attr)
        {
            zend_string *dtype_str = NULL;
            dtype_t dtype = DTYPE_UNKNOWN;

            for (uint32_t j = 0; j < matched_attr->argc; j++)
            {
                zend_attribute_arg *attr_arg = &matched_attr->args[j];

                if (attr_arg->name &&
                    zend_string_equals_literal(attr_arg->name, "dtype") &&
                    Z_TYPE(attr_arg->value) == IS_STRING)
                {
                    dtype_str = Z_STR(attr_arg->value);
                    dtype = map_dtype_string_to_int(dtype_str);
                    break;
                }
            }

            /** @todo ensure dtype is ok */
            if (dtype != DTYPE_UNKNOWN)
            {
                add_parameter_to_list(param_list, current_type, ZSTR_VAL(var_name), dtype);
            }
        }
    }

    zend_string_release(input_lcname);
    zend_string_release(output_lcname);

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
    zend_attribute *attr;
    zend_attribute *matched = NULL;

    ZEND_HASH_FOREACH_PTR(attrs, attr)
    {
        if (zend_string_equals(attr->name, ce_attribute->name))
        {
            matched = attr;
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

        if (zend_string_equals_literal(a->name, "target") &&
            Z_TYPE(a->value) == IS_STRING)
        {
            zend_string_release(args->target);
            args->target = zend_string_copy(Z_STR(a->value));
        }
    }

    return args;
}
