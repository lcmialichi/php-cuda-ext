#include "zend_compile.h"
#include "zend.h"
#include "zend_hash.h"
#include "zend_attributes.h"
#include "cuda_attributes.h"
#include "kernel_reflection.h"
#include "kernel_types.h"
#include "data_types.h"
#include "ext/standard/php_smart_string.h"
#include <stdio.h>


zend_array* kernel_get_closure_use_vars(zend_object *closure_obj)
{
    const zend_function *func = zend_get_closure_method_def(closure_obj);
    if (!func) {
        return NULL;
    }
    
    // Infelizmente, não há função pública para acessar use_vars diretamente
    // Mas podemos tentar através da propriedade dinâmica "__use_vars"
    // (Esta é uma propriedade interna usada pelo PHP)
    
    zval *use_vars_zv = zend_read_property(zend_ce_closure, closure_obj, "__use_vars", sizeof("__use_vars")-1, 1, NULL);
    if (use_vars_zv && Z_TYPE_P(use_vars_zv) == IS_ARRAY) {
        return Z_ARR_P(use_vars_zv);
    }
    
    return NULL;
}

int kernel_extract_closure_source(zend_object *closure_obj, zend_string **out_source)
{
    *out_source = NULL;
    
    if (!closure_obj) {
        return 0;
    }
    
    const zend_function *func = zend_get_closure_method_def(closure_obj);
    if (!func || func->type != ZEND_USER_FUNCTION) {
        return 0;
    }
    
    zend_op_array *op_array = (zend_op_array*)func;
    
    if (!op_array->filename) {
        return 0;
    }
    
    FILE *file = fopen(ZSTR_VAL(op_array->filename), "r");
    if (!file) {
        return 0;
    }
    
    smart_string source = {0};
    smart_string_alloc(&source, 2048, 0);
    
    smart_string_appends(&source, "<?php\n");
    
    char line[4096];
    uint32_t current_line = 1;
    bool in_closure = false;
    int brace_depth = 0;
    
    uint32_t start_line = op_array->line_start;
    uint32_t end_line = op_array->line_end;
    
    if (start_line == 0 || end_line == 0) {
        fclose(file);
        return 0;
    }
    
    while (fgets(line, sizeof(line), file)) {
        if (current_line >= start_line && current_line <= end_line) {
            char *trimmed = line;
            while (*trimmed == ' ' || *trimmed == '\t') trimmed++;
            
            if (!in_closure) {
                if (strstr(trimmed, "function") || strstr(trimmed, "fn") || 
                    (strstr(trimmed, "static") && strstr(trimmed, "function"))) {
                    in_closure = true;
                }
            }
            
            if (in_closure) {
                smart_string_appends(&source, line);
                for (char *p = line; *p; p++) {
                    if (*p == '{') brace_depth++;
                    else if (*p == '}') {
                        brace_depth--;
                        if (brace_depth <= 0) {
                            in_closure = false;
                            break;
                        }
                    }
                }
            }
        }
        
        if (current_line > end_line) break;
        current_line++;
    }
    
    fclose(file);
    
    if (source.len == 0) {
        smart_string_free(&source);
        return 0;
    }
    
    smart_string_0(&source);
    *out_source = zend_string_init(source.c, source.len, 0);
    smart_string_free(&source);
    
    return 1;
}

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
