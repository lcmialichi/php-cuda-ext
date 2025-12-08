#include "php.h"
#include "zend_hash.h"
#include "zend_attributes.h"
#include "cuda_attributes.h"
#include "kernel_reflection.h"

cuda_method_attribute_args *cuda_extract_method_attribute(
    zend_function *fptr,
    zend_class_entry *ce_attribute)
{
    if (!fptr || !ce_attribute || !fptr->common.attributes) {
        return NULL;
    }

    HashTable *attrs = fptr->common.attributes;
    zend_attribute *attr;
    zend_attribute *matched = NULL;

    ZEND_HASH_FOREACH_PTR(attrs, attr)
    {
        const char *attr_name = ZSTR_VAL(attr->name);
        for (uint32_t i = 0; i < attr->argc; i++) {
            zend_attribute_arg *a = &attr->args[i];

            zval tmp;
            ZVAL_COPY(&tmp, &a->value);
            convert_to_string(&tmp);
            php_printf("%s\n", Z_STRVAL(tmp));
            zval_ptr_dtor(&tmp);
        }

        if (zend_string_equals(attr->name, ce_attribute->name)) {
            matched = attr;
        }

    }
    ZEND_HASH_FOREACH_END();

    if (!matched) {
        return NULL;
    }

    cuda_method_attribute_args *args =
        (cuda_method_attribute_args *) emalloc(sizeof(cuda_method_attribute_args));

    args->name   = zend_string_copy(fptr->common.function_name);
    args->target = zend_string_init("sm_60", strlen("sm_60"), 0);

    for (uint32_t i = 0; i < matched->argc; i++) {
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
