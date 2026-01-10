#include "cuda_param.h"
#include "cuda_param_arginfo.h"
#include "zend_attributes.h"

zend_class_entry *cuda_param_attribute_ce;

static const zend_function_entry cuda_tensor_methods[];
static const zend_function_entry cuda_int_methods[];
static const zend_function_entry cuda_float_methods[];
static const zend_function_entry cuda_bool_methods[];

void cuda_param_info_free(cuda_param_info *info)
{
    if (info->name) {
        zend_string_release(info->name);
    }
    if (info->dtype) {
        zend_string_release(info->dtype);
    }
   
    efree(info);
}

ZEND_METHOD(CudaParamAttribute, getDtype)
{
    zend_throw_error(NULL, "Abstract method Cuda\\ParamAttribute::getDtype() must be implemented");
}

ZEND_METHOD(CudaParamAttribute, isList)
{
    RETURN_FALSE;
}

ZEND_METHOD(CudaParamAttribute, isNullable)
{
    RETURN_FALSE;
}

static const zend_function_entry cuda_param_attribute_methods[] = {
    ZEND_ME(CudaParamAttribute, getDtype, arginfo_cuda_param_attribute_getDtype, ZEND_ACC_PUBLIC | ZEND_ACC_ABSTRACT)
    ZEND_ME(CudaParamAttribute, isList, arginfo_cuda_param_attribute_isList, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaParamAttribute, isNullable, arginfo_cuda_param_attribute_isNullable, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

// ==================== TENSOR ====================
ZEND_METHOD(CudaAttr_TensorType, __construct)
{
    zend_string *dtype = NULL;
    zend_bool is_list = 1;
    zend_bool nullable = 0;
    
    ZEND_PARSE_PARAMETERS_START(0, 3)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR(dtype)
        Z_PARAM_BOOL(is_list)
        Z_PARAM_BOOL(nullable)
    ZEND_PARSE_PARAMETERS_END();
    
    if (dtype) {
        zend_update_property_str(cuda_param_attribute_ce, Z_OBJ_P(getThis()), 
                                "dtype", strlen("dtype"), dtype);
    }
    
    zend_update_property_bool(cuda_param_attribute_ce, Z_OBJ_P(getThis()),
                             "is_list", strlen("is_list"), is_list);
    
    zend_update_property_bool(cuda_param_attribute_ce, Z_OBJ_P(getThis()),
                             "nullable", strlen("nullable"), nullable);
}

ZEND_METHOD(CudaAttr_TensorType, getDtype)
{
    zval *dtype = zend_read_property(cuda_param_attribute_ce, Z_OBJ_P(getThis()),
                                    "dtype", strlen("dtype"), 1, NULL);
    
    if (dtype && Z_TYPE_P(dtype) == IS_STRING) {
        RETURN_STR(Z_STR_P(dtype));
    }
    
    RETURN_STRING("float32");
}

ZEND_METHOD(CudaAttr_TensorType, isList)
{
    RETURN_TRUE;
}

// ==================== INT ====================

ZEND_METHOD(CudaAttr_IntType, __construct)
{
    zend_long bits = 32;
    zend_bool nullable = 0;
    
    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(bits)
        Z_PARAM_BOOL(nullable)
    ZEND_PARSE_PARAMETERS_END();
    
    char dtype_str[10];
    snprintf(dtype_str, sizeof(dtype_str), "int%ld", bits);
    zend_string *dtype = zend_string_init(dtype_str, strlen(dtype_str), 0);
    
    zend_update_property_str(cuda_param_attribute_ce, Z_OBJ_P(getThis()),
                            "dtype", strlen("dtype"), dtype);
    zend_string_release(dtype);
    
    zend_update_property_bool(cuda_param_attribute_ce, Z_OBJ_P(getThis()),
                             "nullable", strlen("nullable"), nullable);
}

ZEND_METHOD(CudaAttr_IntType, getDtype)
{
    zval *dtype = zend_read_property(cuda_param_attribute_ce, Z_OBJ_P(getThis()),
                                    "dtype", strlen("dtype"), 1, NULL);
    
    if (dtype && Z_TYPE_P(dtype) == IS_STRING) {
        RETURN_STR(Z_STR_P(dtype));
    }
    
    RETURN_STRING("int32");
}

ZEND_METHOD(CudaAttr_IntType, isList)
{
    RETURN_FALSE;
}

// ==================== FLOAT ====================

ZEND_METHOD(CudaAttr_FloatType, __construct)
{
    zend_long bits = 32;
    zend_bool nullable = 0;
    
    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(bits)
        Z_PARAM_BOOL(nullable)
    ZEND_PARSE_PARAMETERS_END();
    
    char dtype_str[12];
    snprintf(dtype_str, sizeof(dtype_str), "float%ld", bits);
    zend_string *dtype = zend_string_init(dtype_str, strlen(dtype_str), 0);
    
    zend_update_property_str(cuda_param_attribute_ce, Z_OBJ_P(getThis()),
                            "dtype", strlen("dtype"), dtype);
    zend_string_release(dtype);
    
    zend_update_property_bool(cuda_param_attribute_ce, Z_OBJ_P(getThis()),
                             "nullable", strlen("nullable"), nullable);
}

ZEND_METHOD(CudaAttr_FloatType, getDtype)
{
    zval *dtype = zend_read_property(cuda_param_attribute_ce, Z_OBJ_P(getThis()),
                                    "dtype", strlen("dtype"), 1, NULL);
    
    if (dtype && Z_TYPE_P(dtype) == IS_STRING) {
        RETURN_STR(Z_STR_P(dtype));
    }
    
    RETURN_STRING("float32");
}

ZEND_METHOD(CudaAttr_FloatType, isList)
{
    RETURN_FALSE;
}

// ==================== DTYPE_BOOL ====================

ZEND_METHOD(CudaAttr_BoolType, __construct)
{
    zend_bool nullable = 0;
    
    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(nullable)
    ZEND_PARSE_PARAMETERS_END();
    
    zend_string *dtype = zend_string_init("bool", strlen("bool"), 0);
    zend_update_property_str(cuda_param_attribute_ce, Z_OBJ_P(getThis()),
                            "dtype", strlen("dtype"), dtype);
    zend_string_release(dtype);
    
    zend_update_property_bool(cuda_param_attribute_ce, Z_OBJ_P(getThis()),
                             "nullable", strlen("nullable"), nullable);
}

ZEND_METHOD(CudaAttr_BoolType, getDtype)
{
    RETURN_STRING("bool");
}

ZEND_METHOD(CudaAttr_BoolType, isList)
{
    RETURN_FALSE;
}

static const zend_function_entry cuda_tensor_methods[] = {
    ZEND_ME(CudaAttr_TensorType, __construct, arginfo_cuda_attr_tensor___construct, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaAttr_TensorType, getDtype, arginfo_cuda_param_attribute_getDtype, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaAttr_TensorType, isList, arginfo_cuda_param_attribute_isList, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

static const zend_function_entry cuda_int_methods[] = {
    ZEND_ME(CudaAttr_IntType, __construct, arginfo_cuda_attr_int___construct, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaAttr_IntType, getDtype, arginfo_cuda_param_attribute_getDtype, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaAttr_IntType, isList, arginfo_cuda_param_attribute_isList, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

static const zend_function_entry cuda_float_methods[] = {
    ZEND_ME(CudaAttr_FloatType, __construct, arginfo_cuda_attr_float___construct, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaAttr_FloatType, getDtype, arginfo_cuda_param_attribute_getDtype, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaAttr_FloatType, isList, arginfo_cuda_param_attribute_isList, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

static const zend_function_entry cuda_bool_methods[] = {
    ZEND_ME(CudaAttr_BoolType, __construct, arginfo_cuda_attr_bool___construct, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaAttr_BoolType, getDtype, arginfo_cuda_param_attribute_getDtype, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaAttr_BoolType, isList, arginfo_cuda_param_attribute_isList, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

void cuda_param_attribute_init(void)
{
    zend_class_entry ce;
    
    INIT_CLASS_ENTRY(ce, "Cuda\\ParamAttribute", cuda_param_attribute_methods);
    cuda_param_attribute_ce = zend_register_internal_class(&ce);
    cuda_param_attribute_ce->ce_flags |= ZEND_ACC_EXPLICIT_ABSTRACT_CLASS;
    
    zend_string *attr_name = zend_string_init("Attribute", strlen("Attribute"), 1);
    zend_attribute *attr = zend_add_class_attribute(cuda_param_attribute_ce, attr_name, 1);
    ZVAL_LONG(&attr->args[0].value, ZEND_ATTRIBUTE_TARGET_PARAMETER);
    zend_string_release(attr_name);
}

void cuda_register_attributes(void)
{
    zend_class_entry ce;
    zend_string *attr_name;
    zend_attribute *attr;
    
    // Tensor
    INIT_CLASS_ENTRY(ce, "Cuda\\Attr\\TensorType", cuda_tensor_methods);
    zend_class_entry *tensor_ce = zend_register_internal_class_ex(&ce, cuda_param_attribute_ce);
    
    attr_name = zend_string_init("Attribute", strlen("Attribute"), 1);
    attr = zend_add_class_attribute(tensor_ce, attr_name, 0);
    zend_string_release(attr_name);
    
    // Int
    INIT_CLASS_ENTRY(ce, "Cuda\\Attr\\IntType", cuda_int_methods);
    zend_class_entry *int_ce = zend_register_internal_class_ex(&ce, cuda_param_attribute_ce);
    
    attr_name = zend_string_init("Attribute", strlen("Attribute"), 1);
    attr = zend_add_class_attribute(int_ce, attr_name, 0);
    zend_string_release(attr_name);
    
    // Float
    INIT_CLASS_ENTRY(ce, "Cuda\\Attr\\FloatType", cuda_float_methods);
    zend_class_entry *float_ce = zend_register_internal_class_ex(&ce, cuda_param_attribute_ce);
    
    attr_name = zend_string_init("Attribute", strlen("Attribute"), 1);
    attr = zend_add_class_attribute(float_ce, attr_name, 0);
    zend_string_release(attr_name);
    
    // Bool
    INIT_CLASS_ENTRY(ce, "Cuda\\Attr\\BoolType", cuda_bool_methods);
    zend_class_entry *bool_ce = zend_register_internal_class_ex(&ce, cuda_param_attribute_ce);
    
    attr_name = zend_string_init("Attribute", strlen("Attribute"), 1);
    attr = zend_add_class_attribute(bool_ce, attr_name, 0);
    zend_string_release(attr_name);
}