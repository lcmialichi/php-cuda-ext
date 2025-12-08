#include "cuda_attributes.h"
#include "php.h"
#include "zend_attributes.h"
#include "cuda_attributes_arginfo.h"

zend_class_entry *cuda_attr_kernel_ce;
zend_class_entry *cuda_attr_device_ce;
zend_class_entry *cuda_attr_input_ce;
zend_class_entry *cuda_attr_output_ce;

PHP_METHOD(CudaAttr_Method, __construct)
{
	zend_string *name_zs, *target_zs = NULL;

	ZEND_PARSE_PARAMETERS_START(1, 2)
	Z_PARAM_STR(name_zs)
	Z_PARAM_OPTIONAL
	Z_PARAM_STR(target_zs)
	ZEND_PARSE_PARAMETERS_END();

	if (!target_zs)
	{
		target_zs = zend_string_init(ZEND_STRL("sm_60"), 0);
	}
	else
	{
		target_zs = zend_string_copy(target_zs);
	}
	name_zs = zend_string_copy(name_zs);

	zend_update_property_str(NULL, Z_OBJ_P(getThis()), ZEND_STRL("name"), name_zs);

	zend_update_property_str(NULL, Z_OBJ_P(getThis()), ZEND_STRL("target"), target_zs);
}

PHP_METHOD(CudaAttr_Param, __construct)
{
	zend_string *name_zs, *target_zs = NULL;

	ZEND_PARSE_PARAMETERS_START(1, 2)
	Z_PARAM_STR(name_zs)
	Z_PARAM_OPTIONAL
	Z_PARAM_STR(target_zs)
	ZEND_PARSE_PARAMETERS_END();

	if (!target_zs)
	{
		target_zs = zend_string_init(ZEND_STRL("sm_60"), 0);
	}
	else
	{
		target_zs = zend_string_copy(target_zs);
	}
	name_zs = zend_string_copy(name_zs);

	zend_update_property_str(NULL, Z_OBJ_P(getThis()), ZEND_STRL("name"), name_zs);

	zend_update_property_str(NULL, Z_OBJ_P(getThis()), ZEND_STRL("target"), target_zs);
}

const zend_function_entry cuda_method_attribute_methods[] = {
	PHP_ME(CudaAttr_Method, __construct, arginfo_cuda_attr_method___construct, ZEND_ACC_PUBLIC)
		PHP_FE_END};

const zend_function_entry cuda_param_attribute_methods[] = {
	PHP_ME(CudaAttr_Param, __construct, arginfo_cuda_attr_param___construct, ZEND_ACC_PUBLIC)
		PHP_FE_END};



void cuda_attr_init()
{
    zend_class_entry ce, *class_entry;
    zend_string *attr_name;
    zend_attribute *attr;

    //
    // #[Cuda\Attr\Kernel]
    //
    INIT_CLASS_ENTRY(ce, "Cuda\\Attr\\Kernel", cuda_method_attribute_methods);
    cuda_attr_kernel_ce = zend_register_internal_class(&ce);

    attr_name = zend_string_init_interned("Attribute", sizeof("Attribute") - 1, true);
    attr = zend_add_class_attribute(cuda_attr_kernel_ce, attr_name, 1);
    zend_string_release_ex(attr_name, true);
    ZVAL_LONG(&attr->args[0].value, ZEND_ATTRIBUTE_TARGET_METHOD);


    //
    // #[Cuda\Attr\Device]
    //
    INIT_CLASS_ENTRY(ce, "Cuda\\Attr\\Device", cuda_method_attribute_methods);
    cuda_attr_device_ce = zend_register_internal_class(&ce);

    attr_name = zend_string_init_interned("Attribute", sizeof("Attribute") - 1, true);
    attr = zend_add_class_attribute(cuda_attr_device_ce, attr_name, 1);
    zend_string_release_ex(attr_name, true);
    ZVAL_LONG(&attr->args[0].value, ZEND_ATTRIBUTE_TARGET_METHOD);


    //
    // #[Cuda\Attr\Input]
    //
    INIT_CLASS_ENTRY(ce, "Cuda\\Attr\\Input", cuda_param_attribute_methods);
    cuda_attr_input_ce = zend_register_internal_class(&ce);

    attr_name = zend_string_init_interned("Attribute", sizeof("Attribute") - 1, true);
    attr = zend_add_class_attribute(cuda_attr_input_ce, attr_name, 1);
    zend_string_release_ex(attr_name, true);
    ZVAL_LONG(&attr->args[0].value, ZEND_ATTRIBUTE_TARGET_PARAMETER);


    //
    // #[Cuda\Attr\Output]
    //
    INIT_CLASS_ENTRY(ce, "Cuda\\Attr\\Output", cuda_param_attribute_methods);
    cuda_attr_output_ce = zend_register_internal_class(&ce);

    attr_name = zend_string_init_interned("Attribute", sizeof("Attribute") - 1, true);
    attr = zend_add_class_attribute(cuda_attr_output_ce, attr_name, 1);
    zend_string_release_ex(attr_name, true);
    ZVAL_LONG(&attr->args[0].value, ZEND_ATTRIBUTE_TARGET_PARAMETER);
}
