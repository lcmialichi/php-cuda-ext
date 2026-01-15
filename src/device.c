#include "device.h"
#include "php.h"
#include "zend_interfaces.h"
#include "zend_exceptions.h"
#include "kernel_reflection.h"
#include "device_arginfo.h"

zend_class_entry *cuda_device_ce;
static zend_object_handlers device_handlers;

static zend_object *device_create_object(zend_class_entry *class_type);
static void device_free_object(zend_object *object);

ZEND_METHOD(Device, __construct)
{
    cuda_device_object *device;
    zend_fcall_info fci;
    zend_fcall_info_cache fcc;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_FUNC(fci, fcc)
    ZEND_PARSE_PARAMETERS_END();

    device = Z_CUDA_DEVICE_P(ZEND_THIS);

    zend_function *fptr = fcc.function_handler;
    if (!fptr || fptr->type != ZEND_USER_FUNCTION)
    {
        return;
    }

    cuda_method_attribute_args *fargs = cuda_extract_method_attribute(fptr, cuda_attr_device_ce);
    if (!fargs)
    {
        return;
    }

    device->name = zend_string_copy(fargs->name);

    zend_string_release(fargs->name);
    efree(fargs);
}

ZEND_METHOD(Device, fn)
{
    zend_fcall_info fci;
    zend_fcall_info_cache fcc;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_FUNC(fci, fcc)
    ZEND_PARSE_PARAMETERS_END();

    zend_function *fptr = fcc.function_handler;
    if (!fptr || fptr->type != ZEND_USER_FUNCTION)
    {
        return;
    }

    cuda_method_attribute_args *fargs = cuda_extract_method_attribute(fptr, cuda_attr_device_ce);
    if (!fargs)
    {
        return;
    }

    zend_string *device_class_name = zend_string_init("Cuda\\Device", strlen("Cuda\\Device"), 0);
    zend_class_entry *device_ce = zend_lookup_class(device_class_name);
    zval device_zv;
    object_init_ex(&device_zv, device_ce);

    cuda_device_object *device = Z_CUDA_DEVICE_P(&device_zv);

    device->name = zend_string_copy(fargs->name);

    zend_string_release(fargs->name);
    efree(fargs);

    RETURN_ZVAL(&device_zv, 1, 0);
}

ZEND_METHOD(Device, getName)
{
    cuda_device_object *device = Z_CUDA_DEVICE_P(ZEND_THIS);
    RETURN_STR(device->name);
}


static void device_free_object(zend_object *object)
{
    cuda_device_object *device = Z_CUDA_DEVICE_FROM_OBJ(object);

    if (device->name)
    {
        zend_string_release(device->name);
    }

    if (device->target)
    {
        zend_string_release(device->target);
    }

    zend_object_std_dtor(&device->std);
}

static zend_object *device_create_object(zend_class_entry *class_type)
{
    cuda_device_object *device =
        (cuda_device_object *)ecalloc(1, sizeof(cuda_device_object));

    zend_object_std_init(&device->std, class_type);
    device->std.handlers = &device_handlers;

    device->name = NULL;
    device->target = NULL;

    return &device->std;
}

int device_init()
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, DEVICE_CLASS_NAME, device_methods);
    cuda_device_ce = zend_register_internal_class(&ce);
    cuda_device_ce->create_object = device_create_object;
    cuda_device_ce->ce_flags |= ZEND_ACC_FINAL;

    memcpy(&device_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    device_handlers.offset = XtOffsetOf(cuda_device_object, std);
    device_handlers.free_obj = device_free_object;

    return 1;
}