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
    zend_string *name;
    zend_fcall_info fci;
    zend_fcall_info_cache fcc;
    zval *attributes = NULL;
    
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_STR(name)
        Z_PARAM_FUNC(fci, fcc)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY(attributes)
    ZEND_PARSE_PARAMETERS_END();
    
    device = Z_CUDA_DEVICE_P(ZEND_THIS);
    
    // Inicializar device
    device->fci = fci;
    device->fcc = fcc;
    device->name = zend_string_copy(name);
    
    // Extrair target dos atributos
    zend_string *target = NULL;
    if (attributes && Z_TYPE_P(attributes) == IS_ARRAY) {
        zval *target_zv = zend_hash_str_find(Z_ARR_P(attributes), "target", sizeof("target")-1);
        if (target_zv && Z_TYPE_P(target_zv) == IS_STRING) {
            target = Z_STR_P(target_zv);
        }
        device->attributes = zend_array_dup(Z_ARR_P(attributes));
    } else {
        device->attributes = NULL;
    }
    
    device->target = target ? zend_string_copy(target) : zend_string_init("sm_60", 5, 0);
    device->ast = NULL;
    device->ast_arena = NULL;
    
    php_printf("Device function created: %s\n", ZSTR_VAL(name));
}

ZEND_METHOD(Device, compile)
{
    cuda_device_object *device = Z_CUDA_DEVICE_P(ZEND_THIS);
    char *target = NULL;
    size_t target_len;
    
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_STRING(target, target_len)
    ZEND_PARSE_PARAMETERS_END();
    
    php_printf("Compiling device function: %s\n", ZSTR_VAL(device->name));
    
    RETURN_ZVAL(getThis(), 1, 0);
}

ZEND_METHOD(Device, getName)
{
    cuda_device_object *device = Z_CUDA_DEVICE_P(ZEND_THIS);
    RETURN_STR(device->name);
}


ZEND_METHOD(Device, invoke)
{
    cuda_device_object *device = Z_CUDA_DEVICE_P(ZEND_THIS);
    
    zend_fcall_info fci = device->fci;
    zend_fcall_info_cache fcc = device->fcc;
    
    zval *args = NULL;
    int argc = 0;
    
    ZEND_PARSE_PARAMETERS_START(0, -1)
        Z_PARAM_VARIADIC('*', args, argc)
    ZEND_PARSE_PARAMETERS_END();
    
    fci.param_count = argc;
    fci.params = args;
    fci.retval = return_value;
    
    if (zend_call_function(&fci, &fcc) != SUCCESS) {
        php_error_docref(NULL, E_WARNING, "Failed to execute device function");
        RETURN_NULL();
    }
}

static void device_free_object(zend_object *object)
{
    cuda_device_object *device = Z_CUDA_DEVICE_FROM_OBJ(object);
    
    if (device->name) {
        zend_string_release(device->name);
    }
    
    if (device->target) {
        zend_string_release(device->target);
    }
    
    if (device->attributes) {
        zend_array_destroy(device->attributes);
    }
    
    if (device->ast_arena) {
        zend_arena_destroy(device->ast_arena);
    }
    
    zend_object_std_dtor(&device->std);
}

static zend_object *device_create_object(zend_class_entry *class_type)
{
    cuda_device_object *device = 
        (cuda_device_object*)ecalloc(1, sizeof(cuda_device_object));
    
    zend_object_std_init(&device->std, class_type);
    device->std.handlers = &device_handlers;
    
    device->name = NULL;
    device->target = NULL;
    device->attributes = NULL;
    device->ast = NULL;
    device->ast_arena = NULL;
    
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