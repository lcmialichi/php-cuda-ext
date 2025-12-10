#include "kernel.h"
#include "ke_arginfo.h"
#include "php.h"
#include "zend_compile.h"
#include "zend_attributes.h"
#include "kernel_reflection.h"
#include "zend_string.h"
#include "zend_operators.h"
#include "zend_ast.h"
#include "ast_cuda_compiler.h"
#include "ext/standard/php_smart_string.h"
#include "kernel_types.h"
#include "compiler.h"
#include "device.h"


zend_class_entry *kernel_ce;
static zend_object_handlers kernel_handlers;

static void kernel_free_object(zend_object *object);
static zend_object *kernel_create_object(zend_class_entry *class_type);

ZEND_METHOD(Kernel, fn)
{
    zend_fcall_info fci;
    zend_fcall_info_cache fcc;
    zval *attributes = NULL;
    
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_FUNC(fci, fcc)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY(attributes)
    ZEND_PARSE_PARAMETERS_END();
    
    zend_string *name = NULL;
    zend_string *target = NULL;
    if (attributes && Z_TYPE_P(attributes) == IS_ARRAY) {
        zval *name_zv = zend_hash_str_find(Z_ARR_P(attributes), "name", sizeof("name")-1);
        if (name_zv && Z_TYPE_P(name_zv) == IS_STRING) {
            name = Z_STR_P(name_zv);
        }
        
        zval *target_zv = zend_hash_str_find(Z_ARR_P(attributes), "target", sizeof("target")-1);
        if (target_zv && Z_TYPE_P(target_zv) == IS_STRING) {
            target = Z_STR_P(target_zv);
        }
    }
    
    if (!name) {
        static int anonymous_counter = 0;
        char buffer[64];
        snprintf(buffer, sizeof(buffer), "anonymous_device_%d", anonymous_counter++);
        name = zend_string_init(buffer, strlen(buffer), 0);
    }
    
    if (!target) {
        target = zend_string_init("sm_60", 5, 0);
    }
    
    zend_class_entry *device_ce = zend_lookup_class(ZEND_STRL("Cuda\\Device"));
    zval device_zv;
    object_init_ex(&device_zv, device_ce);
    
    cuda_device_object *device = Z_CUDA_DEVICE_P(&device_zv);
    
    device->fci = fci;
    device->fcc = fcc;
    device->name = zend_string_copy(name);
    device->target = zend_string_copy(target);
    device->ast = NULL;
    device->ast_arena = NULL;
    device->attributes = attributes ? zend_array_dup(Z_ARR_P(attributes)) : NULL;
    
    zend_closure *closure = (zend_closure*)zend_fcall_info_get_function(&fci, &fcc);
    if (closure) {
        zend_string *source = NULL;
        int result = kernel_extract_closure_source(closure, &source);
        if (result == 1) {
            device->ast = zend_compile_string_to_ast(source, &device->ast_arena, name);
            zend_string_release(source);
        }
    }
    
    RETURN_ZVAL(&device_zv, 1, 0);
}

ZEND_METHOD(Kernel, __construct)
{
    php_printf("Note: Kernel compilation now done via Cuda\\Compiler\n");
}


static void kernel_free_object(zend_object *object)
{
    kernel_obj *obj = (kernel_obj *)((char *)object - XtOffsetOf(kernel_obj, obj));
    zend_object_std_dtor(&obj->obj);
}

static zend_object *kernel_create_object(zend_class_entry *class_type)
{
    kernel_obj *obj = (kernel_obj *)ecalloc(1, sizeof(kernel_obj));
    zend_object_std_init(&obj->obj, class_type);
    object_properties_init(&obj->obj, class_type);
    obj->obj.handlers = &kernel_handlers;
    return &obj->obj;
}

int kernel_init()
{
    zend_class_entry *kernel_ce_local = register_kernel_class();
    kernel_ce_local->create_object = kernel_create_object;

    memcpy(&kernel_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    kernel_handlers.offset   = XtOffsetOf(kernel_obj, obj);
    kernel_handlers.free_obj = kernel_free_object;

    return 1;
}