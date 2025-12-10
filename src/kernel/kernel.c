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
    kernel_handlers.offset = XtOffsetOf(kernel_obj, obj);
    kernel_handlers.free_obj = kernel_free_object;

    return 1;
}