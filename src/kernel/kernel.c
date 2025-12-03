#include "kernel.h"
#include "ke_arginfo.h"
#include "php.h"
#include "tensor.h"
#include "kernel_fusion.h"
#include "cuda_globals.h"
#include "cuda_array.h"

zend_class_entry *kernel_ce;
static zend_object_handlers kernel_handlers;

static void kernel_free_object(zend_object *object);
static kernel_obj *php_kernel_fetch_object(zend_object *obj);
static zend_object *kernel_create_object(zend_class_entry *class_type);

tensor_t *convert_zend_object_to_tensor_t(zend_object *obj)
{
    cuda_array_obj *ca_obj = (cuda_array_obj *)((char *)obj - XtOffsetOf(cuda_array_obj, obj));
    if (!ca_obj || ca_obj->tensor_handle == NULL)
    {
        zend_error(E_ERROR, "Attempting to access uninitialized tensor!");
        return NULL;
    }

    return ca_obj->tensor_handle;
}

ZEND_METHOD(Kernel, fusion)
{
    zend_fcall_info fci;
    zend_fcall_info_cache fci_cache;

    zval retval;
    ZVAL_UNDEF(&retval);

    tensor_t *final_proxy_tensor = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_FUNC(fci, fci_cache)
    ZEND_PARSE_PARAMETERS_END();

    fci.retval = &retval;
    fci.param_count = 0;
    fci.params = NULL;

    start_kernel_fusions();
    int call_status = zend_call_function(&fci, &fci_cache);

    if (call_status == FAILURE)
    {
        if (!EG(exception))
        {
            zend_throw_error(NULL, "Failed to execute the fusion kernel callable (internal error).");
        }
        stop_kernel_fusions();
        ZVAL_NULL(return_value);
        return;
    }

    if (Z_TYPE(retval) != IS_OBJECT)
    {
        zend_throw_error(NULL, "Cuda\\Kernel::fusion callable must return a Cuda\\CudaArray object.");
        zval_ptr_dtor(&retval);
        stop_kernel_fusions();
        ZVAL_NULL(return_value);
        return;
    }

    final_proxy_tensor = convert_zend_object_to_tensor_t(Z_OBJ(retval));

    if (final_proxy_tensor == NULL)
    {
        zend_throw_error(NULL, "The returned object is not a valid Cuda tensor object.");
        zval_ptr_dtor(&retval);
        stop_kernel_fusions();
        ZVAL_NULL(return_value);
        return;
    }

    fusion_context_t *context = CUDA_G(current_fusion_context);

    if (context == NULL)
    {
        zend_throw_error(NULL, "Fusion context lost.");
        zval_ptr_dtor(&retval);
        stop_kernel_fusions();
        return;
    }

    compile_and_execute_fusion(final_proxy_tensor);
    stop_kernel_fusions();

    RETVAL_ZVAL(&retval, 0, 0);
}

static kernel_obj *php_kernel_fetch_object(zend_object *obj)
{
    return (kernel_obj *)((char *)obj - XtOffsetOf(kernel_obj, obj));
}

static void kernel_free_object(zend_object *object)
{
    kernel_obj *obj = php_kernel_fetch_object(object);
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
    zend_class_entry *kernel_ce = register_kernel_class();

    kernel_ce->create_object = kernel_create_object;

    memcpy(&kernel_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    kernel_handlers.offset = XtOffsetOf(kernel_obj, obj);
    kernel_handlers.free_obj = kernel_free_object;
    return 1;
}