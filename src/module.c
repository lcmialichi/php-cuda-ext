#include "module.h"
#include "php.h"
#include "zend_interfaces.h"
#include "zend_exceptions.h"
#include "module_arginfo.h"

zend_class_entry *cuda_module_ce;
static zend_object_handlers module_handlers;

static zend_object *module_create_object(zend_class_entry *class_type);
static void module_free_object(zend_object *object);

/* Método: CompiledModule->run() */
ZEND_METHOD(CompiledModule, run)
{
    zend_string *kernel_name;
    zval *args = NULL;
    int argc = 0;
    
    ZEND_PARSE_PARAMETERS_START(1, -1)
        Z_PARAM_STR(kernel_name)
        Z_PARAM_VARIADIC('*', args, argc)
    ZEND_PARSE_PARAMETERS_END();
    
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    
    // Verificar se o kernel existe
    zval *kernel_zv = zend_hash_find(module->kernel_functions, kernel_name);
    if (!kernel_zv) {
        zend_throw_exception_ex(NULL, 0, 
            "Kernel '%s' not found in compiled module", ZSTR_VAL(kernel_name));
        RETURN_NULL();
    }
    
    php_printf("Running kernel: %s with %d arguments\n", 
              ZSTR_VAL(kernel_name), argc);
    
    // TODO: Implementar execução real no CUDA
    // Por enquanto, apenas retorna null
    RETURN_NULL();
}

ZEND_METHOD(CompiledModule, hasKernel)
{
    zend_string *name;
    
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(name)
    ZEND_PARSE_PARAMETERS_END();
    
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    
    if (zend_hash_exists(module->kernel_functions, name)) {
        RETURN_TRUE;
    } else {
        RETURN_FALSE;
    }
}

ZEND_METHOD(CompiledModule, getKernels)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    
    array_init(return_value);
    
    zend_string *key;
    zval *val;
    
    ZEND_HASH_FOREACH_STR_KEY_VAL(module->kernel_functions, key, val) {
        add_next_index_string(return_value, ZSTR_VAL(key));
    } ZEND_HASH_FOREACH_END();
}

ZEND_METHOD(CompiledModule, getPtx)
{
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    
    if (module->ptx_code) {
        RETURN_STRING(module->ptx_code);
    } else {
        RETURN_NULL();
    }
}

ZEND_METHOD(CompiledModule, save)
{
    zend_string *filename;
    
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(filename)
    ZEND_PARSE_PARAMETERS_END();
    
    cuda_module_object *module = Z_CUDA_MODULE_P(ZEND_THIS);
    
    if (!module->ptx_code) {
        RETURN_FALSE;
    }
    
    FILE *file = fopen(ZSTR_VAL(filename), "w");
    if (!file) {
        php_error_docref(NULL, E_WARNING, "Failed to open file for writing: %s", 
                        ZSTR_VAL(filename));
        RETURN_FALSE;
    }
    
    size_t written = fwrite(module->ptx_code, 1, module->ptx_size, file);
    fclose(file);
    
    if (written == module->ptx_size) {
        RETURN_TRUE;
    } else {
        php_error_docref(NULL, E_WARNING, "Failed to write all bytes to file");
        RETURN_FALSE;
    }
}

static void module_free_object(zend_object *object)
{
    cuda_module_object *module = Z_CUDA_MODULE_FROM_OBJ(object);
    
    if (module->ptx_code) {
        efree(module->ptx_code);
    }
    
    if (module->functions) {
        zend_hash_destroy(module->functions);
        efree(module->functions);
    }
    
    if (module->kernel_functions) {
        zend_hash_destroy(module->kernel_functions);
        efree(module->kernel_functions);
    }
    
    zend_object_std_dtor(&module->std);
}

static zend_object *module_create_object(zend_class_entry *class_type)
{
    cuda_module_object *module = 
        (cuda_module_object*)ecalloc(1, sizeof(cuda_module_object));
    
    zend_object_std_init(&module->std, class_type);
    module->std.handlers = &module_handlers;
    
    module->ptx_code = NULL;
    module->ptx_size = 0;
    module->functions = NULL;
    module->kernel_functions = NULL;
    
    return &module->std;
}

int module_init()
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, MODULE_CLASS_NAME, module_methods);
    cuda_module_ce = zend_register_internal_class(&ce);
    cuda_module_ce->create_object = module_create_object;
    cuda_module_ce->ce_flags |= ZEND_ACC_FINAL;
    
    memcpy(&module_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    module_handlers.offset = XtOffsetOf(cuda_module_object, std);
    module_handlers.free_obj = module_free_object;
    
    return 1;
}