#include "compiler.h"
#include "compiler_arginfo.h"
#include "php.h"
#include "zend_interfaces.h"
#include "zend_exceptions.h"
#include "zend_closures.h"
#include "kernel_reflection.h"
#include "ast_cuda_compiler.h"

zend_class_entry *cuda_compiler_ce;
static zend_object_handlers compiler_handlers;

static void compiler_free_object(zend_object *object);
static zend_object *compiler_create_object(zend_class_entry *class_type);

ZEND_METHOD(Compiler, __construct)
{
    cuda_compiler_object *compiler;
    zend_string *target_str = NULL;
    zend_long optimization = 3;
    zend_bool debug = 0;
    zend_bool fast_math = 1;

    ZEND_PARSE_PARAMETERS_START(0, 4)
    Z_PARAM_OPTIONAL
    Z_PARAM_STR_OR_NULL(target_str)
    Z_PARAM_LONG(optimization)
    Z_PARAM_BOOL(debug)
    Z_PARAM_BOOL(fast_math)
    ZEND_PARSE_PARAMETERS_END();

    compiler = Z_CUDA_COMPILER_P(ZEND_THIS);

    compiler->kernels = emalloc(sizeof(HashTable));
    zend_hash_init(compiler->kernels, 8, NULL, NULL, 0);

    compiler->devices = emalloc(sizeof(HashTable));
    zend_hash_init(compiler->devices, 8, NULL, NULL, 0);

    if (target_str)
    {
        compiler->target_device = estrndup(ZSTR_VAL(target_str), ZSTR_LEN(target_str));
    }
    else
    {
        compiler->target_device = estrdup("sm_60");
    }

    compiler->optimization_level = optimization;
    compiler->debug_mode = debug;
    compiler->fast_math = fast_math;
    compiler->compilation_context = NULL;
}

PHP_METHOD(Compiler, kernel)
{
    cuda_compiler_object *compiler;
    zend_fcall_info fci;
    zend_fcall_info_cache fcc;
    zval *attributes = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 2)
    Z_PARAM_FUNC(fci, fcc)
    Z_PARAM_OPTIONAL
    Z_PARAM_ARRAY(attributes)
    ZEND_PARSE_PARAMETERS_END();

    compiler = Z_CUDA_COMPILER_P(ZEND_THIS);

    zend_string *name = NULL;
    zend_string *target = NULL;
    zval *grid_zv = NULL;
    zval *block_zv = NULL;

    if (attributes && Z_TYPE_P(attributes) == IS_ARRAY)
    {
        zval *name_zv = zend_hash_str_find(Z_ARR_P(attributes), "name", sizeof("name") - 1);
        if (name_zv && Z_TYPE_P(name_zv) == IS_STRING)
        {
            name = Z_STR_P(name_zv);
        }

        zval *target_zv = zend_hash_str_find(Z_ARR_P(attributes), "target", sizeof("target") - 1);
        if (target_zv && Z_TYPE_P(target_zv) == IS_STRING)
        {
            target = Z_STR_P(target_zv);
        }

        grid_zv = zend_hash_str_find(Z_ARR_P(attributes), "grid", sizeof("grid") - 1);
        block_zv = zend_hash_str_find(Z_ARR_P(attributes), "block", sizeof("block") - 1);
    }

    if (!name)
    {
        static int kernel_counter = 0;
        char buffer[64];
        snprintf(buffer, sizeof(buffer), "kernel_%d", kernel_counter++);
        name = zend_string_init(buffer, strlen(buffer), 0);
    }
    else
    {
        name = zend_string_copy(name);
    }

    if (!target)
    {
        target = zend_string_init(compiler->target_device, strlen(compiler->target_device), 0);
    }
    else
    {
        target = zend_string_copy(target);
    }

    cuda_kernel_object *kernel =
        (cuda_kernel_object *)ecalloc(1, sizeof(cuda_kernel_object));

    zend_object_std_init(&kernel->std, NULL);
    kernel->std.handlers = zend_get_std_object_handlers();

    kernel->fci = fci;
    kernel->fcc = fcc;
    kernel->name = name;
    kernel->target = target;

    kernel->grid[0] = 1;
    kernel->grid[1] = 1;
    kernel->grid[2] = 1;
    kernel->block[0] = 256;
    kernel->block[1] = 1;
    kernel->block[2] = 1;

    if (grid_zv && Z_TYPE_P(grid_zv) == IS_ARRAY)
    {
        zval *x = zend_hash_index_find(Z_ARR_P(grid_zv), 0);
        zval *y = zend_hash_index_find(Z_ARR_P(grid_zv), 1);
        zval *z = zend_hash_index_find(Z_ARR_P(grid_zv), 2);

        if (x)
            kernel->grid[0] = zval_get_long(x);
        if (y)
            kernel->grid[1] = zval_get_long(y);
        if (z)
            kernel->grid[2] = zval_get_long(z);
    }

    if (block_zv && Z_TYPE_P(block_zv) == IS_ARRAY)
    {
        zval *x = zend_hash_index_find(Z_ARR_P(block_zv), 0);
        zval *y = zend_hash_index_find(Z_ARR_P(block_zv), 1);
        zval *z = zend_hash_index_find(Z_ARR_P(block_zv), 2);

        if (x)
            kernel->block[0] = zval_get_long(x);
        if (y)
            kernel->block[1] = zval_get_long(y);
        if (z)
            kernel->block[2] = zval_get_long(z);
    }

    kernel->ast = NULL;
    kernel->ast_arena = NULL;
    kernel->parameters = NULL;

    kernel->used_devices = emalloc(sizeof(HashTable));
    zend_hash_init(kernel->used_devices, 4, NULL, NULL, 0);

    php_printf("  AST extraction not yet implemented\n");

    zend_hash_str_add_ptr(compiler->kernels,
                          ZSTR_VAL(kernel->name),
                          ZSTR_LEN(kernel->name),
                          kernel);

    php_printf("Added kernel: %s (grid: [%d,%d,%d], block: [%d,%d,%d])\n",
               ZSTR_VAL(kernel->name),
               kernel->grid[0], kernel->grid[1], kernel->grid[2],
               kernel->block[0], kernel->block[1], kernel->block[2]);

    RETURN_ZVAL(getThis(), 1, 0);
}

ZEND_METHOD(Compiler, device)
{
    cuda_compiler_object *compiler;
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

    compiler = Z_CUDA_COMPILER_P(ZEND_THIS);

    zend_string *target = NULL;
    if (attributes && Z_TYPE_P(attributes) == IS_ARRAY)
    {
        zval *target_zv = zend_hash_str_find(Z_ARR_P(attributes), "target", sizeof("target") - 1);
        if (target_zv && Z_TYPE_P(target_zv) == IS_STRING)
        {
            target = Z_STR_P(target_zv);
        }
    }

    if (!target)
    {
        target = zend_string_init(compiler->target_device, strlen(compiler->target_device), 0);
    }
    else
    {
        target = zend_string_copy(target);
    }

    zend_string *device_class_name = zend_string_init("Cuda\\Device", strlen("Cuda\\Device"), 0);
    zend_class_entry *device_ce = zend_lookup_class(device_class_name);
    zend_string_release(device_class_name);

    if (!device_ce)
    {
        zend_throw_exception_ex(NULL, 0, "Class Cuda\\Device not found");
        RETURN_NULL();
    }

    zval device_zv;
    object_init_ex(&device_zv, device_ce);

    cuda_device_object *device = Z_CUDA_DEVICE_P(&device_zv);

    device->fci = fci;
    device->fcc = fcc;
    device->name = zend_string_copy(name);
    device->target = target;
    device->ast = NULL;
    device->ast_arena = NULL;
    device->attributes = attributes ? zend_array_dup(Z_ARR_P(attributes)) : NULL;

    zend_hash_add_ptr(compiler->devices, zend_string_copy(name), device);

    php_printf("Added device function: %s\n", ZSTR_VAL(name));

    RETURN_ZVAL(getThis(), 1, 0);
}

ZEND_METHOD(Compiler, compile)
{
    cuda_compiler_object *compiler;
    char *target = NULL;
    size_t target_len;
    zend_bool optimize = 1;
    zend_bool debug = 0;

    ZEND_PARSE_PARAMETERS_START(0, 3)
    Z_PARAM_OPTIONAL
    Z_PARAM_STRING(target, target_len)
    Z_PARAM_BOOL(optimize)
    Z_PARAM_BOOL(debug)
    ZEND_PARSE_PARAMETERS_END();

    compiler = Z_CUDA_COMPILER_P(ZEND_THIS);

    php_printf("=== COMPILING %d KERNELS AND %d DEVICES ===\n",
               zend_hash_num_elements(compiler->kernels),
               zend_hash_num_elements(compiler->devices));

    zend_string *module_class_name = zend_string_init("Cuda\\CompiledModule", strlen("Cuda\\CompiledModule"), 0);
    zend_class_entry *module_ce = zend_lookup_class(module_class_name);

    zval module_zv;
    object_init_ex(&module_zv, module_ce);

    cuda_module_object *module = Z_CUDA_MODULE_P(&module_zv);

    module->ptx_code = estrdup("// Placeholder PTX code\n.version 7.5\n.target sm_60");
    module->ptx_size = strlen(module->ptx_code);

    module->functions = emalloc(sizeof(HashTable));
    zend_hash_init(module->functions, 8, NULL, NULL, 0);

    module->kernel_functions = emalloc(sizeof(HashTable));
    zend_hash_init(module->kernel_functions, 8, NULL, NULL, 0);

    cuda_kernel_object *kernel;
    ZEND_HASH_FOREACH_PTR(compiler->kernels, kernel)
    {
        zval kernel_name;
        ZVAL_STR(&kernel_name, zend_string_copy(kernel->name));
        zend_hash_add(module->kernel_functions, kernel->name, &kernel_name);

        php_printf("  - Kernel: %s\n", ZSTR_VAL(kernel->name));
    }
    ZEND_HASH_FOREACH_END();

    php_printf("Compilation completed (mock implementation)\n");

    RETURN_ZVAL(&module_zv, 1, 0);
}

ZEND_METHOD(Compiler, getKernels)
{
    cuda_compiler_object *compiler = Z_CUDA_COMPILER_P(ZEND_THIS);

    array_init(return_value);

    cuda_kernel_object *kernel;
    ZEND_HASH_FOREACH_PTR(compiler->kernels, kernel)
    {
        zval kernel_zv;
        ZVAL_OBJ(&kernel_zv, &kernel->std);
        zend_hash_add(return_value->value.arr, kernel->name, &kernel_zv);
        Z_ADDREF(kernel_zv);
    }
    ZEND_HASH_FOREACH_END();
}

ZEND_METHOD(Compiler, getDevices)
{
    cuda_compiler_object *compiler = Z_CUDA_COMPILER_P(ZEND_THIS);

    array_init(return_value);

    cuda_device_object *device;
    ZEND_HASH_FOREACH_PTR(compiler->devices, device)
    {
        zval device_zv;
        ZVAL_OBJ(&device_zv, &device->std);
        zend_hash_add(return_value->value.arr, device->name, &device_zv);
        Z_ADDREF(device_zv);
    }
    ZEND_HASH_FOREACH_END();
}

static void compiler_free_object(zend_object *object)
{
    cuda_compiler_object *compiler = Z_CUDA_COMPILER_FROM_OBJ(object);

    if (compiler->target_device)
    {
        efree(compiler->target_device);
    }

    if (compiler->kernels)
    {
        cuda_kernel_object *kernel;
        ZEND_HASH_FOREACH_PTR(compiler->kernels, kernel)
        {
            // Os objetos kernel serão destruídos pelo GC do PHP
        }
        ZEND_HASH_FOREACH_END();
        zend_hash_destroy(compiler->kernels);
        efree(compiler->kernels);
    }

    if (compiler->devices)
    {
        cuda_device_object *device;
        ZEND_HASH_FOREACH_PTR(compiler->devices, device)
        {
            // Os objetos device serão destruídos pelo GC do PHP
        }
        ZEND_HASH_FOREACH_END();
        zend_hash_destroy(compiler->devices);
        efree(compiler->devices);
    }

    if (compiler->compilation_context)
    {
        free_cuda_context(compiler->compilation_context);
    }

    zend_object_std_dtor(&compiler->std);
}

static zend_object *compiler_create_object(zend_class_entry *class_type)
{
    cuda_compiler_object *compiler =
        (cuda_compiler_object *)ecalloc(1, sizeof(cuda_compiler_object));

    zend_object_std_init(&compiler->std, class_type);
    compiler->std.handlers = &compiler_handlers;

    compiler->target_device = NULL;
    compiler->optimization_level = 3;
    compiler->debug_mode = 0;
    compiler->fast_math = 1;
    compiler->kernels = NULL;
    compiler->devices = NULL;
    compiler->compilation_context = NULL;

    return &compiler->std;
}

int compiler_init()
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, COMPILER_CLASS_NAME, compiler_methods);
    cuda_compiler_ce = zend_register_internal_class(&ce);
    cuda_compiler_ce->create_object = compiler_create_object;
    cuda_compiler_ce->ce_flags |= ZEND_ACC_FINAL;

    memcpy(&compiler_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    compiler_handlers.offset = XtOffsetOf(cuda_compiler_object, std);
    compiler_handlers.free_obj = compiler_free_object;

    return 1;
}