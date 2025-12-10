// compiler.c - Novo arquivo para a classe Compiler

#include "php.h"
#include "zend_interfaces.h"
#include "zend_exceptions.h"
#include "kernel_types.h"
#include "kernel_reflection.h"
#include "ast_cuda_compiler.h"

zend_class_entry *cuda_compiler_ce;
static zend_object_handlers compiler_handlers;

PHP_METHOD(Compiler, __construct)
{
    cuda_compiler_object *compiler;
    zend_string *target = NULL;
    zend_long optimization = 3;
    zend_bool debug = 0;
    zend_bool fast_math = 1;
    
    ZEND_PARSE_PARAMETERS_START(0, 4)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR(target)
        Z_PARAM_LONG(optimization)
        Z_PARAM_BOOL(debug)
        Z_PARAM_BOOL(fast_math)
    ZEND_PARSE_PARAMETERS_END();
    
    compiler = Z_CUDA_COMPILER_P(ZEND_THIS);
    
    compiler->kernels = emalloc(sizeof(HashTable));
    zend_hash_init(compiler->kernels, 8, NULL, NULL, 0);
    
    compiler->devices = emalloc(sizeof(HashTable));
    zend_hash_init(compiler->devices, 8, NULL, NULL, 0);
    
    if (target) {
        compiler->target_device = estrndup(ZSTR_VAL(target), ZSTR_LEN(target));
    } else {
        compiler->target_device = estrdup("sm_60");
    }
    
    compiler->optimization_level = optimization;
    compiler->debug_mode = debug;
    compiler->fast_math = fast_math;
    compiler->compilation_context = NULL;
    
    php_printf("CUDA Compiler initialized (target: %s)\n", compiler->target_device);
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
    
    php_printf("Adding kernel to compiler...\n");
    
    // 1. Extrai atributos #[Kernel]
    cuda_method_attribute_args *kernel_attrs = 
        cuda_extract_kernel_attributes(attributes);
    
    if (!kernel_attrs) {
        zend_throw_exception_ex(NULL, 0, 
            "Missing #[Cuda\\Kernel] attribute on closure");
        RETURN_NULL();
    }
    
    cuda_kernel_object *kernel = 
        emalloc(sizeof(cuda_kernel_object));
    
    zend_object_std_init(&kernel->std, cuda_kernel_ce);
    kernel->std.handlers = &kernel_handlers;
    
    kernel->fci = fci;
    kernel->fcc = fcc;
    
    kernel->name = zend_string_copy(kernel_attrs->name);
    kernel->target = zend_string_copy(kernel_attrs->target);
    
    // Extrai grid/block dos atributos
    if (kernel_attrs->grid) {
        // TODO: Parse array [x, y, z]
    }
    
    
    zend_closure *closure = 
        (zend_closure*)zend_fcall_info_get_function(&fci, &fcc);
    
    kernel_result_t r = kernel_extract_closure_source(closure, &kernel->source);
    if (r.status != KERNEL_OK) {
        zend_throw_exception_ex(NULL, 0, 
            "Failed to extract kernel source: %s", r.msg);
        RETURN_NULL();
    }
    
    kernel->ast = zend_compile_string_to_ast(
        kernel->source,
        &kernel->ast_arena,
        kernel->name
    );
    
    if (!kernel->ast) {
        zend_throw_exception_ex(NULL, 0, "Failed to generate AST for kernel");
        RETURN_NULL();
    }
    
    kernel->used_devices = emalloc(sizeof(HashTable));
    zend_hash_init(kernel->used_devices, 4, NULL, NULL, 0);
    
    kernel_analyze_closure_use_vars(closure, kernel->used_devices);
    
    zend_hash_str_add_ptr(
        compiler->kernels,
        ZSTR_VAL(kernel->name),
        ZSTR_LEN(kernel->name),
        kernel
    );
    
    php_printf("Kernel '%s' added successfully\n", ZSTR_VAL(kernel->name));
    
    RETURN_OBJ(&kernel->std);
}

PHP_METHOD(Compiler, compile)
{
    cuda_compiler_object *compiler;
    char *target = NULL;
    size_t target_len;
    zend_bool optimize = 1;
    
    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_STRING(target, target_len)
        Z_PARAM_BOOL(optimize)
    ZEND_PARSE_PARAMETERS_END();
    
    compiler = Z_CUDA_COMPILER_P(ZEND_THIS);
    
    php_printf("=== COMPILING %d KERNELS ===\n", 
              zend_hash_num_elements(compiler->kernels));
    
    compiler->compilation_context = create_cuda_context(NULL);
    
    smart_string cuda_code = {0};
    smart_string_alloc(&cuda_code, 8192, 0);
    
    smart_string_appends(&cuda_code, 
        "#include <cuda_runtime.h>\n"
        "#include <device_launch_parameters.h>\n"
        "#include <cuda_fp16.h>\n\n"
    );
    
    php_printf("Compiling device functions...\n");
    
    cuda_device_object *device;
    ZEND_HASH_FOREACH_PTR(compiler->devices, device) {
        compile_device_to_cuda(device, &cuda_code);
    } ZEND_HASH_FOREACH_END();
    
    php_printf("Compiling kernels...\n");
    
    cuda_kernel_object *kernel;
    ZEND_HASH_FOREACH_PTR(compiler->kernels, kernel) {
        compiler->compilation_context->parameters = kernel->parameters;
        
        smart_string_appends(&cuda_code, 
            "\n// ========================================\n"
            "// Kernel: ");
        smart_string_appends(&cuda_code, ZSTR_VAL(kernel->name));
        smart_string_appends(&cuda_code, "\n// ========================================\n\n");
        
        int result = compile_ast_as_valid_cuda(
            compiler->compilation_context,
            kernel->ast
        );
        
        if (result == 1) {
            smart_string_appendl(
                &cuda_code,
                compiler->compilation_context->cuda_code_buffer->c,
                compiler->compilation_context->cuda_code_buffer->len
            );
        } else {
            php_printf("Failed to compile kernel '%s'\n", ZSTR_VAL(kernel->name));
        }
        
        smart_string_free(compiler->compilation_context->cuda_code_buffer);
        compiler->compilation_context->cuda_code_buffer = NULL;
    } ZEND_HASH_FOREACH_END();
    
    smart_string_0(&cuda_code);
    
    php_printf("\n=== GENERATED CUDA CODE (%zu bytes) ===\n%s\n", 
              cuda_code.len, cuda_code.c);
    
    php_printf("Calling nvcc...\n");
    
    char *ptx_code = compile_with_nvcc(
        cuda_code.c,
        target ? target : compiler->target_device,
        optimize
    );
    
    cuda_module_object *module = create_module_object(ptx_code);
    
    smart_string_free(&cuda_code);
    if (ptx_code) efree(ptx_code);
    
    php_printf("Compilation completed successfully\n");
    
    RETURN_OBJ(&module->std);
}

static char *compile_with_nvcc(
    const char *cuda_source,
    const char *target,
    zend_bool optimize
) {
    const char *mock_ptx = 
        ".version 7.5\n"
        ".target sm_60\n"
        ".address_size 64\n\n"
        ".visible .entry kernel_func(\n"
        ") {\n"
        "}\n";
    
    return estrdup(mock_ptx);
}

static cuda_module_object *create_module_object(const char *ptx_code)
{
    cuda_module_object *module = 
        (cuda_module_object*)ecalloc(1, sizeof(cuda_module_object));
    
    zend_object_std_init(&module->std, cuda_module_ce);
    module->std.handlers = &module_handlers;
    
    module->ptx_code = estrdup(ptx_code);
    module->ptx_size = strlen(ptx_code);
    
    module->functions = emalloc(sizeof(HashTable));
    zend_hash_init(module->functions, 4, NULL, NULL, 0);
    
    module->kernel_functions = emalloc(sizeof(HashTable));
    zend_hash_init(module->kernel_functions, 4, NULL, NULL, 0);
    
    return module;
}