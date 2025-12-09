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

zend_class_entry *kernel_ce;
static zend_object_handlers kernel_handlers;

extern zend_class_entry *cuda_attr_kernel_ce;
extern zend_class_entry *cuda_attr_device_ce;
extern cuda_method_attribute_args *cuda_extract_method_attribute(zend_function *fptr, zend_class_entry *ce_attribute);


zend_string *cuda_extract_method_source_code(zend_function *fptr) {
    const char *source = 
        "<?php \n"
        "if ($a > 10) { \n"
        "    return max($a * $b, 0.0); \n"
        "} \n"
        "$a[$index] = 10;\n"
        "return $b;";
    
    return zend_string_init(source, strlen(source), 0);
}

static void kernel_free_object(zend_object *object) {
    kernel_obj *obj = (kernel_obj *)((char *)object - XtOffsetOf(kernel_obj, obj));
    zend_object_std_dtor(&obj->obj);
}

static zend_object *kernel_create_object(zend_class_entry *class_type) {
    kernel_obj *obj = (kernel_obj *)ecalloc(1, sizeof(kernel_obj));
    zend_object_std_init(&obj->obj, class_type);
    object_properties_init(&obj->obj, class_type);
    obj->obj.handlers = &kernel_handlers;
    return &obj->obj;
}

ZEND_METHOD(Kernel, __construct)
{
    zend_class_entry *ce = Z_OBJCE_P(ZEND_THIS);
    zend_function *fptr;
    zend_string *method_name;

    php_printf("=== STARTING KERNEL COMPILATION for %s ===\n", ZSTR_VAL(ce->name));

    smart_string cuda_output_buffer = {0};
    
    func_parameter_list input_params = {0, NULL};
    func_parameter_list output_params = {0, NULL};
    int compilation_result = 1;

    ZEND_HASH_FOREACH_STR_KEY_PTR(&ce->function_table, method_name, fptr) {
        
        if (ZSTR_VAL(method_name)[0] == '_') {
            continue;
        }

        cuda_method_attribute_args *fargs = cuda_extract_method_attribute(fptr, cuda_attr_device_ce);

        if (fargs) {
            zend_string *method_source_code = NULL;
            zend_arena *ast_arena = NULL;
            zend_ast *root_ast = NULL;

            php_printf("\n-> Processing Kernel: %s (Target: %s)\n", ZSTR_VAL(fptr->common.function_name), ZSTR_VAL(fargs->target));

            method_source_code = cuda_extract_method_source_code(fptr);

            if (method_source_code) {
                
                root_ast = zend_compile_string_to_ast(
                    method_source_code, 
                    &ast_arena, 
                    fptr->common.function_name
                );

                if (root_ast) {
                    // Impressão mínima para mostrar que a AST foi criada
                    php_printf("    [SUCCESS] AST");
                    
                    // --- INÍCIO DA COMPILAÇÃO CUDA BASEADA EM AST ---
                    
                    smart_string_free(&cuda_output_buffer); 
                    
                    php_printf("\n*** START CUDA CODE GENERATION ***\n");
                    
                    // CHAMADA CRÍTICA: Executa o novo compilador AST/CUDA
                    compilation_result = compile_ast_as_valid_cuda(
                        &cuda_output_buffer, 
                        root_ast, 
                        &input_params, 
                        &output_params
                    );

                    if (compilation_result == 1) {
                        smart_string_0(&cuda_output_buffer); // Finaliza a string
                        php_printf("\n*** CUDA CODE OUTPUT ***\n");
                    
                        php_printf("%s", cuda_output_buffer.c);
                        php_printf("*** END CUDA CODE OUTPUT ***\n");
                    } else {
                        php_printf("\n*** CUDA CODE GENERATION FAILED ***\n");
                    }

                } else {
                    php_printf("    [FAILURE] Could not compile source to AST.\n");
                }

                if (ast_arena) {
                    zend_arena_destroy(ast_arena);
                }
                zend_string_release(method_source_code);
            }
        }

    } ZEND_HASH_FOREACH_END();
    
    smart_string_free(&cuda_output_buffer);

    php_printf("=== KERNEL COMPILATION ENDED ===\n");
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