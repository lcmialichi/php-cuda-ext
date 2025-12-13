#ifndef AST_CUDA_COMPILER_H
#define AST_CUDA_COMPILER_H

#include "php.h"
#include "zend_string.h"
#include "kernel_reflection.h"
#include "kernel_types.h"
#include "zend_ast.h"
#include "data_types.h"

typedef struct {
    const char *php_name;    
    const char *cuda_name_f32; 
    const char *cuda_name_f64;
    const char *cuda_name_i32;
    dtype_t return_type_f32;
    dtype_t return_type_f64; 
    dtype_t return_type_i32;
    uint32_t num_params;  
    dtype_t param_types_f32[4];
    dtype_t param_types_f64[4]; 
    dtype_t param_types_i32[4]; 
    const char *header; 
} cuda_function_info_t;

typedef int (*handler)(cuda_compilation_context_t *context, zend_ast *ast);

typedef struct
{
    zend_ast_kind kind;
    handler fn;
} php_ast_handler;

int compile_ast_as_valid_cuda(cuda_compilation_context_t *context, zend_ast *ast);
int compile_ast_to_cuda_fn(cuda_compilation_context_t *context, zend_ast *ast);
char *generate_cuda_headers(HashTable* cuda_headers);
void free_cuda_context(cuda_compilation_context_t *context);
cuda_compilation_context_t *create_cuda_context(func_parameter_list_t *parameters, cuda_fn_type fn_type, zend_string* name);

#endif