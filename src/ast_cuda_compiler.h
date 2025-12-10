#ifndef AST_CUDA_COMPILER_H
#define AST_CUDA_COMPILER_H

#include "php.h"
#include "zend_string.h"
#include "zend_ast.h"
#include "kernel_reflection.h"
#include "data_types.h"
#include "kernel_types.h"

typedef struct
{
    zend_string *name;
    dtype_t dtype;

} local_variable_t;

typedef struct
{
    func_parameter_list_t *parameters;
    HashTable local_variables;
    smart_string *cuda_code_buffer;
    dtype_t last_evaluated_dtype;
    dtype_t return_dtype;
    int loop_depth;
} cuda_compilation_context_t;

typedef int (*handler)(cuda_compilation_context_t *context, zend_ast *ast);

typedef struct
{
    zend_ast_kind kind;
    handler fn;
} php_ast_handler;

int compile_ast_as_valid_cuda(cuda_compilation_context_t *context, zend_ast *ast);
void init_cuda_headers();
char *generate_cuda_headers();
void free_cuda_context(cuda_compilation_context_t *context);
cuda_compilation_context_t *create_cuda_context(func_parameter_list_t *parameters);

#endif