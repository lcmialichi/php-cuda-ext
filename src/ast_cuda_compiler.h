#ifndef AST_CUDA_COMPILER_H
#define AST_CUDA_COMPILER_H

#include "php.h"
#include "zend_string.h"
#include "kernel_reflection.h"
#include "kernel_types.h"
#include "zend_ast.h"
#include "data_types.h"
#include "ast_cuda_types.h"

typedef struct
{
    uint32_t dimensions;
    uint32_t sizes[4];
    dtype_t element_type;
} array_info_t;

typedef struct
{
    char *name;
    dtype_t dtype;
    dtype_t element_dtype;
    int array_size;
    int is_dynamic;
    int declared_in_code;
} shared_memory_var_t;

typedef int (*handler)(cuda_compilation_context_t *context, zend_ast *ast);

typedef struct
{
    zend_ast_kind kind;
    handler fn;
} php_ast_handler;

char *generate_device_functions();
char *generate_cuda_headers(HashTable *cuda_headers);

int compile_ast_as_valid_cuda(cuda_compilation_context_t *context, zend_ast *ast);
int compile_ast_to_cuda_fn(cuda_compilation_context_t *context, zend_ast *ast);
void free_cuda_context(cuda_compilation_context_t *context);
cuda_compilation_context_t *create_cuda_context(
    func_parameter_list_t *parameters,
    cuda_fn_type fn_type,
    zend_string *name,
    HashTable *headers);

#endif