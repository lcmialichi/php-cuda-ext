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

typedef struct {
    const char *php_name;       // Nome que o usuário usa (ex: "max")
    const char *cuda_name_f32;  // Nome CUDA para float (ex: "fmaxf")
    const char *cuda_name_f64;  // Nome CUDA para double (ex: "fmax")
    const char *cuda_name_i32;  // Nome CUDA para int (ex: "abs")
    dtype_t return_type_f32;    // Tipo de retorno para float
    dtype_t return_type_f64;    // Tipo de retorno para double
    dtype_t return_type_i32;    // Tipo de retorno para int
    uint32_t num_params;        // Número de parâmetros
    dtype_t param_types_f32[4]; // Tipos dos parâmetros para float
    dtype_t param_types_f64[4]; // Tipos dos parâmetros para double
    dtype_t param_types_i32[4]; // Tipos dos parâmetros para int
    const char *header;         // Header necessário
} cuda_function_info_t;

typedef struct {
    const char *cuda_name;
    dtype_t return_type;
} cuda_function_match_t;

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