#ifndef AST_CUDA_COMPILER_H
#define AST_CUDA_COMPILER_H

#include "php.h"
#include "zend_compile.h"
#include "zend_string.h"
#include "zend_operators.h"
#include "zend_ast.h"
#include "zend_compile.h"
#include "zend_attributes.h"
#include "kernel_reflection.h"

typedef struct
{
    char name[32];
    int dtype;
} func_parameter;

typedef struct
{   
    int total;
    func_parameter **parameters;
} func_parameter_list;

typedef int (*handler)(smart_string *cuda_code, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput);

typedef struct
{
    zend_ast_kind kind;
    handler fn;
} php_ast_handler;

int compile_ast_as_valid_cuda(smart_string *cuda_code, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput);

#endif