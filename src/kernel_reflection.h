#ifndef KERNEL_REFLECTION_H
#define KERNEL_REFLECTION_H

#include "php.h"
#include "zend_compile.h"
#include "cuda_attributes.h"

typedef enum {
    AST_METHOD,
    AST_FUNCTION_CALL,
    AST_VARIABLE,
    AST_ASSIGNMENT,
    AST_BINARY_OP,
    AST_RETURN,
    AST_IF,
    AST_ARRAY_ACCESS,
    AST_LITERAL
} ast_node_type_t;

typedef struct ast_node {
    ast_node_type_t type;
    char *value;
    struct ast_node **children;
    int child_count;
    int line;
    int column;
} ast_node_t;

typedef struct method_info {
    char *name;
    char *cuda_name;
    char *visibility;
    char *return_type;
    ast_node_t *ast;
    zend_function *func;
    int is_kernel;
    char *target;
} method_info_t;

cuda_method_attribute_args *cuda_extract_method_attribute(
    zend_function *fptr,
    zend_class_entry *ce_attribute);

#endif