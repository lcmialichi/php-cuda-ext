#ifndef CUDA_CONVERTER_STRUCTURES_FINAL_H
#define CUDA_CONVERTER_STRUCTURES_FINAL_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h> 

// --- ENUMS DE TIPAGEM E METADADOS ---

typedef enum
{
    CUDA_TYPE_VOID,
    CUDA_TYPE_FLOAT,
    CUDA_TYPE_DOUBLE,
    CUDA_TYPE_INT32,
    CUDA_TYPE_INT64,
    CUDA_TYPE_BOOL,
    
    CUDA_TYPE_FLOAT_PTR,
    CUDA_TYPE_DOUBLE_PTR,
    CUDA_TYPE_INT32_PTR,
    CUDA_TYPE_INT64_PTR,
    
    CUDA_TYPE_CONST_FLOAT_PTR,

    CUDA_TYPE_UNKNOWN
} cuda_data_type_t;

typedef enum
{
    BLOCK_ASSIGNMENT,
    BLOCK_RETURN,
    BLOCK_VAR_DECL,
    BLOCK_IF,           // Inclui IF, ELSE IF e ELSE
    BLOCK_FOR,          // Loop FOR
    BLOCK_WHILE,        // Loop WHILE ou DO-WHILE
    BLOCK_EXPRESSION_ONLY // Chamadas de função com side effects
} block_type_t;

typedef enum {
    PARAM_USAGE_INPUT,
    PARAM_USAGE_OUTPUT,
    PARAM_USAGE_INOUT
} param_usage_t;

typedef enum {
    EXPR_LITERAL,
    EXPR_VARIABLE,
    EXPR_BINARY_OP,
    EXPR_ARRAY_FETCH,
    EXPR_FUNCTION_CALL
} expression_type_t;


typedef struct expression_t expression_t;
typedef struct code_block_t code_block_t;
typedef struct statement_t statement_t;
typedef struct parameter_t parameter_t;
typedef struct device_t device_t; 
typedef struct kernel_t kernel_t;
typedef struct kernel_file_t kernel_file_t;


/**
 * @brief Estrutura que representa uma chamada de função (interna ou de device).
 */
typedef struct {
    char *function_name_cuda;
    expression_t **arguments;
    size_t arg_count;
    bool is_device_call;
} function_call_t;

/**
 * @brief Representa uma expressão (o valor de uma operação ou variável).
 */
struct expression_t
{
    expression_type_t expr_type;
    cuda_data_type_t result_type;

    union {
        char *literal_value;
        char *variable_name;
        
        // EXPR_ARRAY_FETCH: $a[$idx]
        struct {
            char *array_name;
            expression_t *index_expr;
        } array_fetch;

        // EXPR_BINARY_OP: A op B
        struct {
            char *op_symbol;
            expression_t *left;
            expression_t *right;
        } binary_op;
        
        function_call_t *func_call;

    } data;
};


/**
 * @brief Representa um bloco de comandos (corpo de função, corpo de loop, if/else branch).
 */
struct code_block_t
{
    statement_t **statements;
    size_t count;
};

/**
 * @brief Estrutura de lista encadeada para IF/ELSE IF/ELSE.
 * O AST do PHP pode gerar uma cadeia de ifs, que mapeamos para esta lista.
 */
typedef struct conditional_branch_t {
    expression_t *condition;        // Condição (NULL para o bloco final ELSE)
    code_block_t *body;             // Corpo do código a ser executado
    struct conditional_branch_t *next_branch; // Próximo ELSE IF ou ELSE
} conditional_branch_t;


/**
 * @brief Representa um único comando ou declaração (statement) dentro de um bloco.
 */
struct statement_t
{
    block_type_t type;
    
    union {
        // BLOCK_ASSIGNMENT: LHS = RHS;
        struct {
            expression_t *lhs;
            expression_t *rhs;
        } assignment;
        
        // BLOCK_RETURN: return expr;
        struct {
            expression_t *expression;
        } return_stmt;

        // BLOCK_VAR_DECL: int var = init_val;
        struct {
            char name[64];
            cuda_data_type_t type;
            expression_t *initial_value;
        } var_decl;

        // BLOCK_IF: if (cond) { body } else if (cond2) { body2 } else { else_body }
        struct {
            conditional_branch_t *first_branch; // Início da cadeia IF/ELSE IF/ELSE
        } if_else_chain;
        
        // BLOCK_FOR: for (init; cond; step) { body }
        struct {
            statement_t *init;
            expression_t *condition;
            statement_t *step;
            code_block_t *body;
        } loop_for;

        // BLOCK_WHILE: while (cond) { body } ou do { body } while (cond);
        struct {
            expression_t *condition;
            code_block_t *body;
            bool is_do_while; // Para distinguir 'while' de 'do-while'
        } loop_while;
        
        // BLOCK_EXPRESSION_ONLY
        struct {
            expression_t *expression;
        } expression_only;

    } data;
};


// --- ESTRUTURAS DE FUNÇÃO E PARÂMETROS ---

/**
 * @brief Representa um parâmetro (Input/Output/Inout).
 */
struct parameter_t
{
    char name[64];
    cuda_data_type_t type;
    param_usage_t usage;
};

/**
 * @brief Representa um método auxiliar (__device__).
 */
struct device_t
{
    char name[64];
    char target_arch[16];
    cuda_data_type_t return_type;
    
    parameter_t **parameters;
    size_t param_count;
    
    code_block_t *body;
};

/**
 * @brief Representa um Kernel (__global__).
 */
struct kernel_t
{
    char name[64];
    char target_arch[16];
    
    parameter_t **parameters;
    size_t param_count;

    code_block_t *body;
};


// --- ESTRUTURA PRINCIPAL (ARQUIVO .CU) ---

/**
 * @brief Representa o arquivo CUDA completo a ser gerado.
 */
struct kernel_file_t
{
    char class_name[64];
    
    kernel_t **kernels;
    size_t kernels_count;
    
    device_t **devices;
    size_t devices_count;
};

#endif // CUDA_CONVERTER_STRUCTURES_FINAL_H