#include "ast_cuda_compiler.h"
#include "ext/standard/php_smart_string.h"
#include "ext/standard/php_string.h"
#include "zend_operators.h"
#include "zend_compile.h"
#include "zend_string.h"
#include "zend_operators.h"
#include "zend_ast.h"
#include "zend_compile.h"
#include "zend_attributes.h"
#include "kernel_reflection.h"

static int handle_not_allowed(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput);
static int handle_ast_stmt_list(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput);
static int handler_ast_if(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput);
static int handle_ast_if_elem(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput);
static int handler_ast_zval(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput);
static int handler_ast_var(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput);
static int handler_ast_return(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput);
static int handler_ast_call(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput);
static int handler_ast_binary_op(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput);
static int handler_ast_comp_op(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput);
static int handler_ast_list_container(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput);
static int handler_ast_allowed_simple(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput);

php_ast_handler php_ast_handlers[] = {
    {ZEND_AST_ZVAL, handler_ast_zval},
    {ZEND_AST_CONSTANT, handler_ast_allowed_simple},
    // {ZEND_AST_OP_ARRAY, handle_not_allowed},
    {ZEND_AST_ZNODE, handle_not_allowed},

    // DECLARATION NODES (not allowed)
    {ZEND_AST_FUNC_DECL, handle_not_allowed},
    {ZEND_AST_CLOSURE, handle_not_allowed},
    {ZEND_AST_METHOD, handle_not_allowed},
    {ZEND_AST_CLASS, handle_not_allowed},
    {ZEND_AST_ARROW_FUNC, handle_not_allowed},
    // {ZEND_AST_PROPERTY_HOOK, handle_not_allowed},

    {ZEND_AST_ARG_LIST, handler_ast_list_container},
    {ZEND_AST_ARRAY, handle_not_allowed},
    {ZEND_AST_ENCAPS_LIST, handle_not_allowed},
    {ZEND_AST_EXPR_LIST, handler_ast_list_container},
    {ZEND_AST_STMT_LIST, handle_ast_stmt_list},
    {ZEND_AST_IF, handler_ast_if},

    {ZEND_AST_SWITCH_LIST, handle_not_allowed},
    {ZEND_AST_CATCH_LIST, handle_not_allowed},
    {ZEND_AST_PARAM_LIST, handle_not_allowed},
    {ZEND_AST_CLOSURE_USES, handle_not_allowed},
    {ZEND_AST_PROP_DECL, handle_not_allowed},
    {ZEND_AST_CONST_DECL, handle_not_allowed},
    {ZEND_AST_CLASS_CONST_DECL, handle_not_allowed},
    {ZEND_AST_NAME_LIST, handle_not_allowed},
    {ZEND_AST_TRAIT_ADAPTATIONS, handle_not_allowed},
    {ZEND_AST_USE, handle_not_allowed},
    {ZEND_AST_TYPE_UNION, handle_not_allowed},
    {ZEND_AST_TYPE_INTERSECTION, handle_not_allowed},
    {ZEND_AST_ATTRIBUTE_LIST, handle_not_allowed},
    {ZEND_AST_ATTRIBUTE_GROUP, handle_not_allowed},
    {ZEND_AST_MATCH_ARM_LIST, handle_not_allowed},
    {ZEND_AST_MODIFIER_LIST, handle_not_allowed},

    // 0 CHILD NODES
    {ZEND_AST_MAGIC_CONST, handle_not_allowed},
    {ZEND_AST_TYPE, handle_not_allowed},
    {ZEND_AST_CONSTANT_CLASS, handle_not_allowed},
    {ZEND_AST_CALLABLE_CONVERT, handle_not_allowed},

    // 1 CHILD NODES
    {ZEND_AST_VAR, handler_ast_var},
    {ZEND_AST_CONST, handler_ast_allowed_simple},
    {ZEND_AST_UNPACK, handle_not_allowed},
    {ZEND_AST_UNARY_PLUS, handler_ast_allowed_simple},
    {ZEND_AST_UNARY_MINUS, handler_ast_allowed_simple},
    {ZEND_AST_CAST, handler_ast_allowed_simple},
    // {ZEND_AST_CAST_VOID, handle_not_allowed},
    {ZEND_AST_EMPTY, handle_not_allowed},
    {ZEND_AST_ISSET, handle_not_allowed},
    {ZEND_AST_SILENCE, handle_not_allowed},
    {ZEND_AST_SHELL_EXEC, handle_not_allowed},
    {ZEND_AST_PRINT, handle_not_allowed},
    {ZEND_AST_INCLUDE_OR_EVAL, handle_not_allowed},
    {ZEND_AST_UNARY_OP, handler_ast_allowed_simple},
    {ZEND_AST_PRE_INC, handler_ast_allowed_simple},
    {ZEND_AST_PRE_DEC, handler_ast_allowed_simple},
    {ZEND_AST_POST_INC, handler_ast_allowed_simple},
    {ZEND_AST_POST_DEC, handler_ast_allowed_simple},
    {ZEND_AST_YIELD_FROM, handle_not_allowed},
    {ZEND_AST_CLASS_NAME, handle_not_allowed},
    {ZEND_AST_GLOBAL, handle_not_allowed},
    {ZEND_AST_UNSET, handle_not_allowed},
    {ZEND_AST_RETURN, handler_ast_return},
    {ZEND_AST_LABEL, handle_not_allowed},
    {ZEND_AST_REF, handle_not_allowed},
    {ZEND_AST_HALT_COMPILER, handle_not_allowed},
    {ZEND_AST_ECHO, handle_not_allowed},
    {ZEND_AST_THROW, handle_not_allowed},
    {ZEND_AST_GOTO, handle_not_allowed},
    {ZEND_AST_BREAK, handler_ast_allowed_simple},
    {ZEND_AST_CONTINUE, handler_ast_allowed_simple},
    // {ZEND_AST_PROPERTY_HOOK_SHORT_BODY, handle_not_allowed},

    {ZEND_AST_DIM, handler_ast_allowed_simple},
    {ZEND_AST_PROP, handle_not_allowed},
    {ZEND_AST_NULLSAFE_PROP, handle_not_allowed},
    {ZEND_AST_STATIC_PROP, handle_not_allowed},
    {ZEND_AST_CALL, handler_ast_call},
    {ZEND_AST_CLASS_CONST, handler_ast_allowed_simple},
    {ZEND_AST_ASSIGN, handler_ast_binary_op},
    {ZEND_AST_ASSIGN_REF, handle_not_allowed},
    {ZEND_AST_ASSIGN_OP, handler_ast_binary_op},
    {ZEND_AST_BINARY_OP, handler_ast_binary_op},
    {ZEND_AST_GREATER, handler_ast_comp_op},
    {ZEND_AST_GREATER_EQUAL, handler_ast_comp_op},
    {ZEND_AST_AND, handler_ast_comp_op},
    {ZEND_AST_OR, handler_ast_comp_op},
    {ZEND_AST_ARRAY_ELEM, handle_not_allowed},
    {ZEND_AST_NEW, handle_not_allowed},
    {ZEND_AST_INSTANCEOF, handle_not_allowed},
    {ZEND_AST_YIELD, handle_not_allowed},
    {ZEND_AST_COALESCE, handle_not_allowed},
    {ZEND_AST_ASSIGN_COALESCE, handle_not_allowed},
    {ZEND_AST_STATIC, handle_not_allowed},
    {ZEND_AST_WHILE, handler_ast_allowed_simple},
    {ZEND_AST_DO_WHILE, handler_ast_allowed_simple},
    {ZEND_AST_IF_ELEM, handle_ast_if_elem},
    {ZEND_AST_SWITCH, handler_ast_allowed_simple},
    {ZEND_AST_SWITCH_CASE, handler_ast_allowed_simple},
    {ZEND_AST_DECLARE, handle_not_allowed},
    {ZEND_AST_USE_TRAIT, handle_not_allowed},
    {ZEND_AST_TRAIT_PRECEDENCE, handle_not_allowed},
    {ZEND_AST_METHOD_REFERENCE, handle_not_allowed},
    {ZEND_AST_NAMESPACE, handle_not_allowed},
    {ZEND_AST_USE_ELEM, handle_not_allowed},
    {ZEND_AST_TRAIT_ALIAS, handle_not_allowed},
    {ZEND_AST_GROUP_USE, handle_not_allowed},
    {ZEND_AST_ATTRIBUTE, handle_not_allowed},
    {ZEND_AST_MATCH, handle_not_allowed},
    {ZEND_AST_MATCH_ARM, handle_not_allowed},
    {ZEND_AST_NAMED_ARG, handle_not_allowed},
    // {ZEND_AST_PARENT_PROPERTY_HOOK_CALL, handle_not_allowed},
    // {ZEND_AST_PIPE, handle_not_allowed},

    // 3 CHILD NODES
    {ZEND_AST_METHOD_CALL, handle_not_allowed},
    {ZEND_AST_NULLSAFE_METHOD_CALL, handle_not_allowed},
    {ZEND_AST_STATIC_CALL, handle_not_allowed},
    {ZEND_AST_CONDITIONAL, handler_ast_allowed_simple}, // Operador ternário (a ? b : c)

    {ZEND_AST_TRY, handle_not_allowed},
    {ZEND_AST_CATCH, handle_not_allowed},
    {ZEND_AST_PROP_GROUP, handle_not_allowed},
    {ZEND_AST_CONST_ELEM, handle_not_allowed},
    {ZEND_AST_CLASS_CONST_GROUP, handle_not_allowed},
    {ZEND_AST_CONST_ENUM_INIT, handle_not_allowed},

    // 4 CHILD NODES
    {ZEND_AST_FOR, handler_ast_allowed_simple},
    {ZEND_AST_FOREACH, handle_not_allowed},
    {ZEND_AST_ENUM_CASE, handle_not_allowed},
    {ZEND_AST_PROP_ELEM, handle_not_allowed},

    // 6 CHILD NODES
    {ZEND_AST_PARAM, handle_not_allowed},
};

// --- FUNÇÃO DE DISPATCH ---

handler get_ast_handler(zend_ast_kind kind)
{
    for (int i = 0; i < sizeof(php_ast_handlers) / sizeof(php_ast_handler); i++)
        if (php_ast_handlers[i].kind == kind)
            return php_ast_handlers[i].fn;

    return NULL;
}

// --- FUNÇÃO PRINCIPAL ---

int compile_ast_as_valid_cuda(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput)
{
    if (ast == NULL)
    {
        return 1;
    }

    handler handler_func = get_ast_handler(ast->kind);

    if (!handler_func)
    {
        return handle_not_allowed(cuda_code_buffer, ast, input, ouput);
    }

    // O handler faz a validação, gera o código e chama a recursão nos filhos
    if (handler_func(cuda_code_buffer, ast, input, ouput) != 1)
    {
        return 0; // Falha na validação ou geração de código
    }

    return 1;
}

// ----------------------------------------------------------------------
// --- IMPLEMENTAÇÕES DOS HANDLERS ---
// ----------------------------------------------------------------------

// --- Handlers de Erro e Recursão Simples ---

static int handle_not_allowed(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput)
{
    php_error_docref(NULL, E_ERROR, "Kernel compilation failed: PHP construct is not allowed.");
    return 0;
}

// Handler para nós que são permitidos, mas onde a sintaxe é injetada
// (Requer lógica complexa para injetar operadores, parênteses e ponto-e-vírgula)
static int handler_ast_allowed_simple(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput)
{
    // A implementação real exigiria uma tabela de operadores para prefixos/infixos.
    // Por simplicidade, vamos apenas validar e recursar nos filhos.

    uint32_t children = zend_ast_get_num_children(ast);
    for (uint32_t i = 0; i < children; i++)
    {
        if (compile_ast_as_valid_cuda(cuda_code_buffer, ast->child[i], input, ouput) != 1)
        {
            return 0;
        }
    }

    // NOTA: Para este nó funcionar de verdade (ex: UNARY_MINUS), você precisa
    // injetar o '-' ANTES da recursão, mas saber quando fechar é complexo.
    // Esta função precisaria de uma grande refatoração ou ser substituída por handlers específicos.

    return 1;
}

// Handler de Contêiner de Listas (ex: ARG_LIST, EXPR_LIST)
static int handler_ast_list_container(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput)
{
    zend_ast_list *list = (zend_ast_list *)ast;

    for (uint32_t i = 0; i < list->children; i++)
    {
        if (compile_ast_as_valid_cuda(cuda_code_buffer, list->child[i], input, ouput) != 1)
        {
            return 0;
        }
        // Adiciona vírgula, mas APENAS se não for o último elemento
        if (i < list->children - 1)
        {
            smart_string_appends(cuda_code_buffer, ", ");
        }
    }
    return 1;
}

// --- HANDLERS ESTRUTURAIS ---

static int handle_ast_stmt_list(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput)
{
    zend_ast_list *list = (zend_ast_list *)ast;

    for (uint32_t i = 0; i < list->children; i++)
    {
        // Chamada recursiva para cada statement
        if (compile_ast_as_valid_cuda(cuda_code_buffer, list->child[i], input, ouput) != 1)
        {
            return 0;
        }
    }
    return 1;
}

static int handler_ast_if(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput)
{
    // Apenas itera sobre os IF_ELEM (IF, ELSEIF, ELSE)
    zend_ast_list *list = (zend_ast_list *)ast;
    for (uint32_t i = 0; i < list->children; i++)
    {
        if (compile_ast_as_valid_cuda(cuda_code_buffer, list->child[i], input, ouput) != 1)
        {
            return 0;
        }
    }
    return 1;
}

static int handle_ast_if_elem(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput)
{
    // ast->child[0] é a Condição, ast->child[1] é o Corpo

    // Injeta a sintaxe 'if ('
    smart_string_appends(cuda_code_buffer, "if (");

    // Compila a Condição recursivamente
    if (compile_ast_as_valid_cuda(cuda_code_buffer, ast->child[0], input, ouput) != 1)
    {
        return 0;
    }

    // Finaliza a condição e abre o corpo
    smart_string_appends(cuda_code_buffer, ") {\n");

    // Compila o Corpo (STMT_LIST) recursivamente
    if (compile_ast_as_valid_cuda(cuda_code_buffer, ast->child[1], input, ouput) != 1)
    {
        return 0;
    }

    // Fecha o bloco
    smart_string_appends(cuda_code_buffer, "}\n");
    return 1;
}

static int handler_ast_return(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput)
{
    smart_string_appends(cuda_code_buffer, "return ");

    // Compila a expressão de retorno
    if (compile_ast_as_valid_cuda(cuda_code_buffer, ast->child[0], input, ouput) != 1)
    {
        return 0;
    }

    smart_string_appends(cuda_code_buffer, ";\n");
    return 1;
}

// --- HANDLERS DE EXPRESSÃO ---

static int handler_ast_zval(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput)
{
    zend_ast_zval *zval_node = (zend_ast_zval *)ast;

    switch (Z_TYPE(zval_node->val))
    {
    case IS_LONG:
        smart_string_append_long(cuda_code_buffer, Z_LVAL(zval_node->val));
        break;
    case IS_DOUBLE:
        char buffer[64];
        double value = Z_DVAL(zval_node->val);
        int len;

        len = snprintf(buffer, sizeof(buffer), "%.17g", value);
        if (len < 0 || (size_t)len >= sizeof(buffer))
        {
            zend_error(E_ERROR, "Error formating Double to CUDA.");
            return 0;
        }

        smart_string_appendl(cuda_code_buffer, buffer, len);
        break;
    case IS_TRUE:
        smart_string_appends(cuda_code_buffer, "1");
        break;
    case IS_FALSE:
        smart_string_appends(cuda_code_buffer, "0");
        break;
    default:
        php_error_docref(NULL, E_ERROR, "Literal type is not allowed in CUDA kernel.");
        return 0;
    }
    return 1;
}

static int handler_ast_var(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput)
{
    // A validação de escopo e tipo é crucial aqui, mas omitida por brevidade.

    zend_ast_zval *var_name_node = (zend_ast_zval *)ast->child[0];
    if (Z_TYPE(var_name_node->val) == IS_STRING)
    {
        // Gera o nome da variável C/CUDA (sem o '$')
        smart_string_appendl(cuda_code_buffer, Z_STRVAL(var_name_node->val), Z_STRLEN(var_name_node->val));
    }
    else
    {
        // Variável complexa (ex: $$var)
        php_error_docref(NULL, E_ERROR, "Complex variable names are not allowed in CUDA kernel.");
        return 0;
    }
    return 1;
}

static int handler_ast_binary_op(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput)
{
    const char *op_symbol = NULL;
    // php_printf("DEBUG: handler_ast_binary_op received attribute: %d\n", ast->attr);

    switch (ast->attr)
    {
    case ZEND_NOP:
        return 1;
        break;
    case ZEND_ADD:
        op_symbol = " + ";
        break;
    case ZEND_SUB:
        op_symbol = " - ";
        break;
    case ZEND_MUL:
        op_symbol = " * ";
        break;
    case ZEND_DIV:
        op_symbol = " / ";
        break;
    case ZEND_ASSIGN:
        op_symbol = " = ";
        break;
    // case ZEND_ASSIGN_ADD:
    //     op_symbol = " += ";
    //     break;
    default:
        php_error_docref(NULL, E_ERROR, "Unsupported binary operation in CUDA kernel.");
        return 0;
    }

    smart_string_appendc(cuda_code_buffer, '(');
    if (compile_ast_as_valid_cuda(cuda_code_buffer, ast->child[0], input, ouput) != 1)
        return 0;
    smart_string_appends(cuda_code_buffer, op_symbol);
    if (compile_ast_as_valid_cuda(cuda_code_buffer, ast->child[1], input, ouput) != 1)
        return 0;
    smart_string_appendc(cuda_code_buffer, ')');

    // Se for um statement (ASSIGN), adicione ';'
    if (ast->attr == ZEND_ASSIGN) // || ast->attr == ZEND_ASSIGN_ADD
    {
        smart_string_appends(cuda_code_buffer, ";\n");
    }

    return 1;
}

static int handler_ast_comp_op(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput)
{
    // ast->child[0] é LHS, ast->child[1] é RHS
    const char *op_symbol = NULL;

    // Mapeamento para operadores de comparação
    switch (ast->kind)
    {
    case ZEND_AST_GREATER:
        op_symbol = " > ";
        break;
    case ZEND_AST_GREATER_EQUAL:
        op_symbol = " >= ";
        break;
    case ZEND_AST_AND:
        op_symbol = " && ";
        break;
    case ZEND_AST_OR:
        op_symbol = " || ";
        break;
    default:
        op_symbol = " [COMP_OP_UNHANDLED] ";
    }

    smart_string_appendc(cuda_code_buffer, '(');
    if (compile_ast_as_valid_cuda(cuda_code_buffer, ast->child[0], input, ouput) != 1)
        return 0;
    smart_string_appends(cuda_code_buffer, op_symbol);
    if (compile_ast_as_valid_cuda(cuda_code_buffer, ast->child[1], input, ouput) != 1)
        return 0;
    smart_string_appendc(cuda_code_buffer, ')');
    return 1;
}

static int handler_ast_call(smart_string *cuda_code_buffer, zend_ast *ast, func_parameter_list *input, func_parameter_list *ouput)
{
    // ast->child[0] é o nome da função (ZVAL), ast->child[1] são os argumentos (ARG_LIST)

    // 1. Geração do nome da função
    zend_ast_zval *func_name_node = (zend_ast_zval *)ast->child[0];
    if (Z_TYPE(func_name_node->val) == IS_STRING)
    {
        // Validação: A função 'max' é permitida? (Precisa de uma lista de funções CUDA permitidas)
        smart_string_appendl(cuda_code_buffer, Z_STRVAL(func_name_node->val), Z_STRLEN(func_name_node->val));
    }
    else
    {
        php_error_docref(NULL, E_ERROR, "Complex function call names are not allowed.");
        return 0;
    }

    smart_string_appendc(cuda_code_buffer, '(');

    if (compile_ast_as_valid_cuda(cuda_code_buffer, ast->child[1], input, ouput) != 1)
    {
        return 0;
    }

    smart_string_appendc(cuda_code_buffer, ')');

    return 1;
}