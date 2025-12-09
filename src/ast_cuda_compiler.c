#include "ast_cuda_compiler.h"
#include "ext/standard/php_smart_string.h"
#include "ext/standard/php_string.h"
#include "zend_operators.h"
#include "zend_compile.h"
#include "zend_string.h"
#include "zend_ast.h"
#include "zend_attributes.h"
#include "kernel_reflection.h"
#include "kernel_types.h"
#include "zend_hash.h"

static int handle_not_allowed(cuda_compilation_context_t *context, zend_ast *ast);
static int handle_ast_stmt_list(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_if(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_for(cuda_compilation_context_t *context, zend_ast *ast);
static int handle_ast_if_elem(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_zval(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_var(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_return(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_call(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_binary_op(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_assign(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_comp_op(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_list_container(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_allowed_simple(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_dim(cuda_compilation_context_t *context, zend_ast *ast);

php_ast_handler php_ast_handlers[] = {
    {ZEND_AST_ZVAL, handler_ast_zval},
    {ZEND_AST_CONSTANT, handler_ast_allowed_simple},
    {ZEND_AST_ZNODE, handle_not_allowed},
    {ZEND_AST_FUNC_DECL, handle_not_allowed},
    {ZEND_AST_CLOSURE, handle_not_allowed},
    {ZEND_AST_METHOD, handle_not_allowed},
    {ZEND_AST_CLASS, handle_not_allowed},
    {ZEND_AST_ARROW_FUNC, handle_not_allowed},
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
    {ZEND_AST_MAGIC_CONST, handle_not_allowed},
    {ZEND_AST_TYPE, handle_not_allowed},
    {ZEND_AST_CONSTANT_CLASS, handle_not_allowed},
    {ZEND_AST_CALLABLE_CONVERT, handle_not_allowed},
    {ZEND_AST_VAR, handler_ast_var},
    {ZEND_AST_CONST, handler_ast_allowed_simple},
    {ZEND_AST_UNPACK, handle_not_allowed},
    {ZEND_AST_UNARY_PLUS, handler_ast_allowed_simple},
    {ZEND_AST_UNARY_MINUS, handler_ast_allowed_simple},
    {ZEND_AST_CAST, handler_ast_allowed_simple},
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
    {ZEND_AST_DIM, handler_ast_dim},
    {ZEND_AST_PROP, handle_not_allowed},
    {ZEND_AST_NULLSAFE_PROP, handle_not_allowed},
    {ZEND_AST_STATIC_PROP, handle_not_allowed},
    {ZEND_AST_CALL, handler_ast_call},
    {ZEND_AST_CLASS_CONST, handler_ast_allowed_simple},
    {ZEND_AST_ASSIGN, handler_ast_assign},
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
    {ZEND_AST_METHOD_CALL, handle_not_allowed},
    {ZEND_AST_NULLSAFE_METHOD_CALL, handle_not_allowed},
    {ZEND_AST_STATIC_CALL, handle_not_allowed},
    {ZEND_AST_CONDITIONAL, handler_ast_allowed_simple},
    {ZEND_AST_TRY, handle_not_allowed},
    {ZEND_AST_CATCH, handle_not_allowed},
    {ZEND_AST_PROP_GROUP, handle_not_allowed},
    {ZEND_AST_CONST_ELEM, handle_not_allowed},
    {ZEND_AST_CLASS_CONST_GROUP, handle_not_allowed},
    {ZEND_AST_CONST_ENUM_INIT, handle_not_allowed},
    {ZEND_AST_FOR, handler_ast_for},
    {ZEND_AST_FOREACH, handle_not_allowed},
    {ZEND_AST_ENUM_CASE, handle_not_allowed},
    {ZEND_AST_PROP_ELEM, handle_not_allowed},
    {ZEND_AST_PARAM, handle_not_allowed},
};

// =================================================================================
// FUNÇÕES HELPERS
// =================================================================================

static const char *get_cuda_type_str(dtype_t type)
{
    switch (type)
    {
    case FLOAT32:
        return "float";
    case FLOAT64:
        return "double";
    case INT32:
        return "int";
    case INT64:
        return "long long";
    case BOOL:
        return "bool";
    default:
        return "void";
    }
}

static func_parameter *find_kernel_parameter(func_parameter_list_t *list, const char *name)
{
    if (!list || !list->parameters)
    {
        return NULL;
    }

    for (int i = 0; i < list->total; i++)
    {
        if (strcmp(list->parameters[i]->name, name) == 0)
        {
            return list->parameters[i];
        }
    }
    return NULL;
}

handler get_ast_handler(zend_ast_kind kind)
{
    for (int i = 0; i < sizeof(php_ast_handlers) / sizeof(php_ast_handler); i++)
    {
        if (php_ast_handlers[i].kind == kind)
        {
            return php_ast_handlers[i].fn;
        }
    }
    return NULL;
}

static zend_always_inline zend_bool needs_semicolon(zend_ast *ast)
{
    if (!ast)
        return 0;
    switch (ast->kind)
    {
    case ZEND_AST_IF:
    case ZEND_AST_FOR:
    case ZEND_AST_WHILE:
    case ZEND_AST_DO_WHILE:
    case ZEND_AST_SWITCH:
        return 0;
    default:
        return 1;
    }
}

int compile_ast_as_valid_cuda(cuda_compilation_context_t *context, zend_ast *ast)
{
    if (ast == NULL)
    {
        return 1;
    }

    if (ast->kind != ZEND_AST_STMT_LIST && ast->kind != ZEND_AST_ARG_LIST)
    {
        context->last_evaluated_dtype = DTYPE_UNKNOWN;
    }

    handler handler_func = get_ast_handler(ast->kind);

    if (!handler_func)
    {
        return handle_not_allowed(context, ast);
    }

    if (handler_func(context, ast) != 1)
    {
        return 0;
    }

    return 1;
}

static int handle_not_allowed(cuda_compilation_context_t *context, zend_ast *ast)
{
    php_error_docref(NULL, E_ERROR, "Kernel compilation failed: PHP construct (Kind: %d) is not allowed.", ast->kind);
    return 0;
}

static int handler_ast_allowed_simple(cuda_compilation_context_t *context, zend_ast *ast)
{
    uint32_t children = zend_ast_get_num_children(ast);
    for (uint32_t i = 0; i < children; i++)
    {
        if (compile_ast_as_valid_cuda(context, ast->child[i]) != 1)
        {
            return 0;
        }
    }
    return 1;
}

static int handler_ast_list_container(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast_list *list = (zend_ast_list *)ast;
    for (uint32_t i = 0; i < list->children; i++)
    {
        if (compile_ast_as_valid_cuda(context, list->child[i]) != 1)
        {
            return 0;
        }
        if (i < list->children - 1)
        {
            smart_string_appends(context->cuda_code_buffer, ", ");
        }
    }
    return 1;
}

static int handle_ast_stmt_list(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast_list *list = (zend_ast_list *)ast;
    for (uint32_t i = 0; i < list->children; i++)
    {
        zend_ast *stmt = list->child[i];
        if (compile_ast_as_valid_cuda(context, stmt) != 1)
        {
            return 0;
        }
        if (needs_semicolon(stmt))
        {
            smart_string_appends(context->cuda_code_buffer, ";\n");
        }
    }
    return 1;
}

static int handler_ast_var(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast *name_node = ast->child[0];

    if (!name_node || name_node->kind != ZEND_AST_ZVAL)
    {
        php_error_docref(NULL, E_ERROR, "Complex variable names (e.g., $$var) are not allowed.");
        return 0;
    }

    zend_ast_zval *var_name_node = (zend_ast_zval *)name_node;
    if (Z_TYPE(var_name_node->val) != IS_STRING)
    {
        php_error_docref(NULL, E_ERROR, "Variable name must be a string.");
        return 0;
    }

    zend_string *var_name_zend = Z_STR(var_name_node->val);
    const char *name_c = ZSTR_VAL(var_name_zend);

    func_parameter *param = find_kernel_parameter(context->parameters, name_c);

    if (param)
    {
        context->last_evaluated_dtype = param->dtype;
        smart_string_appendl(context->cuda_code_buffer, name_c, ZSTR_LEN(var_name_zend));
        return 1;
    }

    local_variable_t *local = zend_hash_find_ptr(&context->local_variables, var_name_zend);

    if (local)
    {
        context->last_evaluated_dtype = local->dtype;
        smart_string_appendl(context->cuda_code_buffer, name_c, ZSTR_LEN(var_name_zend));
        return 1;
    }

    php_error_docref(NULL, E_ERROR, "Undefined variable '$%s'. Variable must be a parameter or previously defined.", name_c);
    return 0;
}

static int handler_ast_assign(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast *lvalue = ast->child[0];
    zend_ast *rvalue = ast->child[1];

    smart_string rvalue_buffer = {0};
    smart_string_alloc(&rvalue_buffer, 128, 0);

    // Troca de contexto de buffer
    smart_string *original_buffer = context->cuda_code_buffer;
    context->cuda_code_buffer = &rvalue_buffer;

    if (compile_ast_as_valid_cuda(context, rvalue) != 1)
    {
        smart_string_free(&rvalue_buffer);
        context->cuda_code_buffer = original_buffer;
        return 0;
    }

    dtype_t rvalue_type = context->last_evaluated_dtype;

    context->cuda_code_buffer = original_buffer;

    if (lvalue->kind == ZEND_AST_VAR)
    {
        zend_ast_zval *var_name_node = (zend_ast_zval *)lvalue->child[0];
        zend_string *var_name_zend = Z_STR(var_name_node->val);
        const char *name_c = ZSTR_VAL(var_name_zend);

        func_parameter *param = find_kernel_parameter(context->parameters, name_c);
        local_variable_t *local = zend_hash_find_ptr(&context->local_variables, var_name_zend);

        if (!param && !local)
        {
            if (rvalue_type == DTYPE_UNKNOWN)
            {
                smart_string_free(&rvalue_buffer);
                php_error_docref(NULL, E_ERROR, "Cannot infer type for new variable '$%s'.", name_c);
                return 0;
            }

            local_variable_t *new_var = (local_variable_t *)ecalloc(1, sizeof(local_variable_t));
            new_var->name = zend_string_copy(var_name_zend);
            new_var->dtype = rvalue_type;

            zend_hash_add_ptr(&context->local_variables, var_name_zend, new_var);

            smart_string_appends(context->cuda_code_buffer, get_cuda_type_str(rvalue_type));
            smart_string_appendc(context->cuda_code_buffer, ' ');
            smart_string_appendl(context->cuda_code_buffer, name_c, ZSTR_LEN(var_name_zend));
        }
        else
        {
            dtype_t lvalue_type = param ? param->dtype : local->dtype;

            if (lvalue_type != rvalue_type && rvalue_type != DTYPE_UNKNOWN)
            {
                if (lvalue_type != rvalue_type)
                {
                    smart_string_free(&rvalue_buffer);
                    php_error_docref(NULL, E_ERROR, "Type mismatch for '$%s'. Expected %s, got %s.",
                                     name_c, get_cuda_type_str(lvalue_type), get_cuda_type_str(rvalue_type));
                    return 0;
                }
            }

            smart_string_appendl(context->cuda_code_buffer, name_c, ZSTR_LEN(var_name_zend));
        }
    }
    else if (lvalue->kind == ZEND_AST_DIM)
    {
        if (compile_ast_as_valid_cuda(context, lvalue) != 1)
        {
            smart_string_free(&rvalue_buffer);
            return 0;
        }
    }
    else
    {
        smart_string_free(&rvalue_buffer);
        php_error_docref(NULL, E_ERROR, "Invalid assignment target.");
        return 0;
    }

    smart_string_appends(context->cuda_code_buffer, " = ");
    smart_string_append(context->cuda_code_buffer, &rvalue_buffer);
    smart_string_free(&rvalue_buffer);

    return 1;
}

static int handler_ast_if(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast_list *list = (zend_ast_list *)ast;
    for (uint32_t i = 0; i < list->children; i++)
    {
        if (compile_ast_as_valid_cuda(context, list->child[i]) != 1)
        {
            return 0;
        }
    }
    return 1;
}

static int handle_ast_if_elem(cuda_compilation_context_t *context, zend_ast *ast)
{
    smart_string_appends(context->cuda_code_buffer, "if (");
    if (compile_ast_as_valid_cuda(context, ast->child[0]) != 1)
    {
        return 0;
    }
    smart_string_appends(context->cuda_code_buffer, ") {\n");
    if (compile_ast_as_valid_cuda(context, ast->child[1]) != 1)
    {
        return 0;
    }
    smart_string_appends(context->cuda_code_buffer, "}\n");
    return 1;
}

static int handler_ast_for(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast *init_node = ast->child[0];
    zend_ast *cond_node = ast->child[1];
    zend_ast *loop_node = ast->child[2];
    zend_ast *body_node = ast->child[3];

    smart_string_appends(context->cuda_code_buffer, "for (");

    if (init_node)
    {
        if (compile_ast_as_valid_cuda(context, init_node) != 1)
            return 0;
    }
    smart_string_appends(context->cuda_code_buffer, "; ");

    if (cond_node)
    {
        if (compile_ast_as_valid_cuda(context, cond_node) != 1)
            return 0;
    }
    smart_string_appends(context->cuda_code_buffer, "; ");

    if (loop_node)
    {
        if (compile_ast_as_valid_cuda(context, loop_node) != 1)
            return 0;
    }

    smart_string_appends(context->cuda_code_buffer, ") {\n");

    context->loop_depth++;
    if (body_node)
    {
        if (compile_ast_as_valid_cuda(context, body_node) != 1)
        {
            context->loop_depth--;
            return 0;
        }
    }
    context->loop_depth--;

    smart_string_appends(context->cuda_code_buffer, "}\n");
    return 1;
}

static int handler_ast_return(cuda_compilation_context_t *context, zend_ast *ast)
{
    smart_string_appends(context->cuda_code_buffer, "return ");
    if (compile_ast_as_valid_cuda(context, ast->child[0]) != 1)
    {
        return 0;
    }

    // @todo: validate return type matches defined return type
    return 1;
}

static int handler_ast_dim(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast *array_expr = ast->child[0];
    zend_ast *index_expr = ast->child[1];

    if (compile_ast_as_valid_cuda(context, array_expr) != 1)
        return 0;

    // O tipo do array base
    // dtype_t array_type = context->last_evaluated_dtype;

    smart_string_appendc(context->cuda_code_buffer, '[');
    if (compile_ast_as_valid_cuda(context, index_expr) != 1)
        return 0;
    smart_string_appendc(context->cuda_code_buffer, ']');

    // Se soubéssemos que era float[], agora é float.
    // Como simplificação, mantemos o tipo base ou UNKNOWN.
    return 1;
}

static int handler_ast_zval(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast_zval *zval_node = (zend_ast_zval *)ast;

    switch (Z_TYPE(zval_node->val))
    {
    case IS_LONG:
        smart_string_append_long(context->cuda_code_buffer, Z_LVAL(zval_node->val));
        context->last_evaluated_dtype = INT32;
        break;
    case IS_DOUBLE:
    {
        char buffer[64];
        int len = snprintf(buffer, sizeof(buffer), "%.17g", Z_DVAL(zval_node->val));
        smart_string_appendl(context->cuda_code_buffer, buffer, len);
        context->last_evaluated_dtype = FLOAT32;
        break;
    }
    case IS_TRUE:
        smart_string_appends(context->cuda_code_buffer, "1");
        context->last_evaluated_dtype = BOOL;
        break;
    case IS_FALSE:
        smart_string_appends(context->cuda_code_buffer, "0");
        context->last_evaluated_dtype = BOOL;
        break;
    default:
        php_error_docref(NULL, E_ERROR, "Literal type is not allowed in CUDA kernel.");
        return 0;
    }
    return 1;
}

static int handler_ast_binary_op(cuda_compilation_context_t *context, zend_ast *ast)
{
    const char *op_symbol = NULL;
    switch (ast->attr)
    {
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
    case ZEND_IS_EQUAL:
    case ZEND_IS_IDENTICAL:
        op_symbol = " == ";
        break;
    case ZEND_IS_NOT_EQUAL:
    case ZEND_IS_NOT_IDENTICAL:
        op_symbol = " != ";
        break;
    case ZEND_IS_SMALLER:
        op_symbol = " < ";
        break;
    case ZEND_IS_SMALLER_OR_EQUAL:
        op_symbol = " <= ";
        break;
    // case ZEND_ASSIGN_ADD: op_symbol = " += "; break;
    // case ZEND_ASSIGN_SUB: op_symbol = " -= "; break;
    // case ZEND_ASSIGN_MUL: op_symbol = " *= "; break;
    // case ZEND_ASSIGN_DIV: op_symbol = " /= "; break;
    default:
        return 1; // NOP
    }

    smart_string_appendc(context->cuda_code_buffer, '(');
    if (compile_ast_as_valid_cuda(context, ast->child[0]) != 1)
        return 0;
    dtype_t left = context->last_evaluated_dtype;

    smart_string_appends(context->cuda_code_buffer, op_symbol);

    if (compile_ast_as_valid_cuda(context, ast->child[1]) != 1)
        return 0;
    dtype_t right = context->last_evaluated_dtype;

    smart_string_appendc(context->cuda_code_buffer, ')');

    // if (ast->attr >= ZEND_ASSIGN_ADD && ast->attr <= ZEND_ASSIGN_DIV)
    // {
    //     smart_string_appends(context->cuda_code_buffer, ";\n");
    // }

    // Inferência de tipo simples
    if (left == FLOAT32 || right == FLOAT32)
        context->last_evaluated_dtype = FLOAT32;
    else if (left == FLOAT64 || right == FLOAT64)
        context->last_evaluated_dtype = FLOAT64;
    else
        context->last_evaluated_dtype = INT32;

    return 1;
}

static int handler_ast_comp_op(cuda_compilation_context_t *context, zend_ast *ast)
{
    const char *op_symbol = NULL;
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
        op_symbol = " ? ";
    }

    smart_string_appendc(context->cuda_code_buffer, '(');
    if (compile_ast_as_valid_cuda(context, ast->child[0]) != 1)
        return 0;
    smart_string_appends(context->cuda_code_buffer, op_symbol);
    if (compile_ast_as_valid_cuda(context, ast->child[1]) != 1)
        return 0;
    smart_string_appendc(context->cuda_code_buffer, ')');

    context->last_evaluated_dtype = BOOL;
    return 1;
}

static int handler_ast_call(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast *callee = ast->child[0];
    zend_ast *args = ast->child[1];

    if (callee->kind != ZEND_AST_ZVAL)
    {
        php_error_docref(NULL, E_ERROR, "Complex function calls are not allowed.");
        return 0;
    }

    zend_ast_zval *func_name_node = (zend_ast_zval *)callee;
    if (Z_TYPE(func_name_node->val) != IS_STRING)
        return 0;

    zend_string *name = Z_STR(func_name_node->val);

    // Mapeamento simples
    if (zend_string_equals_literal(name, "threadIdx"))
    {
        smart_string_appends(context->cuda_code_buffer, "threadIdx.x");
        context->last_evaluated_dtype = INT32;
        return 1;
    }
    else if (zend_string_equals_literal(name, "calculateMax"))
    {
        smart_string_appends(context->cuda_code_buffer, "fmaxf");
        context->last_evaluated_dtype = FLOAT32; // Assume float
    }
    else
    {
        smart_string_appendl(context->cuda_code_buffer, ZSTR_VAL(name), ZSTR_LEN(name));
    }

    smart_string_appendc(context->cuda_code_buffer, '(');
    if (compile_ast_as_valid_cuda(context, args) != 1)
        return 0;
    smart_string_appendc(context->cuda_code_buffer, ')');

    return 1;
}