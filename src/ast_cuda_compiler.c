#include "ext/standard/php_smart_string.h"
#include "ext/standard/php_string.h"
#include "ext/standard/php_math.h"
#include "zend_operators.h"
#include "zend_compile.h"
#include "zend_string.h"
#include "zend_ast.h"
#include "zend_attributes.h"
#include "kernel_reflection.h"
#include "kernel_types.h"
#include "zend_hash.h"
#include "zend_globals.h"
#include "ast_cuda_compiler.h"

static const cuda_function_info_t cuda_functions[] = {
    {"max", "fmaxf", "fmax", NULL, FLOAT32, FLOAT64, DTYPE_UNKNOWN, 2, {FLOAT32, FLOAT32}, {FLOAT64, FLOAT64}, {0}, "math_functions.h"},
    {"min", "fminf", "fmin", NULL, FLOAT32, FLOAT64, DTYPE_UNKNOWN, 2, {FLOAT32, FLOAT32}, {FLOAT64, FLOAT64}, {0}, "math_functions.h"},
    {"exp", "expf", "exp", NULL, FLOAT32, FLOAT64, DTYPE_UNKNOWN, 1, {FLOAT32}, {FLOAT64}, {0}, "math_functions.h"},
    {"log", "logf", "log", NULL, FLOAT32, FLOAT64, DTYPE_UNKNOWN, 1, {FLOAT32}, {FLOAT64}, {0}, "math_functions.h"},
    {"sin", "sinf", "sin", NULL, FLOAT32, FLOAT64, DTYPE_UNKNOWN, 1, {FLOAT32}, {FLOAT64}, {0}, "math_functions.h"},
    {"cos", "cosf", "cos", NULL, FLOAT32, FLOAT64, DTYPE_UNKNOWN, 1, {FLOAT32}, {FLOAT64}, {0}, "math_functions.h"},
    {"sqrt", "sqrtf", "sqrt", NULL, FLOAT32, FLOAT64, DTYPE_UNKNOWN, 1, {FLOAT32}, {FLOAT64}, {0}, "math_functions.h"},
    {"pow", "powf", "pow", NULL, FLOAT32, FLOAT64, DTYPE_UNKNOWN, 2, {FLOAT32, FLOAT32}, {FLOAT64, FLOAT64}, {0}, "math_functions.h"},
    {"abs", NULL, NULL, "abs", DTYPE_UNKNOWN, DTYPE_UNKNOWN, INT32, 1, {0}, {0}, {INT32}, "stdlib.h"},
    {"fabs", "fabsf", "fabs", NULL, FLOAT32, FLOAT64, DTYPE_UNKNOWN, 1, {FLOAT32}, {FLOAT64}, {0}, "math_functions.h"},
    {"ceil", "ceilf", "ceil", NULL, FLOAT32, FLOAT64, DTYPE_UNKNOWN, 1, {FLOAT32}, {FLOAT64}, {0}, "math_functions.h"},
    {"floor", "floorf", "floor", NULL, FLOAT32, FLOAT64, DTYPE_UNKNOWN, 1, {FLOAT32}, {FLOAT64}, {0}, "math_functions.h"},
    {"round", "roundf", "round", NULL, FLOAT32, FLOAT64, DTYPE_UNKNOWN, 1, {FLOAT32}, {FLOAT64}, {0}, "math_functions.h"},
    {"threadIdx", "threadIdx.x", NULL, NULL, INT32, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 0, {0}, {0}, {0}, NULL},
    {"blockIdx", "blockIdx.x", NULL, NULL, INT32, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 0, {0}, {0}, {0}, NULL},
    {"blockDim", "blockDim.x", NULL, NULL, INT32, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 0, {0}, {0}, {0}, NULL},
    {"gridDim", "gridDim.x", NULL, NULL, INT32, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 0, {0}, {0}, {0}, NULL},
    {NULL, NULL, NULL, NULL, DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 0, {0}, {0}, {0}, NULL}};

typedef struct
{
    uint32_t dimensions;
    uint32_t sizes[4];
    dtype_t element_type;
} array_info_t;

static int generate_function_signature(cuda_compilation_context_t *context);
static int handle_not_allowed(cuda_compilation_context_t *context, zend_ast *ast);
static int handle_ast_stmt_list(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_if(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_for(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_while(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_do_while(cuda_compilation_context_t *context, zend_ast *ast);
static int handle_ast_if_elem(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_zval(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_var(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_return(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_binary_op(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_unary_op(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_assign(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_comp_op(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_list_container(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_allowed_simple(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_dim(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_cast(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_conditional(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_switch(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_switch_case(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_break_continue(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_inc_dec(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_assign_op(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_method_call(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_prop(cuda_compilation_context_t *context, zend_ast *ast);

static const char *get_ast_kind_name(zend_ast_kind kind)
{
    switch (kind)
    {
    case ZEND_AST_ZVAL:
        return "ZVAL";
    case ZEND_AST_FUNC_DECL:
        return "FUNC_DECL";
    case ZEND_AST_CLOSURE:
        return "CLOSURE";
    case ZEND_AST_METHOD:
        return "METHOD";
    case ZEND_AST_CLASS:
        return "CLASS";
    case ZEND_AST_ARRAY:
        return "ARRAY";
    case ZEND_AST_ENCAPS_LIST:
        return "ENCAPS_LIST";
    case ZEND_AST_STMT_LIST:
        return "STMT_LIST";
    case ZEND_AST_IF:
        return "IF";
    case ZEND_AST_SWITCH_LIST:
        return "SWITCH_LIST";
    case ZEND_AST_VAR:
        return "VAR";
    case ZEND_AST_CONST:
        return "CONST";
    case ZEND_AST_UNPACK:
        return "UNPACK";
    case ZEND_AST_UNARY_PLUS:
        return "UNARY_PLUS";
    case ZEND_AST_UNARY_MINUS:
        return "UNARY_MINUS";
    case ZEND_AST_CAST:
        return "CAST";
    case ZEND_AST_EMPTY:
        return "EMPTY";
    case ZEND_AST_ISSET:
        return "ISSET";
    case ZEND_AST_PRINT:
        return "PRINT";
    case ZEND_AST_UNARY_OP:
        return "UNARY_OP";
    case ZEND_AST_PRE_INC:
        return "PRE_INC";
    case ZEND_AST_PRE_DEC:
        return "PRE_DEC";
    case ZEND_AST_POST_INC:
        return "POST_INC";
    case ZEND_AST_POST_DEC:
        return "POST_DEC";
    case ZEND_AST_GLOBAL:
        return "GLOBAL";
    case ZEND_AST_UNSET:
        return "UNSET";
    case ZEND_AST_RETURN:
        return "RETURN";
    case ZEND_AST_LABEL:
        return "LABEL";
    case ZEND_AST_REF:
        return "REF";
    case ZEND_AST_HALT_COMPILER:
        return "HALT_COMPILER";
    case ZEND_AST_ECHO:
        return "ECHO";
    case ZEND_AST_THROW:
        return "THROW";
    case ZEND_AST_GOTO:
        return "GOTO";
    case ZEND_AST_BREAK:
        return "BREAK";
    case ZEND_AST_CONTINUE:
        return "CONTINUE";
    case ZEND_AST_DIM:
        return "DIM";
    case ZEND_AST_PROP:
        return "PROP";
    case ZEND_AST_STATIC_PROP:
        return "STATIC_PROP";
    case ZEND_AST_CALL:
        return "CALL";
    case ZEND_AST_CLASS_CONST:
        return "CLASS_CONST";
    case ZEND_AST_ASSIGN:
        return "ASSIGN";
    case ZEND_AST_ASSIGN_REF:
        return "ASSIGN_REF";
    case ZEND_AST_ASSIGN_OP:
        return "ASSIGN_OP";
    case ZEND_AST_BINARY_OP:
        return "BINARY_OP";
    case ZEND_AST_ARRAY_ELEM:
        return "ARRAY_ELEM";
    case ZEND_AST_NEW:
        return "NEW";
    case ZEND_AST_INSTANCEOF:
        return "INSTANCEOF";
    case ZEND_AST_YIELD:
        return "YIELD";
    case ZEND_AST_COALESCE:
        return "COALESCE";
    case ZEND_AST_STATIC:
        return "STATIC";
    case ZEND_AST_WHILE:
        return "WHILE";
    case ZEND_AST_DO_WHILE:
        return "DO_WHILE";
    case ZEND_AST_IF_ELEM:
        return "IF_ELEM";
    case ZEND_AST_SWITCH:
        return "SWITCH";
    case ZEND_AST_SWITCH_CASE:
        return "SWITCH_CASE";
    case ZEND_AST_DECLARE:
        return "DECLARE";
    case ZEND_AST_USE_TRAIT:
        return "USE_TRAIT";
    case ZEND_AST_TRAIT_PRECEDENCE:
        return "TRAIT_PRECEDENCE";
    case ZEND_AST_METHOD_REFERENCE:
        return "METHOD_REFERENCE";
    case ZEND_AST_NAMESPACE:
        return "NAMESPACE";
    case ZEND_AST_USE_ELEM:
        return "USE_ELEM";
    case ZEND_AST_TRAIT_ALIAS:
        return "TRAIT_ALIAS";
    case ZEND_AST_GROUP_USE:
        return "GROUP_USE";
    case ZEND_AST_ATTRIBUTE:
        return "ATTRIBUTE";
    case ZEND_AST_MATCH:
        return "MATCH";
    case ZEND_AST_METHOD_CALL:
        return "METHOD_CALL";
    case ZEND_AST_STATIC_CALL:
        return "STATIC_CALL";
    case ZEND_AST_CONDITIONAL:
        return "CONDITIONAL";
    case ZEND_AST_TRY:
        return "TRY";
    case ZEND_AST_CATCH:
        return "CATCH";
    case ZEND_AST_FOR:
        return "FOR";
    case ZEND_AST_FOREACH:
        return "FOREACH";
    case ZEND_AST_ARG_LIST:
        return "ARG_LIST";
    case ZEND_AST_EXPR_LIST:
        return "EXPR_LIST";
    case ZEND_AST_PARAM_LIST:
        return "PARAM_LIST";
    case ZEND_AST_CLOSURE_USES:
        return "CLOSURE_USES";
    case ZEND_AST_PROP_DECL:
        return "PROP_DECL";
    case ZEND_AST_CONST_DECL:
        return "CONST_DECL";
    case ZEND_AST_CLASS_CONST_DECL:
        return "CLASS_CONST_DECL";
    case ZEND_AST_NAME_LIST:
        return "NAME_LIST";
    case ZEND_AST_TRAIT_ADAPTATIONS:
        return "TRAIT_ADAPTATIONS";
    case ZEND_AST_USE:
        return "USE";
    case ZEND_AST_MAGIC_CONST:
        return "MAGIC_CONST";
    default:
        return "UNKNOWN";
    }
}

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
    {ZEND_AST_UNARY_PLUS, handler_ast_unary_op},
    {ZEND_AST_UNARY_MINUS, handler_ast_unary_op},
    {ZEND_AST_CAST, handler_ast_cast},
    {ZEND_AST_EMPTY, handle_not_allowed},
    {ZEND_AST_ISSET, handle_not_allowed},
    {ZEND_AST_SILENCE, handle_not_allowed},
    {ZEND_AST_SHELL_EXEC, handle_not_allowed},
    {ZEND_AST_PRINT, handle_not_allowed},
    {ZEND_AST_INCLUDE_OR_EVAL, handle_not_allowed},
    {ZEND_AST_UNARY_OP, handler_ast_unary_op},
    {ZEND_AST_PRE_INC, handler_ast_inc_dec},
    {ZEND_AST_PRE_DEC, handler_ast_inc_dec},
    {ZEND_AST_POST_INC, handler_ast_inc_dec},
    {ZEND_AST_POST_DEC, handler_ast_inc_dec},
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
    {ZEND_AST_BREAK, handler_ast_break_continue},
    {ZEND_AST_CONTINUE, handler_ast_break_continue},
    {ZEND_AST_DIM, handler_ast_dim},
    {ZEND_AST_PROP, handler_ast_prop},
    {ZEND_AST_NULLSAFE_PROP, handle_not_allowed},
    {ZEND_AST_STATIC_PROP, handle_not_allowed},
    {ZEND_AST_CALL, handle_not_allowed},
    {ZEND_AST_CLASS_CONST, handler_ast_allowed_simple},
    {ZEND_AST_ASSIGN, handler_ast_assign},
    {ZEND_AST_ASSIGN_REF, handle_not_allowed},
    {ZEND_AST_ASSIGN_OP, handler_ast_assign_op},
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
    {ZEND_AST_WHILE, handler_ast_while},
    {ZEND_AST_DO_WHILE, handler_ast_do_while},
    {ZEND_AST_IF_ELEM, handle_ast_if_elem},
    {ZEND_AST_SWITCH, handler_ast_switch},
    {ZEND_AST_SWITCH_CASE, handler_ast_switch_case},
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
    {ZEND_AST_METHOD_CALL, handler_ast_method_call},
    {ZEND_AST_NULLSAFE_METHOD_CALL, handle_not_allowed},
    {ZEND_AST_STATIC_CALL, handle_not_allowed},
    {ZEND_AST_CONDITIONAL, handler_ast_conditional},
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

static void add_cuda_header(cuda_compilation_context_t *context, const char *header)
{
    zend_string *key = zend_string_init(header, strlen(header), 0);
    if (!zend_hash_exists(context->headers, key))
    {
        zend_hash_add_ptr(context->headers, key, (void *)header);
    }

    zend_string_release(key);
}

static const char *get_binary_op_symbol(uint32_t op_type)
{
    switch (op_type)
    {
    case ZEND_ADD:
        return " + ";
    case ZEND_SUB:
        return " - ";
    case ZEND_MUL:
        return " * ";
    case ZEND_DIV:
        return " / ";
    case ZEND_MOD:
        return " % ";
    case ZEND_SL:
        return " << ";
    case ZEND_SR:
        return " >> ";
    case ZEND_BW_OR:
        return " | ";
    case ZEND_BW_AND:
        return " & ";
    case ZEND_BW_XOR:
        return " ^ ";
    case ZEND_BOOL_XOR:
        return " != ";
    case ZEND_IS_EQUAL:
        return " == ";
    case ZEND_IS_NOT_EQUAL:
        return " != ";
    case ZEND_IS_IDENTICAL:
        return " == ";
    case ZEND_IS_NOT_IDENTICAL:
        return " != ";
    case ZEND_IS_SMALLER:
        return " < ";
    case ZEND_IS_SMALLER_OR_EQUAL:
        return " <= ";
    default:
        return NULL;
    }
}

static const char *get_assign_op_symbol(uint32_t op_type)
{
    switch (op_type)
    {
    case ZEND_ADD:
        return " += ";
    case ZEND_SUB:
        return " -= ";
    case ZEND_MUL:
        return " *= ";
    case ZEND_DIV:
        return " /= ";
    case ZEND_MOD:
        return " %= ";
    case ZEND_SL:
        return " <<= ";
    case ZEND_SR:
        return " >>= ";
    case ZEND_BW_OR:
        return " |= ";
    case ZEND_BW_AND:
        return " &= ";
    case ZEND_BW_XOR:
        return " ^= ";
    case ZEND_CONCAT:
        return " += ";
    default:
        return NULL;
    }
}

static const char *get_cuda_object_name(int obj_type)
{
    switch (obj_type)
    {
    case CUDA_OBJ_CUDA:
        return "cuda";
    case CUDA_OBJ_MATH:
        return "math";
    case CUDA_OBJ_ATOMIC:
        return "atomic";
    case CUDA_OBJ_MEMORY:
        return "memory";
    case CUDA_OBJ_SYNC:
        return "sync";
    case CUDA_OBJ_THREADIDX:
        return "threadIdx";
    case CUDA_OBJ_BLOCKIDX:
        return "blockIdx";
    case CUDA_OBJ_BLOCKDIM:
        return "blockDim";
    case CUDA_OBJ_GRIDDIM:
        return "gridDim";
    default:
        return "unknown";
    }
}

static const char *get_unary_op_symbol(uint32_t op_type)
{
    return NULL;
}

static const char *get_cuda_type_str(dtype_t type, dtype_t second_dtype)
{
    static char buffer[256];

    if (type == LIST)
    {
        const char *base_type;

        switch (second_dtype)
        {
        case FLOAT32:
            base_type = "float";
            break;
        case FLOAT64:
            base_type = "double";
            break;
        case INT32:
            base_type = "int";
            break;
        case INT64:
            base_type = "long long";
            break;
        case UINT32:
            base_type = "unsigned int";
            break;
        case UINT64:
            base_type = "unsigned long long";
            break;
        case BOOL:
            base_type = "bool";
            break;
        default:
            base_type = "void";
            break;
        }

        snprintf(buffer, sizeof(buffer), "%s*", base_type);
        return buffer;
    }

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
    case UINT32:
        return "unsigned int";
    case UINT64:
        return "unsigned long long";
    case BOOL:
        return "bool";
    case LIST:
        return "void*";
    default:
        return "void";
    }
}

static cuda_function_match_t find_cuda_function_by_type(
    const char *php_name,
    dtype_t arg_types[],
    uint32_t num_args)
{
    cuda_function_match_t result = {NULL, DTYPE_UNKNOWN};

    for (int i = 0; cuda_functions[i].php_name != NULL; i++)
    {
        const cuda_function_info_t *func = &cuda_functions[i];

        if (strcmp(func->php_name, php_name) != 0)
        {
            continue;
        }

        if (func->num_params != num_args)
        {
            continue;
        }

        dtype_t dominant_type = DTYPE_UNKNOWN;
        for (uint32_t j = 0; j < num_args; j++)
        {
            if (arg_types[j] == FLOAT64)
            {
                dominant_type = FLOAT64;
                break;
            }
            else if (arg_types[j] == FLOAT32)
            {
                if (dominant_type != FLOAT64)
                {
                    dominant_type = FLOAT32;
                }
            }
            else if (arg_types[j] == INT32 || arg_types[j] == INT64)
            {
                if (dominant_type == DTYPE_UNKNOWN)
                {
                    dominant_type = INT32;
                }
            }
        }

        if (dominant_type == FLOAT64 && func->cuda_name_f64 != NULL)
        {
            result.cuda_name = func->cuda_name_f64;
            result.return_type = func->return_type_f64;
            break;
        }
        else if (dominant_type == FLOAT32 && func->cuda_name_f32 != NULL)
        {
            result.cuda_name = func->cuda_name_f32;
            result.return_type = func->return_type_f32;
            break;
        }
        else if ((dominant_type == INT32 || dominant_type == INT64) &&
                 func->cuda_name_i32 != NULL)
        {
            result.cuda_name = func->cuda_name_i32;
            result.return_type = func->return_type_i32;
            break;
        }
    }

    return result;
}

static const cuda_function_info_t *find_cuda_function(const char *php_name)
{
    for (int i = 0; cuda_functions[i].php_name != NULL; i++)
    {
        if (strcmp(cuda_functions[i].php_name, php_name) == 0)
        {
            return &cuda_functions[i];
        }
    }
    return NULL;
}

static func_parameter *find_kernel_parameter(func_parameter_list_t *list, const char *name)
{
    if (!list || !list->parameters)
        return NULL;

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
    for (size_t i = 0; i < sizeof(php_ast_handlers) / sizeof(php_ast_handler); i++)
    {
        if (php_ast_handlers[i].kind == kind)
        {
            return php_ast_handlers[i].fn;
        }
    }
    return NULL;
}

static zend_bool needs_semicolon(zend_ast *ast)
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

int compile_ast_to_cuda_fn(cuda_compilation_context_t *context, zend_ast *ast)
{
    if (generate_function_signature(context) != 1)
    {
        return 0;
    }

    if (!compile_ast_as_valid_cuda(context, ast))
    {
        return 0;
    }

    smart_string_appends(context->cuda_code_buffer, "\n}\n");
    return 1;
}

static int validate_function_parameters(cuda_compilation_context_t *context)
{
    if (!context || !context->parameters)
    {
        return 1;
    }

    for (int i = 0; i < context->parameters->total; i++)
    {
        func_parameter *param_i = context->parameters->parameters[i];

        for (int j = i + 1; j < context->parameters->total; j++)
        {
            func_parameter *param_j = context->parameters->parameters[j];

            if (strcmp(param_i->name, param_j->name) == 0)
            {
                php_error_docref(NULL, E_ERROR,
                                 "Duplicate parameter name: '%s'", param_i->name);
                return 0;
            }
        }

        // switch (context->fn_type) {
        //     case FN_KERNEL:
        //         if (param_i->is_reference) {
        //             php_error_docref(NULL, E_WARNING,
        //                 "Kernel parameter '%s' passed by reference may not work in CUDA",
        //                 param_i->name);
        //         }
        //         break;

        //     case FN_DEVICE:
        //         break;

        //     default:
        //         break;
        // }
    }

    return 1;
}

static int generate_function_signature(cuda_compilation_context_t *context)
{
    if (!context || !context->cuda_code_buffer)
    {
        return 0;
    }

    if (!validate_function_parameters(context))
    {
        return 0;
    }

    const char *qualifier = NULL;
    const char *return_type_str = "void";

    switch (context->fn_type)
    {
    case FN_KERNEL:
        qualifier = "__global__";
        return_type_str = get_cuda_type_str(context->return_dtype, DTYPE_UNKNOWN);
        if (!return_type_str || context->return_dtype == DTYPE_UNKNOWN)
        {
            return_type_str = "void";
        }
        break;

    case FN_DEVICE:
        qualifier = "__device__";
        return_type_str = get_cuda_type_str(context->return_dtype, DTYPE_UNKNOWN);
        if (!return_type_str)
        {
            return_type_str = "void";
        }
        break;

    case FN_GLOBAL:
        qualifier = "";
        return_type_str = get_cuda_type_str(context->return_dtype, DTYPE_UNKNOWN);
        if (!return_type_str)
        {
            return_type_str = "void";
        }
        break;

    default:
        php_error_docref(NULL, E_ERROR, "Invalid CUDA function type");
        return 0;
    }

    if (qualifier && qualifier[0] != '\0')
    {
        smart_string_appends(context->cuda_code_buffer, qualifier);
        smart_string_appendc(context->cuda_code_buffer, ' ');
    }

    smart_string_appends(context->cuda_code_buffer, return_type_str);
    smart_string_appendc(context->cuda_code_buffer, ' ');
    smart_string_appends(context->cuda_code_buffer, ZSTR_VAL(context->name));
    smart_string_appendc(context->cuda_code_buffer, '(');

    if (context->parameters && context->parameters->parameters)
    {
        for (int i = 0; i < context->parameters->total; i++)
        {
            func_parameter *param = context->parameters->parameters[i];

            dtype_t first_type = param->dtype;
            dtype_t second_type = param->second_dtype;

            if (strcmp(param->name, "cuda") == 0)
            {
                php_error_docref(NULL, E_ERROR,
                                 "Parameter '$cuda' is reserved and will be automatically injected");
                return 0;
            }

            const char *type_str = get_cuda_type_str(first_type, second_type);
            if (!type_str)
            {
                php_error_docref(NULL, E_ERROR,
                                 "Invalid type for parameter '%s'", param->name);
                return 0;
            }

            smart_string_appends(context->cuda_code_buffer, type_str);
            smart_string_appendc(context->cuda_code_buffer, ' ');
            smart_string_appends(context->cuda_code_buffer, param->name);
            if (i < context->parameters->total - 1)
            {
                smart_string_appends(context->cuda_code_buffer, ", ");
            }
        }
    }

    smart_string_appends(context->cuda_code_buffer, ") {\n");
    return 1;
}

int compile_ast_as_valid_cuda(cuda_compilation_context_t *context, zend_ast *ast)
{
    if (!ast)
        return 1;

    if (ast->kind != ZEND_AST_STMT_LIST && ast->kind != ZEND_AST_ARG_LIST &&
        ast->kind != ZEND_AST_EXPR_LIST)
    {
        context->last_evaluated_first_dtype = DTYPE_UNKNOWN;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    }

    handler handler_func = get_ast_handler(ast->kind);
    if (!handler_func)
    {

        return handle_not_allowed(context, ast);
    }

    return handler_func(context, ast);
}

static int handle_not_allowed(cuda_compilation_context_t *context, zend_ast *ast)
{
    const char *ast_name = get_ast_kind_name(ast->kind);

    php_error_docref(NULL, E_ERROR,
                     "Kernel compilation failed: PHP construct '%s' is not allowed in CUDA kernels.",
                     ast_name);

    return 0;
}

static int handle_cuda_direct_method(cuda_compilation_context_t *context,
                                     const char *method_name,
                                     zend_ast *args_ast)
{
    if (strcmp(method_name, "threadIdx") == 0)
    {
        if (args_ast && zend_ast_get_num_children(args_ast) > 0)
        {
            php_error_docref(NULL, E_WARNING,
                             "$cuda->threadIdx() doesn't take arguments");
        }

        smart_string_appends(context->cuda_code_buffer, "threadIdx");
        context->last_evaluated_first_dtype = LIST;
        context->last_evaluated_second_dtype = INT32;

        context->current_cuda_object = CUDA_OBJ_THREADIDX;
        return 1;
    }
    else if (strcmp(method_name, "blockIdx") == 0)
    {
        if (args_ast && zend_ast_get_num_children(args_ast) > 0)
        {
            php_error_docref(NULL, E_WARNING,
                             "$cuda->blockIdx() doesn't take arguments");
        }

        smart_string_appends(context->cuda_code_buffer, "blockIdx");
        context->last_evaluated_first_dtype = LIST;
        context->last_evaluated_second_dtype = INT32;

        context->current_cuda_object = CUDA_OBJ_BLOCKIDX;
        return 1;
    }
    else if (strcmp(method_name, "blockDim") == 0)
    {
        if (args_ast && zend_ast_get_num_children(args_ast) > 0)
        {
            php_error_docref(NULL, E_WARNING,
                             "$cuda->blockDim() doesn't take arguments");
        }

        smart_string_appends(context->cuda_code_buffer, "blockDim");
        context->last_evaluated_first_dtype = LIST;
        context->last_evaluated_second_dtype = INT32;

        context->current_cuda_object = CUDA_OBJ_BLOCKDIM;
        return 1;
    }
    else if (strcmp(method_name, "gridDim") == 0)
    {
        if (args_ast && zend_ast_get_num_children(args_ast) > 0)
        {
            php_error_docref(NULL, E_WARNING,
                             "$cuda->gridDim() doesn't take arguments");
        }

        smart_string_appends(context->cuda_code_buffer, "gridDim");
        context->last_evaluated_first_dtype = LIST;
        context->last_evaluated_second_dtype = INT32;

        context->current_cuda_object = CUDA_OBJ_GRIDDIM;
        return 1;
    }

    php_error_docref(NULL, E_ERROR,
                     "Method $cuda->%s() is not supported.", method_name);
    return 0;
}

static int handle_math_method(cuda_compilation_context_t *context,
                              const char *method_name,
                              zend_ast *args_ast)
{
    const cuda_function_info_t *func_info = find_cuda_function(method_name);
    if (!func_info)
    {
        php_error_docref(NULL, E_ERROR,
                         "Method $cuda->math->%s() is not supported.", method_name);
        return 0;
    }

    if (func_info->header)
    {
        add_cuda_header(context, func_info->header);
    }

    const char *cuda_func_name = NULL;
    dtype_t return_type = DTYPE_UNKNOWN;

    if (func_info->cuda_name_f32)
    {
        cuda_func_name = func_info->cuda_name_f32;
        return_type = func_info->return_type_f32;
    }
    else if (func_info->cuda_name_f64)
    {
        cuda_func_name = func_info->cuda_name_f64;
        return_type = func_info->return_type_f64;
    }
    else if (func_info->cuda_name_i32)
    {
        cuda_func_name = func_info->cuda_name_i32;
        return_type = func_info->return_type_i32;
    }

    if (!cuda_func_name)
    {
        php_error_docref(NULL, E_ERROR,
                         "No CUDA implementation for math->%s()", method_name);
        return 0;
    }

    smart_string_appends(context->cuda_code_buffer, cuda_func_name);

    smart_string_appendc(context->cuda_code_buffer, '(');
    if (args_ast && !compile_ast_as_valid_cuda(context, args_ast))
    {
        return 0;
    }
    smart_string_appendc(context->cuda_code_buffer, ')');

    context->last_evaluated_first_dtype = return_type;
    context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    context->current_cuda_object = CUDA_OBJ_NONE;

    return 1;
}

static int handle_sync_method(cuda_compilation_context_t *context,
                              const char *method_name,
                              zend_ast *args_ast)
{
    if (strcmp(method_name, "threads") == 0)
    {
        if (args_ast && zend_ast_get_num_children(args_ast) > 0)
        {
            php_error_docref(NULL, E_WARNING,
                             "$cuda->sync->threads() doesn't take arguments");
        }

        smart_string_appends(context->cuda_code_buffer, "__syncthreads()");
        context->last_evaluated_first_dtype = VOID;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
        context->current_cuda_object = CUDA_OBJ_NONE;
        return 1;
    }
    else if (strcmp(method_name, "warp") == 0)
    {
        if (args_ast && zend_ast_get_num_children(args_ast) > 0)
        {
            php_error_docref(NULL, E_WARNING,
                             "$cuda->sync->warp() doesn't take arguments");
        }

        smart_string_appends(context->cuda_code_buffer, "__syncwarp()");
        context->last_evaluated_first_dtype = VOID;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
        context->current_cuda_object = CUDA_OBJ_NONE;
        return 1;
    }

    php_error_docref(NULL, E_ERROR,
                     "Method $cuda->sync->%s() is not supported.", method_name);
    return 0;
}

static int handle_atomic_method(cuda_compilation_context_t *context,
                                const char *method_name,
                                zend_ast *args_ast)
{
    const char *cuda_func_name = NULL;
    dtype_t return_type = DTYPE_UNKNOWN;

    if (strcmp(method_name, "add") == 0)
    {
        cuda_func_name = "atomicAdd";
        return_type = FLOAT32;
    }
    else if (strcmp(method_name, "sub") == 0)
    {
        cuda_func_name = "atomicSub";
        return_type = FLOAT32;
    }
    else if (strcmp(method_name, "max") == 0)
    {
        cuda_func_name = "atomicMax";
        return_type = INT32;
    }
    else if (strcmp(method_name, "min") == 0)
    {
        cuda_func_name = "atomicMin";
        return_type = INT32;
    }
    else
    {
        php_error_docref(NULL, E_ERROR,
                         "Method $cuda->atomic->%s() is not supported.", method_name);
        return 0;
    }

    smart_string_appends(context->cuda_code_buffer, cuda_func_name);

    smart_string_appendc(context->cuda_code_buffer, '(');
    if (args_ast && !compile_ast_as_valid_cuda(context, args_ast))
    {
        return 0;
    }
    smart_string_appendc(context->cuda_code_buffer, ')');

    context->last_evaluated_first_dtype = return_type;
    context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    context->current_cuda_object = CUDA_OBJ_NONE;

    return 1;
}

static int handle_memory_method(cuda_compilation_context_t *context,
                                const char *method_name,
                                zend_ast *args_ast)
{
    if (strcmp(method_name, "shared") == 0)
    {
        smart_string_appends(context->cuda_code_buffer, "shared_memory");

        smart_string_appendc(context->cuda_code_buffer, '(');
        if (args_ast && !compile_ast_as_valid_cuda(context, args_ast))
        {
            return 0;
        }
        smart_string_appendc(context->cuda_code_buffer, ')');

        context->last_evaluated_first_dtype = LIST;
        context->last_evaluated_second_dtype = FLOAT32;
        context->current_cuda_object = CUDA_OBJ_NONE;
        return 1;
    }
    else if (strcmp(method_name, "global") == 0)
    {
        php_error_docref(NULL, E_ERROR,
                         "$cuda->memory->global() not implemented yet.");
        return 0;
    }

    php_error_docref(NULL, E_ERROR,
                     "Method $cuda->memory->%s() is not supported.", method_name);
    return 0;
}

static int handler_ast_method_call(cuda_compilation_context_t *context, zend_ast *ast)
{
    uint32_t num_children = zend_ast_get_num_children(ast);
    if (num_children < 3)
    {
        php_error_docref(NULL, E_ERROR, "Method call missing parts.");
        return 0;
    }

    zend_ast *object_ast = ast->child[0];
    zend_ast *method_name_ast = ast->child[1];
    zend_ast *args_ast = ast->child[2];

    if (object_ast->kind == ZEND_AST_VAR)
    {
        zend_ast *var_name_ast = object_ast->child[0];
        if (var_name_ast->kind == ZEND_AST_ZVAL)
        {
            zval *obj_zv = zend_ast_get_zval(var_name_ast);
            if (obj_zv && Z_TYPE_P(obj_zv) == IS_STRING)
            {
                zend_string *obj_name = Z_STR_P(obj_zv);
                if (zend_string_equals_literal(obj_name, "cuda"))
                {
                    context->current_cuda_object = CUDA_OBJ_CUDA;
                }
                else
                {
                    php_error_docref(NULL, E_ERROR,
                                     "Only $cuda object calls are allowed, got $%s",
                                     ZSTR_VAL(obj_name));
                    return 0;
                }
            }
        }
    }
    else if (object_ast->kind == ZEND_AST_PROP)
    {
        if (!handler_ast_prop(context, object_ast))
        {
            return 0;
        }
    }
    else if (object_ast->kind == ZEND_AST_METHOD_CALL)
    {
        if (!handler_ast_method_call(context, object_ast))
        {
            return 0;
        }
    }
    else
    {
        php_error_docref(NULL, E_ERROR,
                         "Invalid object for method call.");
        return 0;
    }

    if (method_name_ast->kind != ZEND_AST_ZVAL)
    {
        php_error_docref(NULL, E_ERROR, "Method name must be a literal.");
        return 0;
    }

    zval *method_zv = zend_ast_get_zval(method_name_ast);
    if (!method_zv || Z_TYPE_P(method_zv) != IS_STRING)
    {
        php_error_docref(NULL, E_ERROR, "Method name must be a string.");
        return 0;
    }

    zend_string *method_name = Z_STR_P(method_zv);
    const char *method_name_c = ZSTR_VAL(method_name);

    switch (context->current_cuda_object)
    {
    case CUDA_OBJ_CUDA:
        return handle_cuda_direct_method(context, method_name_c, args_ast);

    case CUDA_OBJ_MATH:
        return handle_math_method(context, method_name_c, args_ast);

    case CUDA_OBJ_ATOMIC:
        return handle_atomic_method(context, method_name_c, args_ast);

    case CUDA_OBJ_MEMORY:
        return handle_memory_method(context, method_name_c, args_ast);

    case CUDA_OBJ_SYNC:
        return handle_sync_method(context, method_name_c, args_ast);

    case CUDA_OBJ_THREADIDX:
    case CUDA_OBJ_BLOCKIDX:
    case CUDA_OBJ_BLOCKDIM:
    case CUDA_OBJ_GRIDDIM:
        php_error_docref(NULL, E_ERROR,
                         "Use property access (->x) not method call for CUDA object members.");
        return 0;

    default:
        php_error_docref(NULL, E_ERROR, "Invalid CUDA object access.");
        return 0;
    }
}

static int handler_ast_allowed_simple(cuda_compilation_context_t *context, zend_ast *ast)
{
    uint32_t children = zend_ast_get_num_children(ast);
    for (uint32_t i = 0; i < children; i++)
    {
        if (!compile_ast_as_valid_cuda(context, ast->child[i]))
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
        if (!compile_ast_as_valid_cuda(context, list->child[i]))
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
        if (!compile_ast_as_valid_cuda(context, stmt))
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
        php_error_docref(NULL, E_ERROR, "Complex variable names are not allowed.");
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
    if (strcmp(name_c, "cuda") == 0)
    {
        context->current_cuda_object = CUDA_OBJ_CUDA;
        return 1;
    }
    if (strcmp(name_c, "threadIdx") == 0)
    {
        context->current_cuda_object = CUDA_OBJ_THREADIDX;
        smart_string_appends(context->cuda_code_buffer, "threadIdx");
        context->last_evaluated_first_dtype = LIST;
        context->last_evaluated_second_dtype = INT32;
        return 1;
    }

    if (strcmp(name_c, "blockIdx") == 0)
    {
        context->current_cuda_object = CUDA_OBJ_BLOCKIDX;
        smart_string_appends(context->cuda_code_buffer, "blockIdx");
        context->last_evaluated_first_dtype = LIST;
        context->last_evaluated_second_dtype = INT32;
        return 1;
    }

    if (strcmp(name_c, "blockDim") == 0)
    {
        context->current_cuda_object = CUDA_OBJ_BLOCKDIM;
        smart_string_appends(context->cuda_code_buffer, "blockDim");
        context->last_evaluated_first_dtype = LIST;
        context->last_evaluated_second_dtype = INT32;
        return 1;
    }

    if (strcmp(name_c, "gridDim") == 0)
    {
        context->current_cuda_object = CUDA_OBJ_GRIDDIM;
        smart_string_appends(context->cuda_code_buffer, "gridDim");
        context->last_evaluated_first_dtype = LIST;
        context->last_evaluated_second_dtype = INT32;
        return 1;
    }

    func_parameter *param = find_kernel_parameter(context->parameters, name_c);
    if (param)
    {
        context->last_evaluated_first_dtype = param->dtype;
        context->last_evaluated_second_dtype = param->second_dtype;

        smart_string_appendl(context->cuda_code_buffer, name_c, ZSTR_LEN(var_name_zend));
        return 1;
    }

    local_variable_t *local = zend_hash_find_ptr(&context->local_variables, var_name_zend);
    if (local)
    {
        context->last_evaluated_first_dtype = local->dtype;
        context->last_evaluated_second_dtype = local->second_dtype;

        smart_string_appendl(context->cuda_code_buffer, name_c, ZSTR_LEN(var_name_zend));
        return 1;
    }

    php_error_docref(NULL, E_ERROR,
                     "Undefined variable '$%s'. Variable must be a parameter or previously defined.",
                     name_c);
    return 0;
}

static int handler_ast_assign(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast *lvalue = ast->child[0];
    zend_ast *rvalue = ast->child[1];

    smart_string rvalue_buffer = {0};
    smart_string_alloc(&rvalue_buffer, 256, 0);

    smart_string *original_buffer = context->cuda_code_buffer;
    context->cuda_code_buffer = &rvalue_buffer;

    if (!compile_ast_as_valid_cuda(context, rvalue))
    {
        smart_string_free(&rvalue_buffer);
        context->cuda_code_buffer = original_buffer;
        return 0;
    }

    dtype_t rvalue_type = context->last_evaluated_first_dtype;
    dtype_t rvalue_second_type = context->last_evaluated_second_dtype;

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
                php_error_docref(NULL, E_ERROR,
                                 "Cannot infer type for new variable '$%s'.", name_c);
                return 0;
            }

            local_variable_t *new_var = (local_variable_t *)ecalloc(1, sizeof(local_variable_t));
            new_var->name = zend_string_copy(var_name_zend);
            new_var->dtype = rvalue_type;
            new_var->second_dtype = rvalue_second_type;

            zend_hash_add_ptr(&context->local_variables, var_name_zend, new_var);

            smart_string_appends(context->cuda_code_buffer, get_cuda_type_str(rvalue_type, rvalue_second_type));
            smart_string_appendc(context->cuda_code_buffer, ' ');
            smart_string_appendl(context->cuda_code_buffer, name_c, ZSTR_LEN(var_name_zend));
        }
        else
        {
            dtype_t lvalue_type = param ? param->dtype : local->dtype;
            dtype_t lvalue_second_type = param ? param->second_dtype : local->second_dtype;

            if (lvalue_type != rvalue_type && rvalue_type != DTYPE_UNKNOWN)
            {
                if (lvalue_type != rvalue_type)
                {
                    smart_string_free(&rvalue_buffer);
                    php_error_docref(NULL, E_ERROR,
                                     "Type mismatch for '$%s'. Expected %s, got %s.",
                                     name_c, get_cuda_type_str(lvalue_type, DTYPE_UNKNOWN), get_cuda_type_str(rvalue_type, DTYPE_UNKNOWN));
                    return 0;
                }

                if (lvalue_second_type != rvalue_second_type && rvalue_type != DTYPE_UNKNOWN)
                {
                    smart_string_free(&rvalue_buffer);

                    if (lvalue_second_type == LIST && rvalue_second_type != LIST)
                    {
                        php_error_docref(NULL, E_ERROR,
                                         "Type mismatch for '$%s'. Expected Array[%s], got %s.",
                                         name_c, get_cuda_type_str(lvalue_type, DTYPE_UNKNOWN), get_cuda_type_str(rvalue_type, DTYPE_UNKNOWN));
                        return 0;
                    }
                    else if (lvalue_second_type != LIST && rvalue_second_type == LIST)
                    {
                        php_error_docref(NULL, E_ERROR,
                                         "Type mismatch for '$%s'. Expected %s, got Array[%s].",
                                         name_c, get_cuda_type_str(lvalue_type, DTYPE_UNKNOWN), get_cuda_type_str(rvalue_type, DTYPE_UNKNOWN));
                        return 0;
                    }
                }
            }

            smart_string_appendl(context->cuda_code_buffer, name_c, ZSTR_LEN(var_name_zend));
        }
    }
    else if (lvalue->kind == ZEND_AST_DIM)
    {
        if (!compile_ast_as_valid_cuda(context, lvalue))
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

static int handler_ast_assign_op(cuda_compilation_context_t *context, zend_ast *ast)
{
    const char *op_symbol = get_assign_op_symbol(ast->attr);
    if (!op_symbol)
    {
        php_error_docref(NULL, E_ERROR, "Assignment operator %d not supported.", ast->attr);
        return 0;
    }

    zend_ast *lvalue = ast->child[0];
    zend_ast *rvalue = ast->child[1];

    if (!compile_ast_as_valid_cuda(context, lvalue))
    {
        return 0;
    }

    smart_string_appends(context->cuda_code_buffer, op_symbol);

    if (!compile_ast_as_valid_cuda(context, rvalue))
    {
        return 0;
    }

    return 1;
}

static int handler_ast_if(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast_list *list = (zend_ast_list *)ast;
    for (uint32_t i = 0; i < list->children; i++)
    {
        if (!compile_ast_as_valid_cuda(context, list->child[i]))
        {
            return 0;
        }
    }
    return 1;
}

static int handle_ast_if_elem(cuda_compilation_context_t *context, zend_ast *ast)
{
    uint32_t num_children = zend_ast_get_num_children(ast);

    zend_ast *cond = ast->child[0];
    zend_ast *stmt = ast->child[1];
    zend_ast *else_stmt = (num_children > 2) ? ast->child[2] : NULL;
    ;

    if (num_children < 2)
    {
        return 0;
    }

    smart_string_appends(context->cuda_code_buffer, "if (");
    if (!compile_ast_as_valid_cuda(context, cond))
    {
        return 0;
    }
    smart_string_appends(context->cuda_code_buffer, ") {\n");

    if (!compile_ast_as_valid_cuda(context, stmt))
    {
        return 0;
    }

    smart_string_appends(context->cuda_code_buffer, "}\n");

    if (else_stmt)
    {
        if (else_stmt->kind == ZEND_AST_IF)
        {
            smart_string_appends(context->cuda_code_buffer, "else ");
            if (!handler_ast_if(context, else_stmt))
            {
                return 0;
            }
        }
        else
        {
            smart_string_appends(context->cuda_code_buffer, "else {\n");
            if (!compile_ast_as_valid_cuda(context, else_stmt))
            {
                return 0;
            }
            smart_string_appends(context->cuda_code_buffer, "}\n");
        }
    }

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
        if (!compile_ast_as_valid_cuda(context, init_node))
        {
            return 0;
        }
    }
    smart_string_appends(context->cuda_code_buffer, "; ");

    if (cond_node)
    {
        if (!compile_ast_as_valid_cuda(context, cond_node))
        {
            return 0;
        }
    }
    smart_string_appends(context->cuda_code_buffer, "; ");

    if (loop_node)
    {
        if (!compile_ast_as_valid_cuda(context, loop_node))
        {
            return 0;
        }
    }

    smart_string_appends(context->cuda_code_buffer, ") {\n");

    context->loop_depth++;
    if (body_node)
    {
        if (!compile_ast_as_valid_cuda(context, body_node))
        {
            context->loop_depth--;
            return 0;
        }
    }
    context->loop_depth--;

    smart_string_appends(context->cuda_code_buffer, "}\n");
    return 1;
}

static int handler_ast_while(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast *cond = ast->child[0];
    zend_ast *body = ast->child[1];

    smart_string_appends(context->cuda_code_buffer, "while (");
    if (!compile_ast_as_valid_cuda(context, cond))
    {
        return 0;
    }
    smart_string_appends(context->cuda_code_buffer, ") {\n");

    context->loop_depth++;
    if (body && !compile_ast_as_valid_cuda(context, body))
    {
        context->loop_depth--;
        return 0;
    }
    context->loop_depth--;

    smart_string_appends(context->cuda_code_buffer, "}\n");
    return 1;
}

static int handler_ast_do_while(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast *body = ast->child[0];
    zend_ast *cond = ast->child[1];

    smart_string_appends(context->cuda_code_buffer, "do {\n");

    context->loop_depth++;
    if (body && !compile_ast_as_valid_cuda(context, body))
    {
        context->loop_depth--;
        return 0;
    }
    context->loop_depth--;

    smart_string_appends(context->cuda_code_buffer, "} while (");
    if (!compile_ast_as_valid_cuda(context, cond))
    {
        return 0;
    }
    smart_string_appends(context->cuda_code_buffer, ");\n");

    return 1;
}

static int handler_ast_return(cuda_compilation_context_t *context, zend_ast *ast)
{
    smart_string_appends(context->cuda_code_buffer, "return ");

    if (ast->child[0])
    {
        if (!compile_ast_as_valid_cuda(context, ast->child[0]))
        {
            return 0;
        }
    }

    return 1;
}

static int handler_ast_dim(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast *array_expr = ast->child[0];
    zend_ast *index_expr = ast->child[1];

    if (!compile_ast_as_valid_cuda(context, array_expr))
    {
        return 0;
    }

    dtype_t first_dtype = context->last_evaluated_first_dtype;
    dtype_t second_dtype = context->last_evaluated_second_dtype;

    smart_string_appendc(context->cuda_code_buffer, '[');
    if (!compile_ast_as_valid_cuda(context, index_expr))
    {
        return 0;
    }

    context->last_evaluated_first_dtype = second_dtype;
    context->last_evaluated_second_dtype = DTYPE_UNKNOWN;

    smart_string_appendc(context->cuda_code_buffer, ']');
    return 1;
}

static int handler_ast_zval(cuda_compilation_context_t *context, zend_ast *ast)
{
    zval *zv = zend_ast_get_zval(ast);

    if (!zv)
    {
        php_error_docref(NULL, E_ERROR, "Invalid ZVAL AST node.");
        return 0;
    }

    switch (Z_TYPE_P(zv))
    {
    case IS_LONG:
    {
        smart_string_append_long(context->cuda_code_buffer, Z_LVAL_P(zv));
        context->last_evaluated_first_dtype = INT32;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;

        break;
    }
    case IS_DOUBLE:
    {
        char buffer[64];
        double value = Z_DVAL_P(zv);
        if (context->last_evaluated_first_dtype == FLOAT64)
        {
            snprintf(buffer, sizeof(buffer), "%.17g", value);
            context->last_evaluated_first_dtype = FLOAT64;
            context->last_evaluated_second_dtype = 0;
        }
        else
        {
            snprintf(buffer, sizeof(buffer), "%.9g", value);

            size_t len = strlen(buffer);
            int needs_f = 1;
            for (size_t i = 0; i < len; i++)
            {
                if (buffer[i] == '.' || buffer[i] == 'e' || buffer[i] == 'E')
                {
                    needs_f = 0;
                    break;
                }
            }

            if (needs_f && len < sizeof(buffer) - 2)
            {
                strcat(buffer, ".0");
            }

            strcat(buffer, "f");
            context->last_evaluated_first_dtype = FLOAT32;
            context->last_evaluated_second_dtype = 0;
        }

        smart_string_appends(context->cuda_code_buffer, buffer);
        break;
    }
    case IS_TRUE:
        smart_string_appends(context->cuda_code_buffer, "true");
        context->last_evaluated_first_dtype = BOOL;
        context->last_evaluated_second_dtype = 0;

        break;
    case IS_FALSE:
        smart_string_appends(context->cuda_code_buffer, "false");
        context->last_evaluated_first_dtype = BOOL;
        context->last_evaluated_second_dtype = 0;

        break;
    case IS_STRING:
    {
        context->last_evaluated_second_dtype = 0;
        zend_string *str = Z_STR_P(zv);
        const char *str_val = ZSTR_VAL(str);

        int is_simple_var = 1;
        for (size_t j = 0; j < ZSTR_LEN(str); j++)
        {
            char c = str_val[j];
            if (j == 0)
            {
                if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'))
                {
                    is_simple_var = 0;
                    break;
                }
            }
            else
            {
                if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                      (c >= '0' && c <= '9') || c == '_'))
                {
                    is_simple_var = 0;
                    break;
                }
            }
        }

        if (is_simple_var && ZSTR_LEN(str) <= 32)
        {
            smart_string_appendl(context->cuda_code_buffer, str_val, ZSTR_LEN(str));

            func_parameter *param = find_kernel_parameter(context->parameters, str_val);
            if (param)
            {
                context->last_evaluated_first_dtype = param->dtype;
                context->last_evaluated_second_dtype = param->second_dtype;
            }
            else
            {
                context->last_evaluated_first_dtype = DTYPE_UNKNOWN;
                context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
            }
        }
        else if (ZSTR_LEN(str) < 32)
        {
            smart_string_appendc(context->cuda_code_buffer, '"');
            smart_string_appendl(context->cuda_code_buffer, str_val, ZSTR_LEN(str));
            smart_string_appendc(context->cuda_code_buffer, '"');
            context->last_evaluated_first_dtype = DTYPE_UNKNOWN;
        }
        else
        {
            php_error_docref(NULL, E_ERROR, "String literals too long for CUDA kernel.");
            return 0;
        }
        break;
    }
    case IS_NULL:
        smart_string_appends(context->cuda_code_buffer, "NULL");
        context->last_evaluated_first_dtype = DTYPE_UNKNOWN;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;

        break;
    default:
        php_error_docref(NULL, E_ERROR, "Literal type %d is not allowed in CUDA kernel.", Z_TYPE_P(zv));
        return 0;
    }

    return 1;
}

static int handler_ast_prop(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast *obj_ast = ast->child[0];
    zend_ast *prop_ast = ast->child[1];

    if (!compile_ast_as_valid_cuda(context, obj_ast))
    {
        return 0;
    }

    if (prop_ast->kind != ZEND_AST_ZVAL)
    {
        php_error_docref(NULL, E_ERROR, "Property name must be literal.");
        return 0;
    }

    zval *prop_zv = zend_ast_get_zval(prop_ast);
    if (!prop_zv || Z_TYPE_P(prop_zv) != IS_STRING)
    {
        php_error_docref(NULL, E_ERROR, "Invalid property name.");
        return 0;
    }

    zend_string *prop_name = Z_STR_P(prop_zv);
    const char *prop_name_c = ZSTR_VAL(prop_name);

    if (context->current_cuda_object == CUDA_OBJ_CUDA)
    {
        if (strcmp(prop_name_c, "math") == 0)
        {
            context->current_cuda_object = CUDA_OBJ_MATH;
            return 1;
        }
        else if (strcmp(prop_name_c, "atomic") == 0)
        {
            context->current_cuda_object = CUDA_OBJ_ATOMIC;
            return 1;
        }
        else if (strcmp(prop_name_c, "memory") == 0)
        {
            context->current_cuda_object = CUDA_OBJ_MEMORY;
            return 1;
        }
        else if (strcmp(prop_name_c, "sync") == 0)
        {
            context->current_cuda_object = CUDA_OBJ_SYNC;
            return 1;
        }
        else
        {
            php_error_docref(NULL, E_ERROR,
                             "Invalid property '%s' on $cuda object.", prop_name_c);
            return 0;
        }
    }
    else if (context->current_cuda_object == CUDA_OBJ_THREADIDX ||
             context->current_cuda_object == CUDA_OBJ_BLOCKIDX ||
             context->current_cuda_object == CUDA_OBJ_BLOCKDIM ||
             context->current_cuda_object == CUDA_OBJ_GRIDDIM)
    {
        const char *valid_members[] = {"x", "y", "z", "width", "height", "depth"};
        int valid = 0;
        for (int i = 0; i < 6; i++)
        {
            if (strcmp(prop_name_c, valid_members[i]) == 0)
            {
                valid = 1;
                break;
            }
        }

        if (!valid)
        {
            php_error_docref(NULL, E_ERROR,
                             "Invalid member '%s' for CUDA object.", prop_name_c);
            return 0;
        }

        smart_string_appendc(context->cuda_code_buffer, '.');
        smart_string_appends(context->cuda_code_buffer, prop_name_c);

        context->last_evaluated_first_dtype = INT32;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;

        context->current_cuda_object = CUDA_OBJ_NONE;

        return 1;
    }
    else if (context->current_cuda_object == CUDA_OBJ_MATH ||
             context->current_cuda_object == CUDA_OBJ_ATOMIC ||
             context->current_cuda_object == CUDA_OBJ_MEMORY ||
             context->current_cuda_object == CUDA_OBJ_SYNC)
    {

        php_error_docref(NULL, E_ERROR,
                         "Method call expected after %s object.",
                         get_cuda_object_name(context->current_cuda_object));
        return 0;
    }
    else
    {
        php_error_docref(NULL, E_ERROR,
                         "Property access not allowed in this context.");
        return 0;
    }
}

static int handler_ast_binary_op(cuda_compilation_context_t *context, zend_ast *ast)
{
    if (ast->attr == ZEND_NOP)
    {
        return 1;
    }

    const char *op_symbol = get_binary_op_symbol(ast->attr);
    if (!op_symbol)
    {
        php_error_docref(NULL, E_ERROR, "Binary operator %d not supported in CUDA.", ast->attr);
        return 0;
    }

    smart_string_appendc(context->cuda_code_buffer, '(');

    if (!compile_ast_as_valid_cuda(context, ast->child[0]))
    {
        return 0;
    }
    dtype_t left_type = context->last_evaluated_first_dtype;
    dtype_t left_second_type = context->last_evaluated_second_dtype;

    smart_string_appends(context->cuda_code_buffer, op_symbol);

    if (!compile_ast_as_valid_cuda(context, ast->child[1]))
    {
        return 0;
    }

    dtype_t right_type = context->last_evaluated_first_dtype;
    dtype_t right_second_type = context->last_evaluated_second_dtype;

    if (right_type == LIST && left_type != LIST)
    {
        php_error_docref(NULL, E_ERROR,
                         "Type mismatch Expected %s, got Array[%s].",
                         get_cuda_type_str(left_type, DTYPE_UNKNOWN), get_cuda_type_str(right_type, right_second_type));
        return 0;
    }

    if (left_type == LIST && right_type != LIST)
    {
        php_error_docref(NULL, E_ERROR,
                         "Type mismatch Expected Array[%s], got %s.",
                         get_cuda_type_str(right_type, DTYPE_UNKNOWN), get_cuda_type_str(left_type, left_second_type));
        return 0;
    }

    /**
     * @todo maybe we can see if right and left type can be casted
     */

    smart_string_appendc(context->cuda_code_buffer, ')');

    if (ast->attr == ZEND_ADD || ast->attr == ZEND_SUB ||
        ast->attr == ZEND_MUL || ast->attr == ZEND_DIV)
    {

        if (left_type == FLOAT64 || right_type == FLOAT64)
        {
            context->last_evaluated_first_dtype = FLOAT64;
        }
        else if (left_type == FLOAT32 || right_type == FLOAT32)
        {
            context->last_evaluated_first_dtype = FLOAT32;
        }
        else if (left_type == INT64 || right_type == INT64)
        {
            context->last_evaluated_first_dtype = INT64;
        }
        else if (left_type == INT32 || right_type == INT32)
        {
            context->last_evaluated_first_dtype = INT32;
        }
        else
        {
            context->last_evaluated_first_dtype = DTYPE_UNKNOWN;
        }
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    }
    else if (ast->attr == ZEND_IS_EQUAL || ast->attr == ZEND_IS_NOT_EQUAL ||
             ast->attr == ZEND_IS_IDENTICAL || ast->attr == ZEND_IS_NOT_IDENTICAL ||
             ast->attr == ZEND_IS_SMALLER || ast->attr == ZEND_IS_SMALLER_OR_EQUAL)
    {
        context->last_evaluated_first_dtype = BOOL;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    }

    return 1;
}

static int handler_ast_unary_op(cuda_compilation_context_t *context, zend_ast *ast)
{
    const char *op_symbol = get_unary_op_symbol(ast->attr);
    if (!op_symbol)
    {
        php_error_docref(NULL, E_ERROR, "Unary operator %d not supported in CUDA.", ast->attr);
        return 0;
    }

    smart_string_appendc(context->cuda_code_buffer, '(');
    smart_string_appends(context->cuda_code_buffer, op_symbol);

    if (!compile_ast_as_valid_cuda(context, ast->child[0]))
    {
        return 0;
    }

    smart_string_appendc(context->cuda_code_buffer, ')');

    if (ast->attr == ZEND_BOOL_NOT)
    {
        context->last_evaluated_first_dtype = BOOL;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    }

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
        php_error_docref(NULL, E_ERROR, "Comparison operator not supported.");
        return 0;
    }

    smart_string_appendc(context->cuda_code_buffer, '(');
    if (!compile_ast_as_valid_cuda(context, ast->child[0]))
    {
        return 0;
    }
    smart_string_appends(context->cuda_code_buffer, op_symbol);
    if (!compile_ast_as_valid_cuda(context, ast->child[1]))
    {
        return 0;
    }
    smart_string_appendc(context->cuda_code_buffer, ')');

    context->last_evaluated_first_dtype = BOOL;
    context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    return 1;
}

static int handler_ast_cast(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_uchar cast_type = (zend_uchar)ast->attr;
    const char *cast_str = NULL;
    dtype_t target_type = DTYPE_UNKNOWN;

    switch (cast_type)
    {
    case IS_LONG:
        cast_str = "(int)";
        target_type = INT32;
        break;
    case IS_DOUBLE:
        cast_str = "(float)";
        target_type = FLOAT32;
        break;
    case _IS_NUMBER:
        cast_str = "";
        target_type = FLOAT32;
        break;
    default:
        php_error_docref(NULL, E_ERROR, "Cast type %d not supported in CUDA.", cast_type);
        return 0;
    }

    smart_string_appends(context->cuda_code_buffer, cast_str);
    smart_string_appendc(context->cuda_code_buffer, '(');

    if (!compile_ast_as_valid_cuda(context, ast->child[0]))
    {
        return 0;
    }

    smart_string_appendc(context->cuda_code_buffer, ')');
    context->last_evaluated_first_dtype = target_type;
    context->last_evaluated_second_dtype = DTYPE_UNKNOWN;

    return 1;
}

static int handler_ast_conditional(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast *cond = ast->child[0];
    zend_ast *true_expr = ast->child[1];
    zend_ast *false_expr = ast->child[2];

    smart_string_appendc(context->cuda_code_buffer, '(');

    if (!compile_ast_as_valid_cuda(context, cond))
    {
        return 0;
    }

    smart_string_appends(context->cuda_code_buffer, " ? ");

    if (!compile_ast_as_valid_cuda(context, true_expr))
    {
        return 0;
    }
    dtype_t true_type = context->last_evaluated_first_dtype;
    dtype_t true_second_type = context->last_evaluated_second_dtype;

    smart_string_appends(context->cuda_code_buffer, " : ");

    if (!compile_ast_as_valid_cuda(context, false_expr))
    {
        return 0;
    }
    dtype_t false_type = context->last_evaluated_first_dtype;
    dtype_t false_second_type = context->last_evaluated_second_dtype;

    if (false_type == LIST && true_type != LIST)
    {
        php_error_docref(NULL, E_ERROR,
                         "Type mismatch Expected %s, got Array[%s].",
                         get_cuda_type_str(false_type, false_second_type), get_cuda_type_str(true_type, DTYPE_UNKNOWN));
        return 0;
    }

    if (true_type == LIST && false_type != LIST)
    {
        php_error_docref(NULL, E_ERROR,
                         "Type mismatch Expected Array[%s], got %s.",
                         get_cuda_type_str(true_type, true_second_type), get_cuda_type_str(false_type, DTYPE_UNKNOWN));
        return 0;
    }

    smart_string_appendc(context->cuda_code_buffer, ')');

    if (true_type == false_type)
    {
        context->last_evaluated_first_dtype = true_type == LIST ? true_second_type : true_type;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    }
    else if (true_type == FLOAT64 || false_type == FLOAT64)
    {
        context->last_evaluated_first_dtype = FLOAT64;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    }
    else if (true_type == FLOAT32 || false_type == FLOAT32)
    {
        context->last_evaluated_first_dtype = FLOAT32;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    }
    else
    {
        context->last_evaluated_first_dtype = true_type == LIST ? true_second_type : true_type;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    }

    return 1;
}

static int handler_ast_switch(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast *expr = ast->child[0];
    zend_ast *cases = ast->child[1];

    smart_string_appends(context->cuda_code_buffer, "switch (");

    if (!compile_ast_as_valid_cuda(context, expr))
    {
        return 0;
    }

    smart_string_appends(context->cuda_code_buffer, ") {\n");

    if (cases && !compile_ast_as_valid_cuda(context, cases))
    {
        return 0;
    }

    smart_string_appends(context->cuda_code_buffer, "}\n");
    return 1;
}

static int handler_ast_switch_case(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast *cond = ast->child[0];
    zend_ast *stmt = ast->child[1];

    if (cond)
    {
        smart_string_appends(context->cuda_code_buffer, "case ");
        if (!compile_ast_as_valid_cuda(context, cond))
        {
            return 0;
        }
        smart_string_appends(context->cuda_code_buffer, ":\n");
    }
    else
    {
        smart_string_appends(context->cuda_code_buffer, "default:\n");
    }

    if (stmt && !compile_ast_as_valid_cuda(context, stmt))
    {
        return 0;
    }

    smart_string_appends(context->cuda_code_buffer, "break;\n");
    return 1;
}

static int handler_ast_break_continue(cuda_compilation_context_t *context, zend_ast *ast)
{
    if (context->loop_depth <= 0)
    {
        php_error_docref(NULL, E_ERROR, "break/continue outside of loop.");
        return 0;
    }

    if (ast->kind == ZEND_AST_BREAK)
    {
        smart_string_appends(context->cuda_code_buffer, "break");
    }
    else
    {
        smart_string_appends(context->cuda_code_buffer, "continue");
    }

    if (ast->child[0])
    {
        php_error_docref(NULL, E_WARNING, "Labels with break/continue not supported in CUDA.");
    }

    return 1;
}

static int handler_ast_inc_dec(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_ast *var_ast = ast->child[0];

    if (var_ast->kind != ZEND_AST_VAR && var_ast->kind != ZEND_AST_DIM)
    {
        php_error_docref(NULL, E_ERROR,
                         "Increment/decrement target must be a variable or array element.");
        return 0;
    }

    const char *op = NULL;
    zend_bool is_pre = 0;

    switch (ast->kind)
    {
    case ZEND_AST_PRE_INC:
        op = "++";
        is_pre = 1;
        break;
    case ZEND_AST_PRE_DEC:
        op = "--";
        is_pre = 1;
        break;
    case ZEND_AST_POST_INC:
        op = "++";
        is_pre = 0;
        break;
    case ZEND_AST_POST_DEC:
        op = "--";
        is_pre = 0;
        break;
    default:
        return 0;
    }

    if (is_pre)
    {
        smart_string_appends(context->cuda_code_buffer, op);
    }

    if (!compile_ast_as_valid_cuda(context, var_ast))
    {
        return 0;
    }

    if (!is_pre)
    {
        smart_string_appends(context->cuda_code_buffer, op);
    }

    return 1;
}

char *generate_cuda_headers(HashTable *cuda_headers)
{
    if (!cuda_headers || zend_hash_num_elements(cuda_headers) == 0)
    {
        return estrdup("");
    }

    smart_string headers = {0};
    smart_string_alloc(&headers, 256, 0);

    zend_string *header;
    ZEND_HASH_FOREACH_STR_KEY(cuda_headers, header)
    {
        smart_string_appends(&headers, "#include <");
        smart_string_appends(&headers, ZSTR_VAL(header));
        smart_string_appends(&headers, ">\n");
    }
    ZEND_HASH_FOREACH_END();

    smart_string_0(&headers);
    char *result = estrdup(headers.c);
    smart_string_free(&headers);

    return result;
}

static void destroy_local_variable(zval *zv)
{
    if (Z_TYPE_P(zv) == IS_PTR)
    {
        local_variable_t *var = (local_variable_t *)Z_PTR_P(zv);
        if (var)
        {
            if (var->name)
            {
                zend_string_release(var->name);
            }
            efree(var);
        }
    }
}

cuda_compilation_context_t *create_cuda_context(
    func_parameter_list_t *parameters,
    cuda_fn_type fn_type,
    zend_string *name,
    HashTable * headers
)
{
    cuda_compilation_context_t *context =
        (cuda_compilation_context_t *)emalloc(sizeof(cuda_compilation_context_t));

    context->headers = headers;
    context->parameters = parameters;
    context->last_evaluated_first_dtype = DTYPE_UNKNOWN;
    context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    context->current_cuda_object = CUDA_OBJ_NONE;
    context->return_dtype = DTYPE_UNKNOWN;
    context->loop_depth = 0;
    context->name = name;
    context->fn_type = fn_type;
    context->dim_access = 0;
    zend_hash_init(&context->local_variables, 8, NULL, destroy_local_variable, 0);

    context->cuda_code_buffer = (smart_string *)ecalloc(1, sizeof(smart_string));
    smart_string_alloc(context->cuda_code_buffer, 512, 0);

    return context;
}

void free_cuda_context(cuda_compilation_context_t *context)
{
    if (!context)
        return;

    zend_hash_destroy(&context->local_variables);

    if (context->cuda_code_buffer)
    {
        smart_string_free(context->cuda_code_buffer);
        efree(context->cuda_code_buffer);
    }

    if (context->headers)
    {
        zend_hash_destroy(context->headers);
        efree(context->headers);
    }

    if (context->parameters)
    {
        // @todo free context parameters
    }

    efree(context);
}