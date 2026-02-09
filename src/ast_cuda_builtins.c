
#include "ast_cuda_builtins.h"
#include "zend_operators.h"
#include "zend_hash.h"
#include "zend_compile.h"

static const cuda_function_info_t cuda_functions[] = {
    {"max", "fmaxf", "fmax", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 2, {DTYPE_FLOAT32, DTYPE_FLOAT32}, {DTYPE_FLOAT64, DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"min", "fminf", "fmin", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 2, {DTYPE_FLOAT32, DTYPE_FLOAT32}, {DTYPE_FLOAT64, DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"exp", "expf", "exp", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"log", "logf", "log", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"sin", "sinf", "sin", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"cos", "cosf", "cos", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"tan", "tanf", "tan", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"asin", "asinf", "asin", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"acos", "acosf", "acos", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"atan", "atanf", "atan", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"atan2", "atan2f", "atan2", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 2, {DTYPE_FLOAT32, DTYPE_FLOAT32}, {DTYPE_FLOAT64, DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"sinh", "sinhf", "sinh", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"cosh", "coshf", "cosh", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"tanh", "tanhf", "tanh", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"sqrt", "sqrtf", "sqrt", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"pow", "powf", "pow", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 2, {DTYPE_FLOAT32, DTYPE_FLOAT32}, {DTYPE_FLOAT64, DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"abs", NULL, NULL, "abs", DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_INT32, 1, {0}, {0}, {DTYPE_INT32}, "stdlib.h", FUNC_CATEGORY_MATH, 0},
    {"fabs", "fabsf", "fabs", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"ceil", "ceilf", "ceil", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"floor", "floorf", "floor", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"round", "roundf", "round", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"trunc", "truncf", "trunc", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"exp2", "exp2f", "exp2", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"exp10", "exp10f", "exp10", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"log2", "log2f", "log2", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},
    {"log10", "log10f", "log10", NULL, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_UNKNOWN, 1, {DTYPE_FLOAT32}, {DTYPE_FLOAT64}, {0}, "math_functions.h", FUNC_CATEGORY_MATH, 0},

    {"threadIdx", "threadIdx.x", NULL, NULL, DTYPE_INT32, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 0, {0}, {0}, {0}, NULL, FUNC_CATEGORY_SYSTEM, 1},
    {"blockIdx", "blockIdx.x", NULL, NULL, DTYPE_INT32, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 0, {0}, {0}, {0}, NULL, FUNC_CATEGORY_SYSTEM, 1},
    {"blockDim", "blockDim.x", NULL, NULL, DTYPE_INT32, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 0, {0}, {0}, {0}, NULL, FUNC_CATEGORY_SYSTEM, 1},
    {"gridDim", "gridDim.x", NULL, NULL, DTYPE_INT32, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 0, {0}, {0}, {0}, NULL, FUNC_CATEGORY_SYSTEM, 1},

    {"add", NULL, NULL, "atomicAdd", DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 2, {0}, {0}, {0}, NULL, FUNC_CATEGORY_ATOMIC, 1},
    {"sub", NULL, NULL, "atomicSub", DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 2, {0}, {0}, {0}, NULL, FUNC_CATEGORY_ATOMIC, 1},
    {"exch", NULL, NULL, "atomicExch", DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 2, {0}, {0}, {0}, NULL, FUNC_CATEGORY_ATOMIC, 1},
    {"min", NULL, NULL, "atomicMin", DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 2, {0}, {0}, {0}, NULL, FUNC_CATEGORY_ATOMIC, 1},
    {"max", NULL, NULL, "atomicMax", DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 2, {0}, {0}, {0}, NULL, FUNC_CATEGORY_ATOMIC, 1},
    {"inc", NULL, NULL, "atomicInc", DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 2, {0}, {0}, {0}, NULL, FUNC_CATEGORY_ATOMIC, 1},
    {"dec", NULL, NULL, "atomicDec", DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 2, {0}, {0}, {0}, NULL, FUNC_CATEGORY_ATOMIC, 1},
    {"cas", NULL, NULL, "atomicCAS", DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 3, {0}, {0}, {0}, NULL, FUNC_CATEGORY_ATOMIC, 1},
    {"and", NULL, NULL, "atomicAnd", DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 2, {0}, {0}, {0}, NULL, FUNC_CATEGORY_ATOMIC, 1},
    {"or", NULL, NULL, "atomicOr", DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 2, {0}, {0}, {0}, NULL, FUNC_CATEGORY_ATOMIC, 1},
    {"xor", NULL, NULL, "atomicXor", DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 2, {0}, {0}, {0}, NULL, FUNC_CATEGORY_ATOMIC, 1},

    {"threads", NULL, NULL, "__syncthreads", DTYPE_VOID, DTYPE_VOID, DTYPE_VOID, 0, {0}, {0}, {0}, NULL, FUNC_CATEGORY_SYNC, 1},
    {"warp", NULL, NULL, "__syncwarp", DTYPE_VOID, DTYPE_VOID, DTYPE_VOID, 0, {0}, {0}, {0}, NULL, FUNC_CATEGORY_SYNC, 1},
    {"threadsCount", NULL, NULL, "__syncthreads_count", DTYPE_INT32, DTYPE_INT32, DTYPE_INT32, 1, {DTYPE_INT32}, {0}, {0}, NULL, FUNC_CATEGORY_SYNC, 1},
    {"threadsAnd", NULL, NULL, "__syncthreads_and", DTYPE_INT32, DTYPE_INT32, DTYPE_INT32, 1, {DTYPE_INT32}, {0}, {0}, NULL, FUNC_CATEGORY_SYNC, 1},
    {"threadsOr", NULL, NULL, "__syncthreads_or", DTYPE_INT32, DTYPE_INT32, DTYPE_INT32, 1, {DTYPE_INT32}, {0}, {0}, NULL, FUNC_CATEGORY_SYNC, 1},

    {"ballot", NULL, NULL, "__ballot_sync", DTYPE_INT32, DTYPE_INT32, DTYPE_INT32, 1, {DTYPE_INT32}, {0}, {0}, NULL, FUNC_CATEGORY_WARP, 1},
    {"shfl", NULL, NULL, "__shfl_sync", DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 4, {0}, {0}, {0}, NULL, FUNC_CATEGORY_WARP, 1},
    {"shflUp", NULL, NULL, "__shfl_up_sync", DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 4, {0}, {0}, {0}, NULL, FUNC_CATEGORY_WARP, 1},
    {"shflDown", NULL, NULL, "__shfl_down_sync", DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 4, {0}, {0}, {0}, NULL, FUNC_CATEGORY_WARP, 1},
    {"shflXor", NULL, NULL, "__shfl_xor_sync", DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 4, {0}, {0}, {0}, NULL, FUNC_CATEGORY_WARP, 1},

    {"shuffle", NULL, NULL, "__shfl", DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 3, {0}, {0}, {0}, NULL, FUNC_CATEGORY_SYSTEM, 1},
    {"laneId", NULL, NULL, "__laneid", DTYPE_INT32, DTYPE_INT32, DTYPE_INT32, 0, {0}, {0}, {0}, NULL, FUNC_CATEGORY_SYSTEM, 1},
    {"warpid", NULL, NULL, "__warpid", DTYPE_INT32, DTYPE_INT32, DTYPE_INT32, 0, {0}, {0}, {0}, NULL, FUNC_CATEGORY_SYSTEM, 1},
    {"clock", NULL, NULL, "__clock", DTYPE_INT32, DTYPE_INT32, DTYPE_INT32, 0, {0}, {0}, {0}, NULL, FUNC_CATEGORY_SYSTEM, 1},
    {"clock64", NULL, NULL, "__clock64", DTYPE_INT64, DTYPE_INT64, DTYPE_INT64, 0, {0}, {0}, {0}, NULL, FUNC_CATEGORY_SYSTEM, 1},

    {NULL, NULL, NULL, NULL, DTYPE_UNKNOWN, DTYPE_UNKNOWN, DTYPE_UNKNOWN, 0, {0}, {0}, {0}, NULL, FUNC_CATEGORY_OTHER, 0}};

const cuda_function_info_t *find_cuda_function_by_category(
    const char *php_name, cuda_func_category_t category)
{
    for (int i = 0; cuda_functions[i].php_name != NULL; i++)
    {
        if (strcmp(cuda_functions[i].php_name, php_name) == 0 &&
            cuda_functions[i].category == category)
        {
            return &cuda_functions[i];
        }
    }
    return NULL;
}

dtype_t determine_dominant_type(dtype_t arg_types[], uint32_t num_args)
{
    dtype_t dominant_type = DTYPE_UNKNOWN;

    for (uint32_t i = 0; i < num_args; i++)
    {
        if (arg_types[i] == DTYPE_FLOAT64)
        {
            dominant_type = DTYPE_FLOAT64;
            break;
        }
        else if (arg_types[i] == DTYPE_FLOAT32)
        {
            if (dominant_type != DTYPE_FLOAT64)
            {
                dominant_type = DTYPE_FLOAT32;
            }
        }
        else if (arg_types[i] == DTYPE_INT64)
        {
            if (dominant_type == DTYPE_UNKNOWN || dominant_type == DTYPE_INT32)
            {
                dominant_type = DTYPE_INT64;
            }
        }
        else if (arg_types[i] == DTYPE_INT32)
        {
            if (dominant_type == DTYPE_UNKNOWN)
            {
                dominant_type = DTYPE_INT32;
            }
        }
        else if (arg_types[i] == DTYPE_UINT64)
        {
            if (dominant_type == DTYPE_UNKNOWN || dominant_type == DTYPE_UINT32)
            {
                dominant_type = DTYPE_UINT64;
            }
        }
        else if (arg_types[i] == DTYPE_UINT32)
        {
            if (dominant_type == DTYPE_UNKNOWN)
            {
                dominant_type = DTYPE_UINT32;
            }
        }
    }

    return dominant_type;
}

cuda_function_match_t find_cuda_function_by_type(
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

        dtype_t dominant_type = determine_dominant_type(arg_types, num_args);

        if (dominant_type == DTYPE_FLOAT64 && func->cuda_name_f64 != NULL)
        {
            result.cuda_name = func->cuda_name_f64;
            result.return_type = func->return_type_f64;
            break;
        }
        else if (dominant_type == DTYPE_FLOAT32 && func->cuda_name_f32 != NULL)
        {
            result.cuda_name = func->cuda_name_f32;
            result.return_type = func->return_type_f32;
            break;
        }
        else if ((dominant_type == DTYPE_INT32 || dominant_type == DTYPE_INT64) &&
                 func->cuda_name_i32 != NULL)
        {
            result.cuda_name = func->cuda_name_i32;
            result.return_type = func->return_type_i32;
            break;
        }
    }

    return result;
}

const cuda_function_info_t *find_cuda_function(const char *php_name)
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

func_parameter *find_kernel_parameter(func_parameter_list_t *list, const char *name)
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

zend_bool types_are_compatible(dtype_t t1, dtype_t t1_second,
                               dtype_t t2, dtype_t t2_second)
{
    if (t1 == t2 && t1_second == t2_second)
    {
        return 1;
    }

    if (t1 == DTYPE_FLOAT64 && (t2 == DTYPE_FLOAT32 || t2 == DTYPE_INT32 || t2 == DTYPE_INT64))
    {
        return 1;
    }

    if (t1 == DTYPE_FLOAT32 && (t2 == DTYPE_INT32 || t2 == DTYPE_INT64))
    {
        return 1;
    }

    if (t1 == DTYPE_INT64 && t2 == DTYPE_INT32)
    {
        return 1;
    }

    if (t1 == DTYPE_UINT64 && t2 == DTYPE_UINT32)
    {
        return 1;
    }

    if (t1 == DTYPE_LIST && t2 == DTYPE_LIST)
    {
        return (t1_second == t2_second);
    }

    return 0;
}

const char *get_ast_kind_name(zend_ast_kind kind)
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

const char *get_binary_op_symbol(uint32_t op_type)
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

const char *get_assign_op_symbol(uint32_t op_type)
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

const char *get_cuda_object_name(int obj_type)
{
    switch (obj_type)
    {
    case CUDA_OBJ_CUDA:
        return "cuda";
    case CUDA_OBJ_MATH:
        return "math";
    case CUDA_OBJ_ATOMIC:
        return "atomic";
    case CUDA_OBJ_SYNC:
        return "sync";
    case CUDA_OBJ_WARP:
        return "warp";
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

const char *get_unary_op_symbol(uint32_t op_type)
{
    switch (op_type)
    {
    case ZEND_BOOL_NOT:
        return "!";
    case ZEND_BW_NOT:
        return "~";
    default:
        return NULL;
    }
}

const char *get_cuda_type_str(dtype_t type, dtype_t second_dtype)
{
    if (type == DTYPE_LIST)
    {
        switch (second_dtype)
        {
        case DTYPE_FLOAT32:
            return "float*";
        case DTYPE_FLOAT64:
            return "double*";
        case DTYPE_INT32:
            return "int*";
        case DTYPE_INT64:
            return "long long*";
        case DTYPE_UINT32:
            return "unsigned int*";
        case DTYPE_UINT64:
            return "unsigned long long*";
        case DTYPE_BOOL:
            return "bool*";
        default:
            return "void*";
        }
    }

    switch (type)
    {
    case DTYPE_VOID:
        return "void";
    case DTYPE_FLOAT32:
        return "float";
    case DTYPE_FLOAT64:
        return "double";
    case DTYPE_INT32:
        return "int";
    case DTYPE_INT64:
        return "long long";
    case DTYPE_UINT32:
        return "unsigned int";
    case DTYPE_UINT64:
        return "unsigned long long";
    case DTYPE_BOOL:
        return "bool";
    case DTYPE_LIST:
        return "void*";
    default:
        return "void";
    }
}