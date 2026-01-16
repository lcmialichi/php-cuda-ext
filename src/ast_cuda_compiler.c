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
#include "zend_exceptions.h"
#include "ast_cuda_builtins.h"

static void add_cuda_header(cuda_compilation_context_t *context, const char *header);
static void cuda_compiler_error_ex(cuda_compilation_context_t *context, const char *format, ...);
static int generate_function_signature(cuda_compilation_context_t *context);
static int handle_not_allowed(cuda_compilation_context_t *context, zend_ast *ast);
static int handle_ast_stmt_list(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_if(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_for(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_while(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_do_while(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_if_elem(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_zval(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_var(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_return(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_binary_op(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_unary_op(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_unary_minus_op(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_unary_plus_op(cuda_compilation_context_t *context, zend_ast *ast);
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
static int handler_ast_foreach(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_try(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_match(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_nullsafe_prop(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_array(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_yield(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_static_var(cuda_compilation_context_t *context, zend_ast *ast);
static int handler_ast_global(cuda_compilation_context_t *context, zend_ast *ast);
static int handle_declare_shared(cuda_compilation_context_t *context, zend_ast *args_ast);
static int handle_declare_shared_extern(cuda_compilation_context_t *context, zend_ast *args_ast);
static int handle_declare_shared_var(cuda_compilation_context_t *context, zend_ast *args_ast);
static int handle_declare_shared_array(cuda_compilation_context_t *context, zend_ast *args_ast);
static int handle_cuda_declare_shared(cuda_compilation_context_t *context, zend_ast *args_ast);

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
    {ZEND_AST_ARRAY, handler_ast_array},
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
    {ZEND_AST_UNARY_PLUS, handler_ast_unary_plus_op},
    {ZEND_AST_UNARY_MINUS, handler_ast_unary_minus_op},
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
    {ZEND_AST_YIELD, handler_ast_yield},
    {ZEND_AST_YIELD_FROM, handle_not_allowed},
    {ZEND_AST_CLASS_NAME, handle_not_allowed},
    {ZEND_AST_GLOBAL, handler_ast_global},
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
    {ZEND_AST_NULLSAFE_PROP, handler_ast_nullsafe_prop},
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
    {ZEND_AST_YIELD, handler_ast_yield},
    {ZEND_AST_COALESCE, handle_not_allowed},
    {ZEND_AST_ASSIGN_COALESCE, handle_not_allowed},
    {ZEND_AST_STATIC, handler_ast_static_var},
    {ZEND_AST_WHILE, handler_ast_while},
    {ZEND_AST_DO_WHILE, handler_ast_do_while},
    {ZEND_AST_IF_ELEM, handler_ast_if_elem},
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
    {ZEND_AST_MATCH, handler_ast_match},
    {ZEND_AST_MATCH_ARM, handle_not_allowed},
    {ZEND_AST_NAMED_ARG, handle_not_allowed},
    {ZEND_AST_METHOD_CALL, handler_ast_method_call},
    {ZEND_AST_NULLSAFE_METHOD_CALL, handle_not_allowed},
    {ZEND_AST_STATIC_CALL, handle_not_allowed},
    {ZEND_AST_CONDITIONAL, handler_ast_conditional},
    {ZEND_AST_TRY, handler_ast_try},
    {ZEND_AST_CATCH, handle_not_allowed},
    {ZEND_AST_PROP_GROUP, handle_not_allowed},
    {ZEND_AST_CONST_ELEM, handle_not_allowed},
    {ZEND_AST_CLASS_CONST_GROUP, handle_not_allowed},
    {ZEND_AST_CONST_ENUM_INIT, handle_not_allowed},
    {ZEND_AST_FOR, handler_ast_for},
    {ZEND_AST_FOREACH, handler_ast_foreach},
    {ZEND_AST_ENUM_CASE, handle_not_allowed},
    {ZEND_AST_PROP_ELEM, handle_not_allowed},
    {ZEND_AST_PARAM, handle_not_allowed},
};

static void cuda_compiler_error_ex(cuda_compilation_context_t *context, const char *format, ...)
{
    va_list args;
    char *message;

    va_start(args, format);
    vspprintf(&message, 0, format, args);
    va_end(args);

    zend_throw_exception_ex(zend_exception_get_default(), 0,
                            "CUDA compilation error: %s", message);

    if (context->current_line > 0)
    {
        zval line_zv;
        ZVAL_LONG(&line_zv, context->current_line);

        zend_object *exception = EG(exception);
        if (exception)
        {
            zend_update_property(zend_exception_get_default(),
                                 (zend_object *)exception,
                                 "cuda_line", sizeof("cuda_line") - 1,
                                 &line_zv);
        }
    }

    efree(message);
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

static void destroy_shared_var(shared_memory_var_t *var)
{
    if (var)
    {
        efree(var->name);
        efree(var);
    }
}

static shared_memory_var_t *create_shared_var(const char *name, dtype_t dtype,
                                              dtype_t element_dtype, int array_size, int is_dynamic)
{
    shared_memory_var_t *var = (shared_memory_var_t *)emalloc(sizeof(shared_memory_var_t));
    var->name = estrdup(name);
    var->dtype = dtype;
    var->element_dtype = element_dtype;
    var->array_size = array_size;
    var->is_dynamic = is_dynamic;
    var->declared_in_code = 0;
    return var;
}

cuda_compilation_context_t *create_cuda_context(
    func_parameter_list_t *parameters,
    cuda_fn_type fn_type,
    zend_string *name,
    HashTable *headers)
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
    context->uses_shared_memory = 0;
    context->uses_static_shared_memory = 0;
    context->shared_memory_declared = 0;
    context->current_line = 0;

    zend_hash_init(&context->local_variables, 8, NULL, destroy_local_variable, 0);
    zend_hash_init(&context->shared_memory_vars, 8, NULL,
                   (dtor_func_t)destroy_shared_var, 0);

    context->cuda_code_buffer = (smart_string *)ecalloc(1, sizeof(smart_string));
    smart_string_alloc(context->cuda_code_buffer, 512, 0);

    return context;
}

void free_cuda_context(cuda_compilation_context_t *context)
{
    if (!context)
        return;

    zend_hash_destroy(&context->local_variables);
    zend_hash_destroy(&context->shared_memory_vars);

    if (context->cuda_code_buffer)
    {
        smart_string_free(context->cuda_code_buffer);
        efree(context->cuda_code_buffer);
    }

    context->headers = NULL;
    efree(context);
}

static void add_cuda_header(cuda_compilation_context_t *context, const char *header)
{
    zend_string *key = zend_string_init(header, strlen(header), 0);
    if (!zend_hash_exists(context->headers, key))
    {
        zend_hash_add_ptr(context->headers, key, (void *)header);
    }

    zend_string_release(key);
}

static void cleanup_loop_variables(cuda_compilation_context_t *context, int loop_level)
{
    zend_string *key;
    zval *val;
    uint32_t num_key;

    ZEND_HASH_FOREACH_KEY_VAL(&context->local_variables, num_key, key, val)
    {
        local_variable_t *var = (local_variable_t *)Z_PTR_P(val);
        if (var->level == loop_level)
        {
            zend_hash_del(&context->local_variables, var->name);
            zend_string_release(var->name);
            efree(var);
        }
    }
    ZEND_HASH_FOREACH_END();
}

static int compile_argument_list(cuda_compilation_context_t *context,
                                 zend_ast *args_ast,
                                 char **arg_strings,
                                 dtype_t *arg_types,
                                 uint32_t max_args)
{
    if (!args_ast || args_ast->kind != ZEND_AST_ARG_LIST)
    {
        return 0;
    }

    zend_ast_list *list = (zend_ast_list *)args_ast;
    if (list->children > max_args)
    {
        cuda_compiler_error_ex(context,
                               "Too many arguments (max %d)", max_args);
        return 0;
    }

    for (uint32_t i = 0; i < list->children; i++)
    {
        smart_string temp_buffer = {0};
        smart_string_alloc(&temp_buffer, 128, 0);

        smart_string *original_buffer = context->cuda_code_buffer;
        context->cuda_code_buffer = &temp_buffer;

        if (!compile_ast_as_valid_cuda(context, list->child[i]))
        {
            smart_string_free(&temp_buffer);
            context->cuda_code_buffer = original_buffer;
            return 0;
        }

        arg_types[i] = context->last_evaluated_first_dtype;
        smart_string_0(&temp_buffer);
        arg_strings[i] = estrdup(temp_buffer.c);

        smart_string_free(&temp_buffer);
        context->cuda_code_buffer = original_buffer;
    }

    return list->children;
}

static int handle_declare_shared(cuda_compilation_context_t *context, zend_ast *args_ast)
{
    if (!args_ast || args_ast->kind != ZEND_AST_ARG_LIST)
    {
        cuda_compiler_error_ex(context,
                               "Declare::shared() requires arguments");
        return 0;
    }

    int num_args = zend_ast_get_num_children(args_ast);

    if (num_args == 2)
    {
        return handle_declare_shared_var(context, args_ast);
    }
    else if (num_args == 3)
    {
        return handle_declare_shared_array(context, args_ast);
    }
    else if (num_args == 4)
    {
        return handle_declare_shared_array(context, args_ast);
    }
    else
    {
        cuda_compiler_error_ex(context,
                               "Declare::shared() requires 2, 3, or 4 arguments");
        return 0;
    }
}

static int handle_declare_shared_var(cuda_compilation_context_t *context, zend_ast *args_ast)
{
    if (!args_ast || args_ast->kind != ZEND_AST_ARG_LIST)
    {
        cuda_compiler_error_ex(context,
                               "Declare::sharedVar() requires arguments");
        return 0;
    }

    int num_args = zend_ast_get_num_children(args_ast);

    if (num_args < 2)
    {
        cuda_compiler_error_ex(context,
                               "Declare::sharedVar() requires at least 2 arguments");
        return 0;
    }

    zend_ast *ref_ast = args_ast->child[0];
    if (!ref_ast || ref_ast->kind != ZEND_AST_REF)
    {
        cuda_compiler_error_ex(context,
                               "First argument must be a reference: &$var");
        return 0;
    }

    zend_ast *var_ast = ref_ast->child[0];
    if (!var_ast || var_ast->kind != ZEND_AST_VAR)
    {
        cuda_compiler_error_ex(context, "Invalid variable reference");
        return 0;
    }

    zend_ast *var_name_ast = var_ast->child[0];
    if (!var_name_ast || var_name_ast->kind != ZEND_AST_ZVAL)
    {
        cuda_compiler_error_ex(context, "Invalid variable name");
        return 0;
    }

    zval *var_zv = zend_ast_get_zval(var_name_ast);
    if (!var_zv || Z_TYPE_P(var_zv) != IS_STRING)
    {
        cuda_compiler_error_ex(context, "Variable name must be a string");
        return 0;
    }

    const char *var_name = Z_STRVAL_P(var_zv);

    zend_ast *type_ast = args_ast->child[1];
    if (!type_ast || type_ast->kind != ZEND_AST_ZVAL)
    {
        cuda_compiler_error_ex(context, "Type argument must be a string");
        return 0;
    }

    zval *type_zv = zend_ast_get_zval(type_ast);
    if (!type_zv || Z_TYPE_P(type_zv) != IS_STRING)
    {
        cuda_compiler_error_ex(context, "Type argument must be a string");
        return 0;
    }

    const char *type_str = Z_STRVAL_P(type_zv);
    dtype_t element_type = string_to_dtype(type_str);

    if (element_type == DTYPE_UNKNOWN)
    {
        cuda_compiler_error_ex(context, "Unsupported type '%s'", type_str);
        return 0;
    }

    shared_memory_var_t *shared_var = create_shared_var(
        var_name, element_type, element_type, 0, 0);

    zend_string *var_key = zend_string_init(var_name, strlen(var_name), 0);
    zend_hash_add_ptr(&context->shared_memory_vars, var_key, shared_var);
    zend_string_release(var_key);

    const char *c_type = get_cuda_type_str(element_type, DTYPE_UNKNOWN);

    smart_string_appends(context->cuda_code_buffer, "__shared__ ");
    smart_string_appends(context->cuda_code_buffer, c_type);
    smart_string_appendc(context->cuda_code_buffer, ' ');
    smart_string_appends(context->cuda_code_buffer, var_name);
    smart_string_appends(context->cuda_code_buffer, ";\n");

    context->last_evaluated_first_dtype = element_type;
    context->last_evaluated_second_dtype = DTYPE_UNKNOWN;

    return 1;
}
static int handle_declare_shared_array(cuda_compilation_context_t *context, zend_ast *args_ast)
{
    if (!args_ast || args_ast->kind != ZEND_AST_ARG_LIST)
    {
        cuda_compiler_error_ex(context,
                               "Declare::sharedArray() requires arguments");
        return 0;
    }

    zend_ast_list *list = (zend_ast_list *)args_ast;

    if (list->children < 3)
    {
        cuda_compiler_error_ex(context,
                               "Declare::sharedArray() requires at least 3 arguments");
        return 0;
    }

    zend_ast *var_ast = list->child[0];
    if (!var_ast || var_ast->kind != ZEND_AST_VAR)
    {
        cuda_compiler_error_ex(context,
                               "First argument must be a variable: $var");
        return 0;
    }

    zend_ast *var_name_ast = var_ast->child[0];
    if (!var_name_ast || var_name_ast->kind != ZEND_AST_ZVAL)
    {
        cuda_compiler_error_ex(context, "Invalid variable name");
        return 0;
    }

    zval *var_zv = zend_ast_get_zval(var_name_ast);
    if (!var_zv || Z_TYPE_P(var_zv) != IS_STRING)
    {
        cuda_compiler_error_ex(context, "Variable name must be a string");
        return 0;
    }

    const char *var_name = Z_STRVAL_P(var_zv);

    zend_ast *type_ast = list->child[1];
    if (!type_ast || type_ast->kind != ZEND_AST_ZVAL)
    {
        cuda_compiler_error_ex(context, "Type argument must be a string");
        return 0;
    }

    zval *type_zv = zend_ast_get_zval(type_ast);
    if (!type_zv || Z_TYPE_P(type_zv) != IS_STRING)
    {
        cuda_compiler_error_ex(context, "Type argument must be a string");
        return 0;
    }

    const char *type_str = Z_STRVAL_P(type_zv);
    dtype_t element_type = string_to_dtype(type_str);

    if (element_type == DTYPE_UNKNOWN)
    {
        cuda_compiler_error_ex(context, "Unsupported type '%s'", type_str);
        return 0;
    }

    zend_ast *size_ast = list->child[2];

    shared_memory_var_t *shared_var = create_shared_var(
        var_name, DTYPE_LIST, element_type, 1, 0);

    zend_string *var_key = zend_string_init(var_name, strlen(var_name), 0);
    zend_hash_add_ptr(&context->shared_memory_vars, var_key, shared_var);
    zend_string_release(var_key);

    const char *c_type = get_cuda_type_str(element_type, DTYPE_UNKNOWN);

    smart_string_appends(context->cuda_code_buffer, "__shared__ ");
    smart_string_appends(context->cuda_code_buffer, c_type);
    smart_string_appendc(context->cuda_code_buffer, ' ');
    smart_string_appends(context->cuda_code_buffer, var_name);
    smart_string_appendc(context->cuda_code_buffer, '[');

    if (!compile_ast_as_valid_cuda(context, size_ast))
    {
        return 0;
    }

    smart_string_appendc(context->cuda_code_buffer, ']');
    smart_string_appends(context->cuda_code_buffer, ";\n");

    context->last_evaluated_first_dtype = DTYPE_LIST;
    context->last_evaluated_second_dtype = element_type;

    return 1;
}

static int handle_declare_shared_extern(cuda_compilation_context_t *context, zend_ast *args_ast)
{
    if (!args_ast || args_ast->kind != ZEND_AST_ARG_LIST)
    {
        cuda_compiler_error_ex(context,
                               "Declare::sharedExtern() requires arguments");
        return 0;
    }

    int num_args = zend_ast_get_num_children(args_ast);

    if (num_args < 2)
    {
        cuda_compiler_error_ex(context,
                               "Declare::sharedExtern() requires at least 2 arguments");
        return 0;
    }

    zend_ast *ref_ast = args_ast->child[0];
    if (!ref_ast || ref_ast->kind != ZEND_AST_REF)
    {
        cuda_compiler_error_ex(context,
                               "First argument must be a reference: &$var");
        return 0;
    }

    zend_ast *var_ast = ref_ast->child[0];
    if (!var_ast || var_ast->kind != ZEND_AST_VAR)
    {
        cuda_compiler_error_ex(context, "Invalid variable reference");
        return 0;
    }

    zend_ast *var_name_ast = var_ast->child[0];
    if (!var_name_ast || var_name_ast->kind != ZEND_AST_ZVAL)
    {
        cuda_compiler_error_ex(context, "Invalid variable name");
        return 0;
    }

    zval *var_zv = zend_ast_get_zval(var_name_ast);
    if (!var_zv || Z_TYPE_P(var_zv) != IS_STRING)
    {
        cuda_compiler_error_ex(context, "Variable name must be a string");
        return 0;
    }

    const char *var_name = Z_STRVAL_P(var_zv);

    zend_ast *type_ast = args_ast->child[1];
    if (!type_ast || type_ast->kind != ZEND_AST_ZVAL)
    {
        cuda_compiler_error_ex(context, "Type argument must be a string");
        return 0;
    }

    zval *type_zv = zend_ast_get_zval(type_ast);
    if (!type_zv || Z_TYPE_P(type_zv) != IS_STRING)
    {
        cuda_compiler_error_ex(context, "Type argument must be a string");
        return 0;
    }

    const char *type_str = Z_STRVAL_P(type_zv);
    dtype_t element_type = string_to_dtype(type_str);

    if (element_type == DTYPE_UNKNOWN)
    {
        cuda_compiler_error_ex(context, "Unsupported type '%s'", type_str);
        return 0;
    }

    context->uses_shared_memory = 1;
    context->shared_memory_declared = 1;

    shared_memory_var_t *shared_var = create_shared_var(
        var_name, DTYPE_LIST, element_type, -1, 1);

    zend_string *var_key = zend_string_init(var_name, strlen(var_name), 0);
    zend_hash_add_ptr(&context->shared_memory_vars, var_key, shared_var);
    zend_string_release(var_key);

    const char *c_type = get_cuda_type_str(element_type, DTYPE_UNKNOWN);

    smart_string_appends(context->cuda_code_buffer, "extern __shared__ ");
    smart_string_appends(context->cuda_code_buffer, c_type);
    smart_string_appendc(context->cuda_code_buffer, ' ');
    smart_string_appends(context->cuda_code_buffer, var_name);
    smart_string_appends(context->cuda_code_buffer, "[];\n");

    context->last_evaluated_first_dtype = DTYPE_LIST;
    context->last_evaluated_second_dtype = element_type;

    return 1;
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
                cuda_compiler_error_ex(context,
                                       "Duplicate parameter name: '%s'", param_i->name);
                return 0;
            }
        }
    }

    return 1;
}

static int handle_warp_functions(cuda_compilation_context_t *context,
                                 const char *method_name,
                                 zend_ast *args_ast)
{
    const cuda_function_info_t *func = find_cuda_function_by_category(method_name, FUNC_CATEGORY_WARP);
    if (!func)
    {
        return 0;
    }

    char *arg_strings[4] = {NULL};
    dtype_t arg_types[4] = {DTYPE_UNKNOWN};
    uint32_t num_args = compile_argument_list(context, args_ast, arg_strings, arg_types, 4);

    if (num_args != func->num_params)
    {
        for (uint32_t i = 0; i < num_args; i++)
        {
            if (arg_strings[i])
                efree(arg_strings[i]);
        }
        cuda_compiler_error_ex(context,
                               "Function %s expects %d arguments, got %d",
                               method_name, func->num_params, num_args);
        return 0;
    }

    smart_string_appends(context->cuda_code_buffer, func->cuda_name_i32);
    smart_string_appendc(context->cuda_code_buffer, '(');

    if (num_args > 0)
    {
        smart_string_appends(context->cuda_code_buffer, "0xFFFFFFFF");
        if (num_args > 1)
        {
            smart_string_appends(context->cuda_code_buffer, ", ");
        }
    }

    for (uint32_t i = 0; i < num_args; i++)
    {
        if (arg_strings[i])
        {
            if (i > 0)
            {
                if (i > 1)
                    smart_string_appends(context->cuda_code_buffer, ", ");
                smart_string_appends(context->cuda_code_buffer, arg_strings[i]);
            }
            efree(arg_strings[i]);
        }
    }

    smart_string_appendc(context->cuda_code_buffer, ')');
    context->last_evaluated_first_dtype = func->return_type_i32;
    return 1;
}

static int handle_cuda_dump_with_format(cuda_compilation_context_t *context,
                                        zend_ast_list *args_list)
{
    zend_ast *format_ast = args_list->child[0];
    zval *format_zv = zend_ast_get_zval(format_ast);
    const char *format_str = Z_STRVAL_P(format_zv);

    smart_string_appends(context->cuda_code_buffer, "printf(\"");

    for (size_t i = 0; i < Z_STRLEN_P(format_zv); i++)
    {
        unsigned char c = (unsigned char)format_str[i];

        switch (c)
        {
        case '\n':
            smart_string_appends(context->cuda_code_buffer, "\\n");
            break;
        case '\r':
            smart_string_appends(context->cuda_code_buffer, "\\r");
            break;
        case '\t':
            smart_string_appends(context->cuda_code_buffer, "\\t");
            break;
        case '\\':
            smart_string_appends(context->cuda_code_buffer, "\\\\");
            break;
        case '"':
            smart_string_appends(context->cuda_code_buffer, "\\\"");
            break;
        default:
            if (c >= 32 && c < 127)
            {
                smart_string_appendc(context->cuda_code_buffer, c);
            }
            else if (c == 0)
            {
                break;
            }
            else
            {
                char buf[5];
                snprintf(buf, sizeof(buf), "\\x%02x", c);
                smart_string_appends(context->cuda_code_buffer, buf);
            }
            break;
        }
    }

    smart_string_appends(context->cuda_code_buffer, "\"");

    for (uint32_t i = 1; i < args_list->children; i++)
    {
        smart_string_appends(context->cuda_code_buffer, ", ");

        if (!compile_ast_as_valid_cuda(context, args_list->child[i]))
        {
            return 0;
        }

        if (context->last_evaluated_first_dtype == DTYPE_BOOL)
        {
            smart_string temp_buffer = {0};
            smart_string_alloc(&temp_buffer, 256, 0);

            smart_string *original = context->cuda_code_buffer;
            context->cuda_code_buffer = &temp_buffer;

            if (!compile_ast_as_valid_cuda(context, args_list->child[i]))
            {
                smart_string_free(&temp_buffer);
                context->cuda_code_buffer = original;
                return 0;
            }

            context->cuda_code_buffer = original;

            smart_string_appends(context->cuda_code_buffer, "(");
            smart_string_append(context->cuda_code_buffer, &temp_buffer);
            smart_string_appends(context->cuda_code_buffer, " ? \"true\" : \"false\")");

            smart_string_free(&temp_buffer);
        }
    }

    smart_string_appendc(context->cuda_code_buffer, ')');

    context->last_evaluated_first_dtype = DTYPE_VOID;
    context->last_evaluated_second_dtype = DTYPE_UNKNOWN;

    return 1;
}

static int handle_cuda_dump_simple(cuda_compilation_context_t *context,
                                   zend_ast_list *args_list)
{
    smart_string_appends(context->cuda_code_buffer, "printf(\"");

    for (uint32_t i = 0; i < args_list->children; i++)
    {
        if (i > 0)
        {
            smart_string_appends(context->cuda_code_buffer, " ");
        }

        smart_string temp_buffer = {0};
        smart_string_alloc(&temp_buffer, 256, 0);

        smart_string *original = context->cuda_code_buffer;
        context->cuda_code_buffer = &temp_buffer;

        if (!compile_ast_as_valid_cuda(context, args_list->child[i]))
        {
            smart_string_free(&temp_buffer);
            context->cuda_code_buffer = original;
            return 0;
        }

        dtype_t arg_type = context->last_evaluated_first_dtype;

        context->cuda_code_buffer = original;

        if (arg_type == DTYPE_INT32)
        {
            smart_string_appends(context->cuda_code_buffer, "%d");
        }
        else if (arg_type == DTYPE_INT64)
        {
            smart_string_appends(context->cuda_code_buffer, "%lld");
        }
        else if (arg_type == DTYPE_FLOAT32 || arg_type == DTYPE_FLOAT64)
        {
            smart_string_appends(context->cuda_code_buffer, "%f");
        }
        else if (arg_type == DTYPE_BOOL)
        {
            smart_string_appends(context->cuda_code_buffer, "%s");
        }
        else if (arg_type == DTYPE_LIST)
        {
            smart_string_appends(context->cuda_code_buffer, "%p");
        }
        else
        {
            smart_string_appends(context->cuda_code_buffer, "[unknown]");
        }
    }

    smart_string_appends(context->cuda_code_buffer, "\\n\"");
    for (uint32_t i = 0; i < args_list->children; i++)
    {
        smart_string_appends(context->cuda_code_buffer, ", ");

        smart_string temp_buffer = {0};
        smart_string_alloc(&temp_buffer, 256, 0);

        smart_string *original = context->cuda_code_buffer;
        context->cuda_code_buffer = &temp_buffer;

        if (!compile_ast_as_valid_cuda(context, args_list->child[i]))
        {
            smart_string_free(&temp_buffer);
            context->cuda_code_buffer = original;
            return 0;
        }

        dtype_t arg_type = context->last_evaluated_first_dtype;

        context->cuda_code_buffer = original;

        if (arg_type == DTYPE_BOOL)
        {
            smart_string_appends(context->cuda_code_buffer, "(");
            smart_string_append(context->cuda_code_buffer, &temp_buffer);
            smart_string_appends(context->cuda_code_buffer, " ? \"true\" : \"false\")");
        }
        else if (arg_type == DTYPE_LIST)
        {
            smart_string_appends(context->cuda_code_buffer, "(void*)");
            smart_string_append(context->cuda_code_buffer, &temp_buffer);
        }
        else
        {
            smart_string_append(context->cuda_code_buffer, &temp_buffer);
        }

        smart_string_free(&temp_buffer);
    }

    smart_string_appendc(context->cuda_code_buffer, ')');

    context->last_evaluated_first_dtype = DTYPE_VOID;
    context->last_evaluated_second_dtype = DTYPE_UNKNOWN;

    return 1;
}

static int handle_cuda_dump(cuda_compilation_context_t *context,
                            zend_ast *args_ast)
{
    if (!args_ast || args_ast->kind != ZEND_AST_ARG_LIST)
    {
        cuda_compiler_error_ex(context,
                               "$cuda->dump() requires at least one argument");
        return 0;
    }

    zend_ast_list *list = (zend_ast_list *)args_ast;

    if (list->children == 0)
    {
        smart_string_appends(context->cuda_code_buffer,
                             "printf(\"\\n\")");
        context->last_evaluated_first_dtype = DTYPE_VOID;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
        return 1;
    }

    zend_ast *first_arg = list->child[0];

    if (first_arg->kind == ZEND_AST_ZVAL)
    {
        zval *zv = zend_ast_get_zval(first_arg);
        if (zv && Z_TYPE_P(zv) == IS_STRING)
        {
            return handle_cuda_dump_with_format(context, list);
        }
    }

    return handle_cuda_dump_simple(context, list);
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
        context->last_evaluated_first_dtype = DTYPE_LIST;
        context->last_evaluated_second_dtype = DTYPE_INT32;

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
        context->last_evaluated_first_dtype = DTYPE_LIST;
        context->last_evaluated_second_dtype = DTYPE_INT32;

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
        context->last_evaluated_first_dtype = DTYPE_LIST;
        context->last_evaluated_second_dtype = DTYPE_INT32;

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
        context->last_evaluated_first_dtype = DTYPE_LIST;
        context->last_evaluated_second_dtype = DTYPE_INT32;
        context->current_cuda_object = CUDA_OBJ_GRIDDIM;
        return 1;
    }
    else if (strcmp(method_name, "globalIdx") == 0)
    {
        if (args_ast && zend_ast_get_num_children(args_ast) > 0)
        {
            php_error_docref(NULL, E_WARNING,
                             "$cuda->globalIdx() doesn't take arguments");
        }

        context->current_cuda_object = CUDA_OBJ_NONE;
        smart_string_appends(context->cuda_code_buffer, "blockIdx.x * blockDim.x + threadIdx.x");
        context->last_evaluated_first_dtype = DTYPE_INT32;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
        return 1;
    }
    else if (strcmp(method_name, "__declare_shared") == 0)
    {
        return handle_cuda_declare_shared(context, args_ast);
    }
    else if (strcmp(method_name, "dump") == 0)
    {
        return handle_cuda_dump(context, args_ast);
    }

    cuda_compiler_error_ex(context,
                           "Method $cuda->%s() is not supported.", method_name);
    return 0;
}

static int handle_cuda_declare_shared(cuda_compilation_context_t *context, zend_ast *args_ast)
{
    if (!args_ast || args_ast->kind != ZEND_AST_ARG_LIST)
    {
        cuda_compiler_error_ex(context,
                               "$cuda->__declare_shared() requires arguments");
        return 0;
    }

    zend_ast_list *list = (zend_ast_list *)args_ast;

    if (list->children < 3)
    {
        cuda_compiler_error_ex(context,
                               "$cuda->__declare_shared() requires at least 3 arguments");
        return 0;
    }

    zend_ast *var_ast = list->child[0];
    if (!var_ast || var_ast->kind != ZEND_AST_VAR)
    {
        cuda_compiler_error_ex(context,
                               "First argument must be a variable: $var");
        return 0;
    }

    zend_ast *var_name_ast = var_ast->child[0];
    if (!var_name_ast || var_name_ast->kind != ZEND_AST_ZVAL)
    {
        cuda_compiler_error_ex(context, "Invalid variable name");
        return 0;
    }

    zval *var_zv = zend_ast_get_zval(var_name_ast);
    if (!var_zv || Z_TYPE_P(var_zv) != IS_STRING)
    {
        cuda_compiler_error_ex(context, "Variable name must be a string");
        return 0;
    }

    const char *var_name = Z_STRVAL_P(var_zv);

    zend_ast *type_ast = list->child[1];
    if (!type_ast || type_ast->kind != ZEND_AST_ZVAL)
    {
        cuda_compiler_error_ex(context, "Type argument must be a string");
        return 0;
    }

    zval *type_zv = zend_ast_get_zval(type_ast);
    if (!type_zv || Z_TYPE_P(type_zv) != IS_STRING)
    {
        cuda_compiler_error_ex(context, "Type argument must be a string");
        return 0;
    }

    const char *type_str = Z_STRVAL_P(type_zv);
    dtype_t element_type = string_to_dtype(type_str);

    if (element_type == DTYPE_UNKNOWN)
    {
        cuda_compiler_error_ex(context, "Unsupported type '%s'", type_str);
        return 0;
    }

    zend_ast *size_ast = list->child[2];

    int is_array = 0;
    int num_dimensions = 0;
    long static_dimensions[3] = {0};

    if (size_ast->kind == ZEND_AST_ARRAY)
    {
        zend_ast_list *dim_list = (zend_ast_list *)size_ast;

        if (dim_list->children == 0)
        {
            cuda_compiler_error_ex(context,
                                   "Dimension array cannot be empty");
            return 0;
        }

        if (dim_list->children > 3)
        {
            cuda_compiler_error_ex(context,
                                   "Shared memory arrays support up to 3 dimensions, got %d",
                                   dim_list->children);
            return 0;
        }

        is_array = 1;
        num_dimensions = dim_list->children;

        for (uint32_t i = 0; i < dim_list->children; i++)
        {
            zend_ast *elem_ast = dim_list->child[i];

            if (elem_ast->kind != ZEND_AST_ARRAY_ELEM)
            {
                cuda_compiler_error_ex(context,
                                       "Invalid array element at position %d", i + 1);
                return 0;
            }

            zend_ast *value_ast = elem_ast->child[0];
            if (!value_ast)
            {
                cuda_compiler_error_ex(context,
                                       "Missing value for dimension %d", i + 1);
                return 0;
            }

            if (value_ast->kind != ZEND_AST_ZVAL)
            {
                cuda_compiler_error_ex(context,
                                       "Dimension %d must be a literal value, not an expression", i + 1);
                return 0;
            }

            zval *dim_zv = zend_ast_get_zval(value_ast);
            if (!dim_zv)
            {
                cuda_compiler_error_ex(context,
                                       "Invalid dimension value at position %d", i + 1);
                return 0;
            }

            if (Z_TYPE_P(dim_zv) != IS_LONG)
            {
                cuda_compiler_error_ex(context,
                                       "Dimension %d must be an integer, got type %d",
                                       i + 1, Z_TYPE_P(dim_zv));
                return 0;
            }

            long dim_value = Z_LVAL_P(dim_zv);
            if (dim_value <= 0)
            {
                cuda_compiler_error_ex(context,
                                       "Dimension %d must be positive, got %ld",
                                       i + 1, dim_value);
                return 0;
            }

            static_dimensions[i] = dim_value;
        }
    }
    else
    {
        is_array = 1;
        num_dimensions = 1;

        if (size_ast->kind == ZEND_AST_ZVAL)
        {
            zval *size_zv = zend_ast_get_zval(size_ast);
            if (size_zv && Z_TYPE_P(size_zv) == IS_LONG)
            {
                long size = Z_LVAL_P(size_zv);
                if (size <= 1)
                {
                    is_array = 0;
                    num_dimensions = 0;
                }
                else
                {
                    static_dimensions[0] = size;
                }
            }
        }
    }

    zend_string *var_name_zend = zend_string_init(var_name, strlen(var_name), 0);
    local_variable_t *new_var = (local_variable_t *)ecalloc(1, sizeof(local_variable_t));
    new_var->name = zend_string_copy(var_name_zend);
    new_var->dtype = is_array ? DTYPE_LIST : element_type;
    new_var->second_dtype = is_array ? element_type : DTYPE_UNKNOWN;
    new_var->level = context->loop_depth;
    new_var->var_type = VAR_LOCAL_SHARED;
    new_var->array_dimensions = num_dimensions;

    zend_hash_add_ptr(&context->local_variables, var_name_zend, new_var);
    zend_string_release(var_name_zend);

    const char *c_type = get_cuda_type_str(element_type, DTYPE_UNKNOWN);

    smart_string_appends(context->cuda_code_buffer, "__shared__ ");
    smart_string_appends(context->cuda_code_buffer, c_type);
    smart_string_appendc(context->cuda_code_buffer, ' ');
    smart_string_appends(context->cuda_code_buffer, var_name);

    if (is_array)
    {
        if (size_ast->kind == ZEND_AST_ARRAY)
        {
            for (uint32_t i = 0; i < num_dimensions; i++)
            {
                smart_string_appendc(context->cuda_code_buffer, '[');
                smart_string_append_long(context->cuda_code_buffer, static_dimensions[i]);
                smart_string_appendc(context->cuda_code_buffer, ']');
            }
        }
        else
        {
            smart_string_appendc(context->cuda_code_buffer, '[');

            if (size_ast->kind == ZEND_AST_ZVAL)
            {
                zval *size_zv = zend_ast_get_zval(size_ast);
                if (size_zv && Z_TYPE_P(size_zv) == IS_LONG)
                {
                    smart_string_append_long(context->cuda_code_buffer, Z_LVAL_P(size_zv));
                }
                else
                {
                    cuda_compiler_error_ex(context, "Invalid size value");
                    return 0;
                }
            }
            else
            {
                if (!compile_ast_as_valid_cuda(context, size_ast))
                {
                    return 0;
                }
            }

            smart_string_appendc(context->cuda_code_buffer, ']');
        }
    }

    context->last_evaluated_first_dtype = DTYPE_UNKNOWN;
    context->last_evaluated_second_dtype = DTYPE_UNKNOWN;

    return 1;
}

static int handle_cuda_method_by_category(cuda_compilation_context_t *context,
                                          cuda_func_category_t category,
                                          const char *method_name,
                                          zend_ast *args_ast)
{
    const cuda_function_info_t *func_info = find_cuda_function_by_category(method_name, category);
    if (!func_info)
    {
        cuda_compiler_error_ex(context,
                               "Method $cuda->%s() is not supported.", method_name);
        return 0;
    }

    if (func_info->header)
    {
        add_cuda_header(context, func_info->header);
    }

    char *arg_strings[4] = {NULL};
    dtype_t arg_types[4] = {DTYPE_UNKNOWN};
    uint32_t num_args = compile_argument_list(context, args_ast, arg_strings, arg_types, 4);

    // if (num_args == 0 && args_ast != NULL)
    // {
    //     return 0;
    // }

    const char *cuda_func_name = NULL;
    dtype_t return_type = DTYPE_UNKNOWN;

    if (category == FUNC_CATEGORY_MATH)
    {
        dtype_t dominant_type = determine_dominant_type(arg_types, num_args);

        if (dominant_type == DTYPE_FLOAT64 && func_info->cuda_name_f64)
        {
            cuda_func_name = func_info->cuda_name_f64;
            return_type = func_info->return_type_f64;
        }
        else if (dominant_type == DTYPE_FLOAT32 && func_info->cuda_name_f32)
        {
            cuda_func_name = func_info->cuda_name_f32;
            return_type = func_info->return_type_f32;
        }
        else if ((dominant_type == DTYPE_INT32 || dominant_type == DTYPE_INT64) &&
                 func_info->cuda_name_i32)
        {
            cuda_func_name = func_info->cuda_name_i32;
            return_type = func_info->return_type_i32;
        }
        else
        {
            cuda_func_name = func_info->cuda_name_f32 ? func_info->cuda_name_f32 : func_info->cuda_name_f64 ? func_info->cuda_name_f64
                                                                                                            : func_info->cuda_name_i32;
            return_type = func_info->return_type_f32;
        }
    }
    else if (category == FUNC_CATEGORY_ATOMIC)
    {
        cuda_func_name = func_info->cuda_name_i32;
        return_type = func_info->return_type_i32;
    }
    else if (category == FUNC_CATEGORY_SYNC)
    {
        cuda_func_name = func_info->cuda_name_i32;
        return_type = func_info->return_type_i32;
    }
    else if (category == FUNC_CATEGORY_WARP)
    {
        return handle_warp_functions(context, method_name, args_ast);
    }
    else
    {
        cuda_func_name = func_info->cuda_name_f32 ? func_info->cuda_name_f32 : func_info->cuda_name_f64 ? func_info->cuda_name_f64
                                                                                                        : func_info->cuda_name_i32;
        return_type = func_info->return_type_f32;
    }

    if (!cuda_func_name)
    {
        for (uint32_t i = 0; i < num_args; i++)
        {
            if (arg_strings[i])
                efree(arg_strings[i]);
        }
        cuda_compiler_error_ex(context,
                               "No CUDA implementation for %s()", method_name);
        return 0;
    }

    smart_string_appends(context->cuda_code_buffer, cuda_func_name);
    smart_string_appendc(context->cuda_code_buffer, '(');

    for (uint32_t i = 0; i < num_args; i++)
    {
        if (arg_strings[i])
        {
            smart_string_appends(context->cuda_code_buffer, arg_strings[i]);
            efree(arg_strings[i]);
        }
        if (i < num_args - 1)
        {
            smart_string_appends(context->cuda_code_buffer, ", ");
        }
    }

    smart_string_appendc(context->cuda_code_buffer, ')');

    context->last_evaluated_first_dtype = return_type;
    context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    context->current_cuda_object = CUDA_OBJ_NONE;

    return 1;
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
        cuda_compiler_error_ex(context, "Invalid CUDA function type");
        return 0;
    }

    if (qualifier && qualifier[0] != '\0')
    {
        smart_string_appends(context->cuda_code_buffer, "extern \"C\" ");
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
                cuda_compiler_error_ex(context,
                                       "Parameter '$cuda' is reserved and will be automatically injected");
                return 0;
            }

            const char *type_str = get_cuda_type_str(first_type, second_type);
            if (!type_str)
            {
                cuda_compiler_error_ex(context,
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

    context->current_line = zend_ast_get_lineno(ast);
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

    cuda_compiler_error_ex(context,
                           "Kernel compilation failed: PHP construct '%s' is not allowed in CUDA kernels.",
                           ast_name);

    return 0;
}

static int handler_ast_method_call(cuda_compilation_context_t *context, zend_ast *ast)
{
    uint32_t num_children = zend_ast_get_num_children(ast);
    if (num_children < 3)
    {
        cuda_compiler_error_ex(context, "Method call missing parts.");
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
                    cuda_compiler_error_ex(context,
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
        cuda_compiler_error_ex(context,
                               "Invalid object for method call.");
        return 0;
    }

    if (method_name_ast->kind != ZEND_AST_ZVAL)
    {
        cuda_compiler_error_ex(context, "Method name must be a literal.");
        return 0;
    }

    zval *method_zv = zend_ast_get_zval(method_name_ast);
    if (!method_zv || Z_TYPE_P(method_zv) != IS_STRING)
    {
        cuda_compiler_error_ex(context, "Method name must be a string.");
        return 0;
    }

    zend_string *method_name = Z_STR_P(method_zv);
    const char *method_name_c = ZSTR_VAL(method_name);

    switch (context->current_cuda_object)
    {
    case CUDA_OBJ_CUDA:
        return handle_cuda_direct_method(context, method_name_c, args_ast);

    case CUDA_OBJ_MATH:
        return handle_cuda_method_by_category(context, FUNC_CATEGORY_MATH, method_name_c, args_ast);

    case CUDA_OBJ_ATOMIC:
        return handle_cuda_method_by_category(context, FUNC_CATEGORY_ATOMIC, method_name_c, args_ast);

    case CUDA_OBJ_SYNC:
        return handle_cuda_method_by_category(context, FUNC_CATEGORY_SYNC, method_name_c, args_ast);

    case CUDA_OBJ_WARP:
        return handle_cuda_method_by_category(context, FUNC_CATEGORY_WARP, method_name_c, args_ast);

    case CUDA_OBJ_THREADIDX:
    case CUDA_OBJ_BLOCKIDX:
    case CUDA_OBJ_BLOCKDIM:
    case CUDA_OBJ_GRIDDIM:
        cuda_compiler_error_ex(context,
                               "Use property access (->x) not method call for CUDA object members.");
        return 0;

    default:
        cuda_compiler_error_ex(context, "Invalid CUDA object access.");
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
        cuda_compiler_error_ex(context, "Complex variable names are not allowed.");
        return 0;
    }

    zend_ast_zval *var_name_node = (zend_ast_zval *)name_node;
    if (Z_TYPE(var_name_node->val) != IS_STRING)
    {
        cuda_compiler_error_ex(context, "Variable name must be a string.");
        return 0;
    }

    zend_string *var_name_zend = Z_STR(var_name_node->val);
    const char *name_c = ZSTR_VAL(var_name_zend);

    if (strcmp(name_c, "cuda") == 0)
    {
        context->current_cuda_object = CUDA_OBJ_CUDA;
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

    cuda_compiler_error_ex(context,
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
                cuda_compiler_error_ex(context,
                                       "Cannot infer type for new variable '$%s'.", name_c);
                return 0;
            }

            local_variable_t *new_var = (local_variable_t *)ecalloc(1, sizeof(local_variable_t));
            new_var->name = zend_string_copy(var_name_zend);
            new_var->dtype = rvalue_type;
            new_var->second_dtype = rvalue_second_type;
            new_var->level = context->loop_depth;
            new_var->var_type = VAR_LOCAL;
            new_var->array_dimensions = 0;

            zend_hash_add_ptr(&context->local_variables, var_name_zend, new_var);

            smart_string_appends(context->cuda_code_buffer, get_cuda_type_str(rvalue_type, rvalue_second_type));
            smart_string_appendc(context->cuda_code_buffer, ' ');
            smart_string_appendl(context->cuda_code_buffer, name_c, ZSTR_LEN(var_name_zend));
        }
        else
        {
            dtype_t lvalue_type = param ? param->dtype : local->dtype;
            dtype_t lvalue_second_type = param ? param->second_dtype : local->second_dtype;

            if (!types_are_compatible(lvalue_type, lvalue_second_type,
                                      rvalue_type, rvalue_second_type))
            {

                smart_string_free(&rvalue_buffer);
                cuda_compiler_error_ex(context,
                                       "Type mismatch for '$%s'. Expected %s, got %s.",
                                       name_c,
                                       get_cuda_type_str(lvalue_type, lvalue_second_type),
                                       get_cuda_type_str(rvalue_type, rvalue_second_type));
                return 0;
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
        cuda_compiler_error_ex(context, "Invalid assignment target.");
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
        cuda_compiler_error_ex(context, "Assignment operator %d not supported.", ast->attr);
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
    int has_else = 0;

    for (uint32_t i = 0; i < list->children; i++)
    {
        if (list->child[i]->kind == ZEND_AST_IF_ELEM)
        {
            zend_ast *if_elem = list->child[i];
            uint32_t elem_children = zend_ast_get_num_children(if_elem);

            if (elem_children >= 1)
            {
                zend_ast *cond = if_elem->child[0];

                if (cond == NULL || (cond->kind == ZEND_AST_ZVAL && zend_ast_get_zval(cond) == NULL))
                {
                    has_else = 1;
                    smart_string_appends(context->cuda_code_buffer, "} else {\n");

                    if (elem_children >= 2 && !compile_ast_as_valid_cuda(context, if_elem->child[1]))
                    {
                        return 0;
                    }
                    smart_string_appends(context->cuda_code_buffer, "}\n");
                }
                else
                {
                    if (i == 0)
                    {
                        smart_string_appends(context->cuda_code_buffer, "if (");
                    }
                    else
                    {
                        smart_string_appends(context->cuda_code_buffer, "} else if (");
                    }

                    if (!compile_ast_as_valid_cuda(context, cond))
                    {
                        return 0;
                    }
                    smart_string_appends(context->cuda_code_buffer, ") {\n");

                    if (elem_children >= 2 && !compile_ast_as_valid_cuda(context, if_elem->child[1]))
                    {
                        return 0;
                    }
                }
            }
        }
    }

    if (!has_else)
    {
        smart_string_appends(context->cuda_code_buffer, "}\n");
    }

    return 1;
}

static int handler_ast_if_elem(cuda_compilation_context_t *context, zend_ast *ast)
{
    uint32_t num_children = zend_ast_get_num_children(ast);

    if (num_children < 2)
    {
        return 0;
    }

    zend_ast *cond = ast->child[0];
    zend_ast *stmt = ast->child[1];
    zend_ast *else_stmt = (num_children > 2) ? ast->child[2] : NULL;

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

            zend_ast_list *if_list = (zend_ast_list *)else_stmt;

            if (if_list->children > 0 && if_list->child[0]->kind == ZEND_AST_IF_ELEM)
            {
                if (!handler_ast_if_elem(context, if_list->child[0]))
                {
                    return 0;
                }
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

    cleanup_loop_variables(context, context->loop_depth);
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

    cleanup_loop_variables(context, context->loop_depth);
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

static zend_string *extract_base_var_name_from_ast(zend_ast *ast)
{
    while (ast)
    {
        if (ast->kind == ZEND_AST_VAR)
        {
            zend_ast *name_ast = ast->child[0];
            if (name_ast && name_ast->kind == ZEND_AST_ZVAL)
            {
                zval *zv = zend_ast_get_zval(name_ast);
                if (zv && Z_TYPE_P(zv) == IS_STRING)
                {
                    zend_string *result = Z_STR_P(zv);
                    return result;
                }
            }

            return NULL;
        }
        else if (ast->kind == ZEND_AST_DIM)
        {
            ast = ast->child[0];
        }
        else
        {
            return NULL;
        }
    }

    return NULL;
}

static int count_dim_access_levels(zend_ast *ast)
{
    int levels = 0;

    while (ast && ast->kind == ZEND_AST_DIM)
    {
        levels++;
        ast = ast->child[0];
    }

    return levels;
}

static int compile_array_access_recursive(cuda_compilation_context_t *context,
                                          zend_ast *ast,
                                          int *total_levels)
{
    if (ast->kind == ZEND_AST_VAR)
    {
        zend_ast *name_ast = ast->child[0];
        if (name_ast && name_ast->kind == ZEND_AST_ZVAL)
        {
            zval *zv = zend_ast_get_zval(name_ast);
            if (zv && Z_TYPE_P(zv) == IS_STRING)
            {
                smart_string_appends(context->cuda_code_buffer, Z_STRVAL_P(zv));
                return 1;
            }
        }
        return 0;
    }
    else if (ast->kind == ZEND_AST_DIM)
    {
        (*total_levels)++;

        if (!compile_array_access_recursive(context, ast->child[0], total_levels))
        {
            return 0;
        }

        smart_string_appendc(context->cuda_code_buffer, '[');

        if (!compile_ast_as_valid_cuda(context, ast->child[1]))
        {
            return 0;
        }

        smart_string_appendc(context->cuda_code_buffer, ']');
        return 1;
    }

    return 0;
}

static int handler_ast_dim(cuda_compilation_context_t *context, zend_ast *ast)
{
    zend_string *base_var_name = extract_base_var_name_from_ast(ast);
    if (base_var_name == NULL)
    {
        cuda_compiler_error_ex(context,
                               "Cannot determine base variable name for array access");
        return 0;
    }

    const char *var_name_cstr = ZSTR_VAL(base_var_name);
    size_t var_name_len = ZSTR_LEN(base_var_name);

    int access_levels = count_dim_access_levels(ast);
    zval *zv = zend_hash_str_find(&context->local_variables,
                                  var_name_cstr, var_name_len);

    local_variable_t *var = NULL;
    func_parameter *param = NULL;

    if (zv != NULL && Z_TYPE_P(zv) == IS_PTR)
    {
        var = (local_variable_t *)Z_PTR_P(zv);
    }
    else
    {
        param = find_kernel_parameter(context->parameters, var_name_cstr);
    }

    if (var == NULL && param == NULL)
    {
        cuda_compiler_error_ex(context,
                               "Undefined variable '%.*s'",
                               (int)var_name_len, var_name_cstr);
        return 0;
    }

    if (var != NULL && var->dtype != DTYPE_LIST)
    {
        const char *type_str = get_cuda_type_str(param->dtype, param->second_dtype);
        cuda_compiler_error_ex(context,
                               "Variable '%.*s': Type mismatch Expected Array got %s",
                               (int)var_name_len, var_name_cstr, type_str);
        return 0;
    }
    else if (param != NULL && param->dtype != DTYPE_LIST)
    {
        const char *type_str = get_cuda_type_str(param->dtype, param->second_dtype);
        cuda_compiler_error_ex(context,
                               "Variable '%.*s': Type mismatch Expected Array got %s",
                               (int)var_name_len, var_name_cstr, type_str);
        return 0;
    }

    if (var != NULL && var->var_type == VAR_LOCAL_SHARED)
    {
        if (access_levels > var->array_dimensions)
        {
            cuda_compiler_error_ex(context,
                                   "Shared array '%.*s' has %d dimension(s) but accessed with %d index(es)",
                                   (int)var_name_len, var_name_cstr,
                                   var->array_dimensions, access_levels);
            return 0;
        }
    }

    int levels_counted = 0;
    if (!compile_array_access_recursive(context, ast, &levels_counted))
    {
        return 0;
    }

    if (var != NULL)
    {
        if (var->var_type == VAR_LOCAL_SHARED)
        {
            if (access_levels < var->array_dimensions)
            {
                context->last_evaluated_first_dtype = DTYPE_LIST;
                context->last_evaluated_second_dtype = var->second_dtype;
            }
            else
            {
                context->last_evaluated_first_dtype = var->second_dtype;
                context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
            }
        }
        else
        {
            context->last_evaluated_first_dtype = DTYPE_UNKNOWN;
            context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
        }
    }
    else if (param != NULL)
    {
        context->last_evaluated_first_dtype = param->second_dtype;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    }

    return 1;
}

static int handler_ast_zval(cuda_compilation_context_t *context, zend_ast *ast)
{
    zval *zv = zend_ast_get_zval(ast);

    if (!zv)
    {
        cuda_compiler_error_ex(context, "Invalid ZVAL AST node.");
        return 0;
    }

    switch (Z_TYPE_P(zv))
    {
    case IS_LONG:
    {
        smart_string_append_long(context->cuda_code_buffer, Z_LVAL_P(zv));
        context->last_evaluated_first_dtype = DTYPE_INT32;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;

        break;
    }
    case IS_DOUBLE:
    {
        char buffer[64];
        double value = Z_DVAL_P(zv);
        if (context->last_evaluated_first_dtype == DTYPE_FLOAT64)
        {
            snprintf(buffer, sizeof(buffer), "%.17g", value);
            context->last_evaluated_first_dtype = DTYPE_FLOAT64;
            context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
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
            context->last_evaluated_first_dtype = DTYPE_FLOAT32;
            context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
        }

        smart_string_appends(context->cuda_code_buffer, buffer);
        break;
    }
    case IS_TRUE:
        smart_string_appends(context->cuda_code_buffer, "true");
        context->last_evaluated_first_dtype = DTYPE_BOOL;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;

        break;
    case IS_FALSE:
        smart_string_appends(context->cuda_code_buffer, "false");
        context->last_evaluated_first_dtype = DTYPE_BOOL;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;

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
            context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
        }
        else
        {
            cuda_compiler_error_ex(context, "String literals too long for CUDA kernel.");
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
        cuda_compiler_error_ex(context, "Literal type %d is not allowed in CUDA kernel.", Z_TYPE_P(zv));
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
        cuda_compiler_error_ex(context, "Property name must be literal.");
        return 0;
    }

    zval *prop_zv = zend_ast_get_zval(prop_ast);
    if (!prop_zv || Z_TYPE_P(prop_zv) != IS_STRING)
    {
        cuda_compiler_error_ex(context, "Invalid property name.");
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
        else if (strcmp(prop_name_c, "sync") == 0)
        {
            context->current_cuda_object = CUDA_OBJ_SYNC;
            return 1;
        }
        else if (strcmp(prop_name_c, "warp") == 0)
        {
            context->current_cuda_object = CUDA_OBJ_WARP;
            return 1;
        }
        else
        {
            cuda_compiler_error_ex(context,
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
            cuda_compiler_error_ex(context,
                                   "Invalid member '%s' for CUDA object.", prop_name_c);
            return 0;
        }

        smart_string_appendc(context->cuda_code_buffer, '.');
        smart_string_appends(context->cuda_code_buffer, prop_name_c);

        context->last_evaluated_first_dtype = DTYPE_INT32;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;

        context->current_cuda_object = CUDA_OBJ_NONE;

        return 1;
    }
    else if (context->current_cuda_object == CUDA_OBJ_MATH ||
             context->current_cuda_object == CUDA_OBJ_ATOMIC ||
             context->current_cuda_object == CUDA_OBJ_SYNC ||
             context->current_cuda_object == CUDA_OBJ_WARP)
    {

        cuda_compiler_error_ex(context,
                               "Method call expected after %s object.",
                               get_cuda_object_name(context->current_cuda_object));
        return 0;
    }
    else
    {
        cuda_compiler_error_ex(context,
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
        cuda_compiler_error_ex(context, "Binary operator %d not supported in CUDA.", ast->attr);
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

    if (right_type == DTYPE_LIST && left_type != DTYPE_LIST)
    {
        cuda_compiler_error_ex(context,
                               "Type mismatch Expected %s, got Array[%s].",
                               get_cuda_type_str(left_type, DTYPE_UNKNOWN), get_cuda_type_str(right_type, right_second_type));
        return 0;
    }

    if (left_type == DTYPE_LIST && right_type != DTYPE_LIST)
    {
        cuda_compiler_error_ex(context,
                               "Type mismatch Expected Array[%s], got %s.",
                               get_cuda_type_str(right_type, DTYPE_UNKNOWN), get_cuda_type_str(left_type, left_second_type));
        return 0;
    }

    smart_string_appendc(context->cuda_code_buffer, ')');

    if (ast->attr == ZEND_ADD || ast->attr == ZEND_SUB ||
        ast->attr == ZEND_MUL || ast->attr == ZEND_DIV)
    {

        if (left_type == DTYPE_FLOAT64 || right_type == DTYPE_FLOAT64)
        {
            context->last_evaluated_first_dtype = DTYPE_FLOAT64;
        }
        else if (left_type == DTYPE_FLOAT32 || right_type == DTYPE_FLOAT32)
        {
            context->last_evaluated_first_dtype = DTYPE_FLOAT32;
        }
        else if (left_type == DTYPE_INT64 || right_type == DTYPE_INT64)
        {
            context->last_evaluated_first_dtype = DTYPE_INT64;
        }
        else if (left_type == DTYPE_INT32 || right_type == DTYPE_INT32)
        {
            context->last_evaluated_first_dtype = DTYPE_INT32;
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
        context->last_evaluated_first_dtype = DTYPE_BOOL;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    }

    return 1;
}

static int handler_ast_unary_op(cuda_compilation_context_t *context, zend_ast *ast)
{

    if (ast->attr == ZEND_NOP)
    {
        return compile_ast_as_valid_cuda(context, ast->child[0]);
    }

    const char *op_symbol = get_unary_op_symbol(ast->attr);
    if (!op_symbol)
    {
        cuda_compiler_error_ex(context, "Unary operator %d not supported in CUDA.", ast->attr);
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
        context->last_evaluated_first_dtype = DTYPE_BOOL;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    }

    return 1;
}

static int handler_ast_unary_minus_op(cuda_compilation_context_t *context, zend_ast *ast)
{
    smart_string_appends(context->cuda_code_buffer, "(-");
    if (!compile_ast_as_valid_cuda(context, ast->child[0]))
    {
        return 0;
    }

    smart_string_appendc(context->cuda_code_buffer, ')');
    return 1;
}

static int handler_ast_unary_plus_op(cuda_compilation_context_t *context, zend_ast *ast)
{

    smart_string_appends(context->cuda_code_buffer, "(+");
    if (!compile_ast_as_valid_cuda(context, ast->child[0]))
    {
        return 0;
    }

    smart_string_appendc(context->cuda_code_buffer, ')');
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
        cuda_compiler_error_ex(context, "Comparison operator not supported.");
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

    context->last_evaluated_first_dtype = DTYPE_BOOL;
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
        target_type = DTYPE_INT32;
        break;
    case IS_DOUBLE:
        cast_str = "(float)";
        target_type = DTYPE_FLOAT32;
        break;
    case _IS_NUMBER:
        cast_str = "";
        target_type = DTYPE_FLOAT32;
        break;
    default:
        cuda_compiler_error_ex(context, "Cast type %d not supported in CUDA.", cast_type);
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

    if (false_type == DTYPE_LIST && true_type != DTYPE_LIST)
    {
        cuda_compiler_error_ex(context,
                               "Type mismatch Expected %s, got Array[%s].",
                               get_cuda_type_str(false_type, false_second_type), get_cuda_type_str(true_type, DTYPE_UNKNOWN));
        return 0;
    }

    if (true_type == DTYPE_LIST && false_type != DTYPE_LIST)
    {
        cuda_compiler_error_ex(context,
                               "Type mismatch Expected Array[%s], got %s.",
                               get_cuda_type_str(true_type, true_second_type), get_cuda_type_str(false_type, DTYPE_UNKNOWN));
        return 0;
    }

    smart_string_appendc(context->cuda_code_buffer, ')');

    if (true_type == false_type)
    {
        context->last_evaluated_first_dtype = true_type == DTYPE_LIST ? true_second_type : true_type;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    }
    else if (true_type == DTYPE_FLOAT64 || false_type == DTYPE_FLOAT64)
    {
        context->last_evaluated_first_dtype = DTYPE_FLOAT64;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    }
    else if (true_type == DTYPE_FLOAT32 || false_type == DTYPE_FLOAT32)
    {
        context->last_evaluated_first_dtype = DTYPE_FLOAT32;
        context->last_evaluated_second_dtype = DTYPE_UNKNOWN;
    }
    else
    {
        context->last_evaluated_first_dtype = true_type == DTYPE_LIST ? true_second_type : true_type;
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
        cuda_compiler_error_ex(context, "break/continue outside of loop.");
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
        cuda_compiler_error_ex(context,
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

static int handler_ast_foreach(cuda_compilation_context_t *context, zend_ast *ast)
{
    cuda_compiler_error_ex(context,
                           "foreach loops are not supported in CUDA kernels. Use for loops instead.");
    return 0;
}

static int handler_ast_try(cuda_compilation_context_t *context, zend_ast *ast)
{
    cuda_compiler_error_ex(context,
                           "Exception handling (try/catch) is not supported in CUDA kernels.");
    return 0;
}

static int handler_ast_match(cuda_compilation_context_t *context, zend_ast *ast)
{
    cuda_compiler_error_ex(context,
                           "match expressions are not supported in CUDA kernels.");
    return 0;
}

static int handler_ast_nullsafe_prop(cuda_compilation_context_t *context, zend_ast *ast)
{
    cuda_compiler_error_ex(context,
                           "Nullsafe operator (?->) is not supported in CUDA kernels.");
    return 0;
}

static int handler_ast_array(cuda_compilation_context_t *context, zend_ast *ast)
{
    cuda_compiler_error_ex(context,
                           "Array creation is not supported in CUDA kernels. Use parameters or local variables.");
    return 0;
}

static int handler_ast_yield(cuda_compilation_context_t *context, zend_ast *ast)
{
    cuda_compiler_error_ex(context,
                           "Generators (yield) are not supported in CUDA kernels.");
    return 0;
}

static int handler_ast_static_var(cuda_compilation_context_t *context, zend_ast *ast)
{
    cuda_compiler_error_ex(context,
                           "Static variables are not supported in CUDA kernels.");
    return 0;
}

static int handler_ast_global(cuda_compilation_context_t *context, zend_ast *ast)
{
    cuda_compiler_error_ex(context,
                           "Global variables are not supported in CUDA kernels.");
    return 0;
}
