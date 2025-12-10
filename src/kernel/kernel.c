#include "kernel.h"
#include "ke_arginfo.h"
#include "php.h"
#include "zend_compile.h"
#include "zend_attributes.h"
#include "kernel_reflection.h"
#include "zend_string.h"
#include "zend_operators.h"
#include "zend_ast.h"
#include "ast_cuda_compiler.h"
#include "ext/standard/php_smart_string.h"
#include "kernel_types.h"

zend_class_entry *kernel_ce;
static zend_object_handlers kernel_handlers;

extern zend_class_entry *cuda_attr_kernel_ce;
extern zend_class_entry *cuda_attr_device_ce;
extern zend_class_entry *cuda_attr_input_ce;
extern zend_class_entry *cuda_attr_output_ce;
extern cuda_method_attribute_args *cuda_extract_method_attribute(zend_function *fptr, zend_class_entry *ce_attribute);

static int indent_level = 0;

static void print_indent()
{
    for (int i = 0; i < indent_level; i++)
    {
        php_printf("  ");
    }
}

static const char *get_ast_kind_name(zend_ast *ast)
{
    if (ast == NULL)
        return "NULL";

    switch (ast->kind)
    {
    case ZEND_AST_ZVAL:
        return "LITERAL";
    case ZEND_AST_CONSTANT:
        return "CONSTANT";

    case ZEND_AST_STMT_LIST:
        return "STMT_LIST";
    case ZEND_AST_IF:
        return "IF";
    case ZEND_AST_IF_ELEM:
        return "IF_ELEM";
    case ZEND_AST_WHILE:
        return "WHILE";
    case ZEND_AST_DO_WHILE:
        return "DO_WHILE";
    case ZEND_AST_FOR:
        return "FOR";
    case ZEND_AST_ARG_LIST:
        return "ARG_LIST";
    case ZEND_AST_EXPR_LIST:
        return "EXPR_LIST";

    case ZEND_AST_VAR:
        return "VAR";
    case ZEND_AST_RETURN:
        return "RETURN";
    case ZEND_AST_CONST:
        return "CONST_FETCH";

    case ZEND_AST_DIM:
        return "ARRAY_ACCESS";
    case ZEND_AST_CALL:
        return "CALL";

    case ZEND_AST_ASSIGN:
        return "ASSIGN";
    case ZEND_AST_ASSIGN_OP:
        return "ASSIGN_OP";
    case ZEND_AST_BINARY_OP:
        return "BINARY_OP";

    case ZEND_AST_GREATER:
        return "OP_GREATER";
    case ZEND_AST_GREATER_EQUAL:
        return "OP_GEQUAL";
    case ZEND_AST_AND:
        return "OP_AND";
    case ZEND_AST_OR:
        return "OP_OR";

    case ZEND_AST_METHOD_CALL:
        return "METHOD_CALL";
    case ZEND_AST_CONDITIONAL:
        return "TERNARY_OP";
    case ZEND_AST_POST_DEC:
        return "POST_DEC";
    case ZEND_AST_POST_INC:
        return "POST_INC";

    default:
        php_printf("UNKNOWN_KIND(%d)", ast->kind);
        return "UNKNOWN";
    }
}

static void print_ast_recursive(zend_ast *ast)
{
    if (ast == NULL)
        return;

    print_indent();

    php_printf("[%s] ", get_ast_kind_name(ast));

    if (ast->kind == ZEND_AST_ZVAL)
    {
        zend_ast_zval *zval_node = (zend_ast_zval *)ast;
        php_printf("Value: ");
        zend_print_zval(&zval_node->val, 0);
        php_printf("\n");
        return;
    }

    if (ast->kind == ZEND_AST_VAR)
    {
        php_printf("Variable:\n");
        indent_level++;
        print_ast_recursive(ast->child[0]);
        indent_level--;
        return;
    }

    if (ast->kind == ZEND_AST_BINARY_OP)
    {
        php_printf("Opcode: %d\n", ast->attr);
    }
    else
    {
        php_printf("\n");
    }

    indent_level++;

    if (zend_ast_is_list(ast))
    {
        zend_ast_list *list = (zend_ast_list *)ast;
        for (uint32_t i = 0; i < list->children; i++)
        {
            print_ast_recursive(list->child[i]);
        }
    }
    else
    {
        uint32_t children = zend_ast_get_num_children(ast);
        for (uint32_t i = 0; i < children; i++)
        {
            print_ast_recursive(ast->child[i]);
        }
    }

    indent_level--;
}
zend_string *cuda_extract_method_source_code(zend_function *fptr)
{
    const char *source =
        "<?php \n"
        "if ($a > 10 || $a == 100) { \n"
        "    return max($a * $b, 0.0); \n"
        "} \n"
        "$newVar = 1;\n"
        "$newVar = 1 * 2;\n"
        "for ($i = 0; $i <= 10; $i = $i+ 1) { \n"
        "    $a[$i] = $a[$i] * $i; \n"
        "} \n"
        "return $a;";

    return zend_string_init(source, strlen(source), 0);
}

static void kernel_free_object(zend_object *object)
{
    kernel_obj *obj = (kernel_obj *)((char *)object - XtOffsetOf(kernel_obj, obj));
    zend_object_std_dtor(&obj->obj);
}

static zend_object *kernel_create_object(zend_class_entry *class_type)
{
    kernel_obj *obj = (kernel_obj *)ecalloc(1, sizeof(kernel_obj));
    zend_object_std_init(&obj->obj, class_type);
    object_properties_init(&obj->obj, class_type);
    obj->obj.handlers = &kernel_handlers;
    return &obj->obj;
}

ZEND_METHOD(Kernel, __construct)
{
    zend_class_entry *ce = Z_OBJCE_P(ZEND_THIS);
    zend_function *fptr;
    zend_string *method_name;

    smart_string cuda_output_buffer = {0};

    php_printf("=== STARTING KERNEL COMPILATION for %s ===\n", ZSTR_VAL(ce->name));

    func_parameter_list_t output_params = {0, NULL};
    int compilation_result = 1;

    ZEND_HASH_FOREACH_STR_KEY_PTR(&ce->function_table, method_name, fptr)
    {

        if (ZSTR_VAL(method_name)[0] == '_')
        {
            continue;
        }

        cuda_method_attribute_args *fargs = cuda_extract_method_attribute(fptr, cuda_attr_device_ce);
        func_parameter_list_t *params = cuda_extract_parameter_list(fptr, cuda_attr_input_ce, cuda_attr_output_ce);

        if (fargs)
        {
            zend_string *method_source_code = NULL;
            zend_arena *ast_arena = NULL;
            zend_ast *root_ast = NULL;

            php_printf("\n-> Processing Kernel: %s (Target: %s)\n", ZSTR_VAL(fptr->common.function_name), ZSTR_VAL(fargs->target));

            method_source_code = cuda_extract_method_source_code(fptr);

            if (method_source_code)
            {

                root_ast = zend_compile_string_to_ast(
                    method_source_code,
                    &ast_arena,
                    fptr->common.function_name);

                if (root_ast)
                {
                    php_printf("    [SUCCESS] AST");
                    init_cuda_headers();

                    cuda_compilation_context_t *context = create_cuda_context(params);
                    php_printf("\n*** START CUDA CODE GENERATION ***\n");
                    print_ast_recursive(root_ast);

                    compilation_result = compile_ast_as_valid_cuda(
                        context,
                        root_ast);

                    if (compilation_result == 1)
                    {
                        smart_string_0(context->cuda_code_buffer);
                        php_printf("\n*** CUDA CODE OUTPUT ***\n");

                        php_printf("%s", context->cuda_code_buffer->c);
                        php_printf("*** END CUDA CODE OUTPUT ***\n");
                    }
                    else
                    {
                        php_printf("\n*** CUDA CODE GENERATION FAILED ***\n");
                    }

                    free_cuda_context(context);
                }
                else
                {
                    php_printf("    [FAILURE] Could not compile source to AST.\n");
                }

                if (ast_arena)
                {
                    zend_arena_destroy(ast_arena);
                }

                zend_string_release(method_source_code);
            }
        }
    }
    ZEND_HASH_FOREACH_END();
    php_printf("=== KERNEL COMPILATION ENDED ===\n");
}

int kernel_init()
{
    zend_class_entry *kernel_ce_local = register_kernel_class();

    kernel_ce_local->create_object = kernel_create_object;

    memcpy(&kernel_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    kernel_handlers.offset = XtOffsetOf(kernel_obj, obj);
    kernel_handlers.free_obj = kernel_free_object;

    return 1;
}