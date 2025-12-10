#include "compiler.h"
#include "compiler_arginfo.h"
#include "php.h"
#include "zend_interfaces.h"
#include "zend_exceptions.h"
#include "zend_closures.h"
#include "kernel_reflection.h"
#include "ast_cuda_compiler.h"
#include "zend_ast.h"
#include "zend_compile.h"

zend_class_entry *cuda_compiler_ce;
static zend_object_handlers compiler_handlers;

static void compiler_free_object(zend_object *object);
static zend_object *compiler_create_object(zend_class_entry *class_type);
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

static char *read_entire_file(const char *filename, size_t *out_len)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *buffer = (char *)emalloc(file_size + 1);
    size_t read_size = fread(buffer, 1, file_size, file);
    buffer[read_size] = '\0';

    fclose(file);

    if (out_len)
    {
        *out_len = read_size;
    }

    return buffer;
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

static char *extract_function_body_for_ast(
    const char *source,
    size_t source_len,
    uint32_t start_line,
    uint32_t end_line,
    size_t *out_len)
{
    if (start_line == 0 || end_line == 0 || start_line > end_line)
    {
        return NULL;
    }

    const char **line_offsets = emalloc(sizeof(char *) * (end_line + 3));
    uint32_t current_line = 1;
    line_offsets[1] = source;

    for (size_t i = 0; i < source_len; i++)
    {
        if (source[i] == '\n')
        {
            current_line++;
            if (current_line > end_line + 1)
                break;
            line_offsets[current_line] = &source[i + 1];
        }
    }

    if (end_line > current_line)
    {
        efree(line_offsets);
        return NULL;
    }

    const char *func_start = line_offsets[start_line];
    const char *func_end =
        (end_line < current_line) ? line_offsets[end_line + 1] : source + source_len;

    const char *body_start = NULL;
    const char *body_end = NULL;
    int brace_level = 0;
    int found_open = 0;

    for (const char *p = func_start; p < func_end; p++)
    {
        if (*p == '{')
        {
            if (!found_open)
            {
                body_start = p + 1;
                found_open = 1;
            }
            brace_level++;
        }
        else if (*p == '}')
        {
            brace_level--;
            if (brace_level == 0 && found_open)
            {
                body_end = p;
                break;
            }
        }
    }

    if (!body_start || !body_end || body_end <= body_start)
    {
        efree(line_offsets);
        return NULL;
    }

    size_t body_len = body_end - body_start;

    const char *prefix = "<?php\n";
    size_t prefix_len = strlen(prefix);

    char *out = emalloc(prefix_len + body_len + 1);

    memcpy(out, prefix, prefix_len);
    memcpy(out + prefix_len, body_start, body_len);

    out[prefix_len + body_len] = '\0';

    if (out_len)
    {
        *out_len = prefix_len + body_len;
    }

    efree(line_offsets);
    return out;
}

ZEND_METHOD(Compiler, __construct)
{
    cuda_compiler_object *compiler;
    zend_string *target_str = NULL;
    zend_long optimization = 3;
    zend_bool debug = 0;
    zend_bool fast_math = 1;

    ZEND_PARSE_PARAMETERS_START(0, 4)
    Z_PARAM_OPTIONAL
    Z_PARAM_STR_OR_NULL(target_str)
    Z_PARAM_LONG(optimization)
    Z_PARAM_BOOL(debug)
    Z_PARAM_BOOL(fast_math)
    ZEND_PARSE_PARAMETERS_END();

    compiler = Z_CUDA_COMPILER_P(ZEND_THIS);

    compiler->kernels = emalloc(sizeof(HashTable));
    zend_hash_init(compiler->kernels, 8, NULL, NULL, 0);

    compiler->devices = emalloc(sizeof(HashTable));
    zend_hash_init(compiler->devices, 8, NULL, NULL, 0);

    if (target_str)
    {
        compiler->target_device = estrndup(ZSTR_VAL(target_str), ZSTR_LEN(target_str));
    }
    else
    {
        compiler->target_device = estrdup("sm_60");
    }

    compiler->optimization_level = optimization;
    compiler->debug_mode = debug;
    compiler->fast_math = fast_math;
    compiler->compilation_context = NULL;
}

ZEND_METHOD(Compiler, kernel)
{
    cuda_compiler_object *compiler;
    zend_fcall_info fci;
    zend_fcall_info_cache fcc;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_FUNC(fci, fcc)
    ZEND_PARSE_PARAMETERS_END();

    compiler = Z_CUDA_COMPILER_P(ZEND_THIS);

    zend_string *name = NULL;
    zend_string *target = NULL;
    zval *grid_zv = NULL;
    zval *block_zv = NULL;

    zend_function *fptr = fcc.function_handler;
    if (!fptr || fptr->type != ZEND_USER_FUNCTION)
    {
        return;
    }

    if (fptr->op_array.fn_flags & ZEND_ACC_USES_THIS)
    {
        zend_throw_exception_ex(NULL, 0,
                                "Kernel functions cannot use object context.");
        return;
    }

    if (fptr->op_array.static_variables != NULL)
    {
        zend_string *device_class_name =
            zend_string_init("Cuda\\Device", strlen("Cuda\\Device"), 0);

        zend_class_entry *device_ce = zend_lookup_class(device_class_name);
        zend_string_release(device_class_name);

        if (!device_ce)
        {
            zend_throw_exception_ex(
                NULL,
                0,
                "Class Cuda\\Device not found");
            return;
        }

        zend_string *key;
        zval *zv;
        ZEND_HASH_FOREACH_STR_KEY_VAL(fptr->op_array.static_variables, key, zv)
        {
            if (Z_TYPE_P(zv) != IS_OBJECT)
            {
                zend_throw_exception_ex(
                    NULL,
                    0,
                    "Kernel closures may only capture instances of Cuda\\Device; "
                    "captured variable '%s' is not an object.",
                    key ? ZSTR_VAL(key) : "<unknown>");
                return;
            }

            if (!instanceof_function(Z_OBJCE_P(zv), device_ce))
            {
                zend_throw_exception_ex(
                    NULL,
                    0,
                    "Kernel closures may only capture instances of Cuda\\Device; "
                    "captured variable '%s' is not a Cuda\\Device.",
                    key ? ZSTR_VAL(key) : "<unknown>");
                return;
            }

            cuda_device_object *dev = Z_CUDA_DEVICE_P(zv);
            zend_string *dev_name = zend_string_copy(dev->name);

            if (!zend_hash_exists(compiler->devices, dev_name))
            {
                zend_hash_add_ptr(compiler->devices, dev_name, dev);
            }
            else
            {
                zend_string_release(dev_name);
            }
        }
        ZEND_HASH_FOREACH_END();
    }

    cuda_method_attribute_args *fargs = cuda_extract_method_attribute(fptr, cuda_attr_kernel_ce);
    if (!fargs)
    {
        return;
    }

    zend_op_array *op_array = &fptr->op_array;
    if (!op_array->filename || op_array->line_start == 0 || op_array->line_end == 0)
    {
        zend_string_release(name);
        zend_string_release(target);
        zend_string_release(fargs->name);
        zend_string_release(fargs->target);
        efree(fargs);
        return;
    }

    size_t file_len = 0;
    char *file_content = read_entire_file(ZSTR_VAL(op_array->filename), &file_len);
    if (!file_content)
    {
        zend_string_release(name);
        zend_string_release(target);
        zend_string_release(fargs->name);
        zend_string_release(fargs->target);
        efree(fargs);
        return;
    }

    size_t src_len = 0;
    char *src = extract_function_body_for_ast(
        file_content, file_len,
        op_array->line_start, op_array->line_end,
        &src_len);

    efree(file_content);

    zend_string *source_code = NULL;
    if (src)
    {
        source_code = zend_string_init(src, src_len, 0);
        efree(src);
    }
    else
    {
        const char *fallback = "<?php\nreturn 0;";
        source_code = zend_string_init(fallback, strlen(fallback), 0);
    }

    zend_arena *ast_arena = NULL;
    zend_ast *ast = zend_compile_string_to_ast(source_code, &ast_arena, fargs->name);

    cuda_kernel_data *kernel = (cuda_kernel_data *)ecalloc(1, sizeof(cuda_kernel_data));
    kernel->name = zend_string_copy(fargs->name);
    kernel->target = zend_string_copy(fargs->target);
    kernel->fci = fci;
    kernel->fcc = fcc;

    kernel->grid[0] = 1;
    kernel->grid[1] = 1;
    kernel->grid[2] = 1;
    kernel->block[0] = 256;
    kernel->block[1] = 1;
    kernel->block[2] = 1;

    if (grid_zv && Z_TYPE_P(grid_zv) == IS_ARRAY)
    {
        zval *x = zend_hash_index_find(Z_ARR_P(grid_zv), 0);
        zval *y = zend_hash_index_find(Z_ARR_P(grid_zv), 1);
        zval *z = zend_hash_index_find(Z_ARR_P(grid_zv), 2);
        if (x)
            kernel->grid[0] = zval_get_long(x);
        if (y)
            kernel->grid[1] = zval_get_long(y);
        if (z)
            kernel->grid[2] = zval_get_long(z);
    }

    if (block_zv && Z_TYPE_P(block_zv) == IS_ARRAY)
    {
        zval *x = zend_hash_index_find(Z_ARR_P(block_zv), 0);
        zval *y = zend_hash_index_find(Z_ARR_P(block_zv), 1);
        zval *z = zend_hash_index_find(Z_ARR_P(block_zv), 2);
        if (x)
            kernel->block[0] = zval_get_long(x);
        if (y)
            kernel->block[1] = zval_get_long(y);
        if (z)
            kernel->block[2] = zval_get_long(z);
    }

    kernel->ast = ast;
    kernel->ast_arena = ast_arena;
    kernel->source_code = source_code;

    kernel->parameters = cuda_extract_parameter_list(fptr, cuda_attr_input_ce, cuda_attr_output_ce);

    kernel->used_devices = emalloc(sizeof(HashTable));
    zend_hash_init(kernel->used_devices, 4, NULL, NULL, 0);

    zend_hash_str_add_ptr(compiler->kernels,
                          ZSTR_VAL(kernel->name),
                          ZSTR_LEN(kernel->name),
                          kernel);

    zend_string_release(fargs->name);
    zend_string_release(fargs->target);
    efree(fargs);

    RETURN_ZVAL(getThis(), 1, 0);
}

ZEND_METHOD(Compiler, device)
{
    cuda_compiler_object *compiler;
    zend_string *name;
    zend_fcall_info fci;
    zend_fcall_info_cache fcc;
    zval *attributes = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_FUNC(fci, fcc)
    ZEND_PARSE_PARAMETERS_END();

    compiler = Z_CUDA_COMPILER_P(ZEND_THIS);

    zend_function *fptr = fcc.function_handler;
    if (!fptr || fptr->type != ZEND_USER_FUNCTION)
    {
        return;
    }

    cuda_method_attribute_args *fargs = cuda_extract_method_attribute(fptr, cuda_attr_device_ce);
    if (!fargs)
    {
        return;
    }

    zend_string *device_class_name = zend_string_init("Cuda\\Device", strlen("Cuda\\Device"), 0);
    zend_class_entry *device_ce = zend_lookup_class(device_class_name);
    zend_string_release(device_class_name);

    if (!device_ce)
    {
        zend_throw_exception_ex(NULL, 0, "Class Cuda\\Device not found");
        RETURN_NULL();
    }

    zval device_zv;
    object_init_ex(&device_zv, device_ce);

    cuda_device_object *device = Z_CUDA_DEVICE_P(&device_zv);

    device->fci = fci;
    device->fcc = fcc;
    device->name = zend_string_copy(fargs->name);
    device->target = zend_string_copy(fargs->target);
    device->ast = NULL;
    device->ast_arena = NULL;

    zend_hash_add_ptr(compiler->devices, zend_string_copy(fargs->name), device);
    zend_string_release(fargs->name);
    zend_string_release(fargs->target);
    efree(fargs);

    RETURN_ZVAL(getThis(), 1, 0);
}

ZEND_METHOD(Compiler, compile)
{
    cuda_compiler_object *compiler;
    char *target = NULL;
    size_t target_len;
    zend_bool optimize = 1;
    zend_bool debug = 0;

    ZEND_PARSE_PARAMETERS_START(0, 3)
    Z_PARAM_OPTIONAL
    Z_PARAM_STRING(target, target_len)
    Z_PARAM_BOOL(optimize)
    Z_PARAM_BOOL(debug)
    ZEND_PARSE_PARAMETERS_END();

    compiler = Z_CUDA_COMPILER_P(ZEND_THIS);

    zend_string *module_class_name = zend_string_init("Cuda\\CompiledModule", strlen("Cuda\\CompiledModule"), 0);
    zend_class_entry *module_ce = zend_lookup_class(module_class_name);

    zval module_zv;
    object_init_ex(&module_zv, module_ce);

    cuda_module_object *module = Z_CUDA_MODULE_P(&module_zv);

    module->ptx_code = estrdup("// Placeholder PTX code\n.version 7.5\n.target sm_60");
    module->ptx_size = strlen(module->ptx_code);

    module->functions = emalloc(sizeof(HashTable));
    zend_hash_init(module->functions, 8, NULL, NULL, 0);

    module->kernel_functions = emalloc(sizeof(HashTable));
    zend_hash_init(module->kernel_functions, 8, NULL, NULL, 0);

    cuda_kernel_object *kernel;
    ZEND_HASH_FOREACH_PTR(compiler->kernels, kernel)
    {
        zval kernel_name;
        ZVAL_STR(&kernel_name, zend_string_copy(kernel->name));
        zend_hash_add(module->kernel_functions, kernel->name, &kernel_name);
    }
    ZEND_HASH_FOREACH_END();
    RETURN_ZVAL(&module_zv, 1, 0);
}

ZEND_METHOD(Compiler, getKernels)
{
    cuda_compiler_object *compiler = Z_CUDA_COMPILER_P(ZEND_THIS);

    array_init(return_value);

    cuda_kernel_data *kernel;
    ZEND_HASH_FOREACH_PTR(compiler->kernels, kernel)
    {
        zval kernel_info;
        array_init(&kernel_info);

        add_assoc_str(&kernel_info, "name", zend_string_copy(kernel->name));
        add_assoc_str(&kernel_info, "target", zend_string_copy(kernel->target));

        zval grid_zv;
        array_init(&grid_zv);
        add_next_index_long(&grid_zv, kernel->grid[0]);
        add_next_index_long(&grid_zv, kernel->grid[1]);
        add_next_index_long(&grid_zv, kernel->grid[2]);
        add_assoc_zval(&kernel_info, "grid", &grid_zv);

        zval block_zv;
        array_init(&block_zv);
        add_next_index_long(&block_zv, kernel->block[0]);
        add_next_index_long(&block_zv, kernel->block[1]);
        add_next_index_long(&block_zv, kernel->block[2]);
        add_assoc_zval(&kernel_info, "block", &block_zv);

        add_assoc_zval(return_value, ZSTR_VAL(kernel->name), &kernel_info);
    }
    ZEND_HASH_FOREACH_END();
}

ZEND_METHOD(Compiler, getDevices)
{
    cuda_compiler_object *compiler = Z_CUDA_COMPILER_P(ZEND_THIS);

    array_init(return_value);

    cuda_device_object *device;
    ZEND_HASH_FOREACH_PTR(compiler->devices, device)
    {
        zval device_zv;
        ZVAL_OBJ(&device_zv, &device->std);
        zend_hash_add(return_value->value.arr, device->name, &device_zv);
        Z_ADDREF(device_zv);
    }
    ZEND_HASH_FOREACH_END();
}

static void compiler_free_object(zend_object *object)
{
    cuda_compiler_object *compiler = Z_CUDA_COMPILER_FROM_OBJ(object);

    if (compiler->target_device)
    {
        efree(compiler->target_device);
    }

    if (compiler->kernels)
    {
        cuda_kernel_object *kernel;
        ZEND_HASH_FOREACH_PTR(compiler->kernels, kernel)
        {
            // @todo destroy?
        }
        ZEND_HASH_FOREACH_END();
        zend_hash_destroy(compiler->kernels);
        efree(compiler->kernels);
    }

    if (compiler->devices)
    {
        cuda_device_object *device;
        ZEND_HASH_FOREACH_PTR(compiler->devices, device)
        {
            // @todo destroy?
        }
        ZEND_HASH_FOREACH_END();
        zend_hash_destroy(compiler->devices);
        efree(compiler->devices);
    }

    if (compiler->compilation_context)
    {
        free_cuda_context(compiler->compilation_context);
    }

    zend_object_std_dtor(&compiler->std);
}

static zend_object *compiler_create_object(zend_class_entry *class_type)
{
    cuda_compiler_object *compiler =
        (cuda_compiler_object *)ecalloc(1, sizeof(cuda_compiler_object));

    zend_object_std_init(&compiler->std, class_type);
    compiler->std.handlers = &compiler_handlers;

    compiler->target_device = NULL;
    compiler->optimization_level = 3;
    compiler->debug_mode = 0;
    compiler->fast_math = 1;
    compiler->kernels = NULL;
    compiler->devices = NULL;
    compiler->compilation_context = NULL;

    return &compiler->std;
}

int compiler_init()
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, COMPILER_CLASS_NAME, compiler_methods);
    cuda_compiler_ce = zend_register_internal_class(&ce);
    cuda_compiler_ce->create_object = compiler_create_object;
    cuda_compiler_ce->ce_flags |= ZEND_ACC_FINAL;

    memcpy(&compiler_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    compiler_handlers.offset = XtOffsetOf(cuda_compiler_object, std);
    compiler_handlers.free_obj = compiler_free_object;

    return 1;
}