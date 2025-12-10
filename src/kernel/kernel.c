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

typedef enum {
    KERNEL_OK = 0,
    KERNEL_ERR = 1
} kernel_status_t;

typedef struct {
    kernel_status_t status;
    const char *msg; 
} kernel_result_t;

static inline kernel_result_t kernel_ok(const char *msg) { return (kernel_result_t){ KERNEL_OK, msg }; }
static inline kernel_result_t kernel_err(const char *msg) { return (kernel_result_t){ KERNEL_ERR, msg }; }

static void klog(const char *tag, const char *fmt, ...)
{
    va_list args;
    php_printf("%s", tag);

    va_start(args, fmt);
    vspprintf(NULL, 0, fmt, args);
    char *buf;
    vspprintf(&buf, 0, fmt, args);
    va_end(args);

    php_printf("%s", buf);
    efree(buf);
}

static const char *ast_kind_name(zend_ast *ast)
{
    if (!ast) return "NULL";

    switch (ast->kind) {
        case ZEND_AST_ZVAL: return "LITERAL";
        case ZEND_AST_CONSTANT: return "CONSTANT";

        case ZEND_AST_STMT_LIST: return "STMT_LIST";
        case ZEND_AST_IF: return "IF";
        case ZEND_AST_IF_ELEM: return "IF_ELEM";
        case ZEND_AST_WHILE: return "WHILE";
        case ZEND_AST_DO_WHILE: return "DO_WHILE";
        case ZEND_AST_FOR: return "FOR";
        case ZEND_AST_ARG_LIST: return "ARG_LIST";
        case ZEND_AST_EXPR_LIST: return "EXPR_LIST";

        case ZEND_AST_VAR: return "VAR";
        case ZEND_AST_RETURN: return "RETURN";
        case ZEND_AST_CONST: return "CONST_FETCH";

        case ZEND_AST_DIM: return "ARRAY_ACCESS";
        case ZEND_AST_CALL: return "CALL";

        case ZEND_AST_ASSIGN: return "ASSIGN";
        case ZEND_AST_ASSIGN_OP: return "ASSIGN_OP";
        case ZEND_AST_BINARY_OP: return "BINARY_OP";

        case ZEND_AST_GREATER: return "OP_GREATER";
        case ZEND_AST_GREATER_EQUAL: return "OP_GEQUAL";
        case ZEND_AST_AND: return "OP_AND";
        case ZEND_AST_OR: return "OP_OR";

        case ZEND_AST_METHOD_CALL: return "METHOD_CALL";
        case ZEND_AST_CONDITIONAL: return "TERNARY_OP";
        case ZEND_AST_POST_DEC: return "POST_DEC";
        case ZEND_AST_POST_INC: return "POST_INC";
        default: return "UNKNOWN";
    }
}

static void ast_print(zend_ast *ast, uint32_t indent)
{
    if (!ast) return;

    for (uint32_t i = 0; i < indent; ++i) php_printf("  ");
    php_printf("[%s] ", ast_kind_name(ast));

    if (ast->kind == ZEND_AST_ZVAL) {
        zend_ast_zval *zn = (zend_ast_zval *)ast;
        php_printf("Value: ");
        zend_print_zval(&zn->val, 0);
        php_printf("\n");
        return;
    }

    if (ast->kind == ZEND_AST_VAR) {
        php_printf("Variable:\n");
        ast_print(ast->child[0], indent + 1);
        return;
    }

    if (ast->kind == ZEND_AST_BINARY_OP) {
        php_printf("Opcode: %d\n", ast->attr);
    } else {
        php_printf("\n");
    }

    if (zend_ast_is_list(ast)) {
        zend_ast_list *list = (zend_ast_list *)ast;
        for (uint32_t i = 0; i < list->children; ++i) {
            ast_print(list->child[i], indent + 1);
        }
    } else {
        uint32_t children = zend_ast_get_num_children(ast);
        for (uint32_t i = 0; i < children; ++i) {
            ast_print(ast->child[i], indent + 1);
        }
    }
}

typedef struct {
    zend_function *fptr;
    zend_string   *name;
    zend_string   *filename;
    uint32_t       start_line;
    uint32_t       end_line;
} kernel_method_meta_t;

static inline bool kernel_is_user_method(zend_function *fptr)
{
    return (fptr && fptr->type == ZEND_USER_FUNCTION);
}

static inline bool kernel_is_private_like(zend_string *name)
{
    return (name && ZSTR_VAL(name)[0] == '_');
}

static kernel_method_meta_t kernel_build_method_meta(zend_function *fptr)
{
    kernel_method_meta_t m = {0};

    if (!kernel_is_user_method(fptr)) return m;

    zend_op_array *op_array = &fptr->op_array;
    m.fptr        = fptr;
    m.name        = fptr->common.function_name;
    m.filename    = op_array->filename;
    m.start_line  = op_array->line_start;
    m.end_line    = op_array->line_end;
    return m;
}


static kernel_result_t kernel_read_method_body(
    const kernel_method_meta_t *meta,
    zend_string **out_code)
{
    *out_code = NULL;

    if (!meta || !meta->filename || meta->start_line == 0 || meta->end_line == 0) {
        return kernel_err("missing filename/line range");
    }

    FILE *file = fopen(ZSTR_VAL(meta->filename), "r");
    if (!file) {
        return kernel_err("cannot open source file");
    }

    smart_string code = {0};
    smart_string_alloc(&code, 2048, 0);

    smart_string_appends(&code, "<?php\n\n");

    if (meta->fptr->common.num_args > 0) {
        smart_string_appends(&code, "// Parameters:\n");
        for (uint32_t i = 0; i < meta->fptr->common.num_args; ++i) {
            zend_arg_info *arg = &meta->fptr->common.arg_info[i];
            if (!arg->name) continue;
            smart_string_appends(&code, "// $");
            smart_string_appends(&code, ZSTR_VAL(arg->name));
            if (i + 1 < meta->fptr->common.num_args) smart_string_appends(&code, ", ");
        }
        smart_string_appends(&code, "\n\n");
    }

    char line[4096];
    uint32_t current_line = 1;
    bool in_method_body = false;
    int brace_depth = 0;

    while (fgets(line, sizeof(line), file)) {
        if (current_line >= meta->start_line && current_line <= meta->end_line) {

            char *trimmed = line;
            while (*trimmed == ' ' || *trimmed == '\t') ++trimmed;

            if (!in_method_body) {
                for (char *p = trimmed; *p; ++p) {
                    if (*p == '{') {
                        in_method_body = true;
                        brace_depth = 1;

                        char *after_brace = p + 1;
                        while (*after_brace == ' ' || *after_brace == '\t') ++after_brace;
                        if (*after_brace && *after_brace != '\n' && *after_brace != '\r') {
                            smart_string_appends(&code, after_brace);
                        }
                        break;
                    }
                }
            } else {
                for (char *p = line; *p; ++p) {
                    if (*p == '{') {
                        ++brace_depth;
                    } else if (*p == '}') {
                        --brace_depth;
                        if (brace_depth == 0) {
                            *p = '\0'; 
                            smart_string_appends(&code, line);
                            in_method_body = false;
                            break;
                        }
                    }
                }

                if (in_method_body && brace_depth > 0) {
                    smart_string_appends(&code, line);
                }
            }
        }

        if (++current_line > meta->end_line) break;
    }

    fclose(file);

    while (code.len > 0 &&
           (code.c[0] == ' ' || code.c[0] == '\t' || code.c[0] == '\n' || code.c[0] == '\r')) {
        memmove(code.c, code.c + 1, code.len - 1);
        code.len--;
    }

    if (code.len == 0) {
        smart_string_free(&code);
        return kernel_err("extracted body is empty");
    }

    smart_string_0(&code);
    *out_code = zend_string_init(code.c, code.len, 0);

    klog("  ", "[SUCCESS] Extracted %zu bytes\n", code.len);
    return kernel_ok("source extracted");
}

typedef struct {
    cuda_method_attribute_args *attrs;
    func_parameter_list_t      *params;
    zend_string                *source;
    zend_arena                 *ast_arena;
    zend_ast                   *ast;
    cuda_compilation_context_t *ctx;
} kernel_compile_session_t;

static void kernel_session_init(kernel_compile_session_t *s)
{
    memset(s, 0, sizeof(*s));
}

static void kernel_session_cleanup(kernel_compile_session_t *s)
{
    if (s->source) zend_string_release(s->source);
    if (s->ast_arena) zend_arena_destroy(s->ast_arena);

    if (s->attrs) {
        zend_string_release(s->attrs->name);
        zend_string_release(s->attrs->target);
        efree(s->attrs);
    }

    if (s->params) efree(s->params);
    if (s->ctx) free_cuda_context(s->ctx);
}

static kernel_result_t kernel_compile_one_method(zend_function *fptr)
{
    kernel_compile_session_t s;
    kernel_session_init(&s);

    s.attrs = cuda_extract_method_attribute(fptr, cuda_attr_device_ce);
    if (!s.attrs) {
        kernel_session_cleanup(&s);
        return kernel_ok("not a device kernel (no attribute)");
    }

    s.params = cuda_extract_parameter_list(fptr, cuda_attr_input_ce, cuda_attr_output_ce);

    klog("-> ", "Processing Kernel: %s (Target: %s)\n",
         ZSTR_VAL(fptr->common.function_name),
         ZSTR_VAL(s.attrs->target));

    kernel_method_meta_t meta = kernel_build_method_meta(fptr);
    kernel_result_t r = kernel_read_method_body(&meta, &s.source);
    if (r.status != KERNEL_OK) {
        klog("  ", "[ERROR] Could not extract method body (%s). Using fallback mock code.\n", r.msg);

        const char *fallback_source =
            "<?php\n"
            "if ($a > 10 || $a == 100) {\n"
            "    return $this->max($a * $b, 0.0);\n"
            "}\n"
            "$newVar = $this->threadIdx();\n"
            "$newVar = 1 * 2;\n"
            "for ($i = 0; $i <= 10; $i = $i + 1) {\n"
            "    $a[$i] = $a[$i] * $i;\n"
            "}\n"
            "return $a;";
        s.source = zend_string_init(fallback_source, strlen(fallback_source), 0);
    }

    s.ast = zend_compile_string_to_ast(s.source, &s.ast_arena, fptr->common.function_name);
    if (!s.ast) {
        klog("  ", "[FAILURE] Could not compile source to AST\n");
        kernel_session_cleanup(&s);
        return kernel_err("ast compilation failed");
    }

    klog("  ", "[SUCCESS] AST generated\n");
    init_cuda_headers();

    s.ctx = create_cuda_context(s.params);
    klog("\n", "*** START CUDA CODE GENERATION ***\n");

    ast_print(s.ast, 0);

    int ok = compile_ast_as_valid_cuda(s.ctx, s.ast);
    if (ok == 1) {
        smart_string_0(s.ctx->cuda_code_buffer);
        klog("\n", "*** CUDA CODE OUTPUT ***\n%s\n*** END CUDA CODE OUTPUT ***\n",
             s.ctx->cuda_code_buffer->c);
    } else {
        klog("\n", "*** CUDA CODE GENERATION FAILED ***\n");
    }

    kernel_session_cleanup(&s);
    return kernel_ok("kernel compiled");
}


ZEND_METHOD(Kernel, __construct)
{
    zend_class_entry *ce = Z_OBJCE_P(ZEND_THIS);

    klog("", "=== STARTING KERNEL COMPILATION for %s ===\n", ZSTR_VAL(ce->name));

    zend_function *fptr;
    zend_string *method_name;

    ZEND_HASH_FOREACH_STR_KEY_PTR(&ce->function_table, method_name, fptr) {
        if (kernel_is_private_like(method_name)) {
            continue;
        }
        if (!kernel_is_user_method(fptr)) {
            continue;
        }

        kernel_compile_one_method(fptr);
    } ZEND_HASH_FOREACH_END();

    klog("", "=== KERNEL COMPILATION ENDED ===\n");
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

int kernel_init()
{
    zend_class_entry *kernel_ce_local = register_kernel_class();
    kernel_ce_local->create_object = kernel_create_object;

    memcpy(&kernel_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    kernel_handlers.offset   = XtOffsetOf(kernel_obj, obj);
    kernel_handlers.free_obj = kernel_free_object;

    return 1;
}
