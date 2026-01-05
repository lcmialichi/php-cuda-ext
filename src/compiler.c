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
#include "ext/standard/php_smart_string.h"
#include "cuda_globals.h"
#include "ext/hash/php_hash.h"
#include "ext/standard/md5.h"
#include <time.h>

#include <nvrtc.h>
#include <cuda.h>
#include <stdio.h>
#include <string.h>

zend_class_entry *cuda_compiler_ce;
static zend_object_handlers compiler_handlers;

static void compiler_free_object(zend_object *object);
static zend_object *compiler_create_object(zend_class_entry *class_type);
static char *read_entire_file(const char *filename, size_t *out_len);
static char *extract_function_body_for_ast(const char *source, size_t source_len,
                                           uint32_t start_line, uint32_t end_line,
                                           size_t *out_len);
static char *build_complete_cuda_program(cuda_compiler_object *compiler, size_t *out_len);
static func_parameter_list_t *copy_parameter_list(func_parameter_list_t *src);
static char *compute_program_hash(cuda_compiler_object *compiler);

static cuda_kernel_data *copy_kernel_data(cuda_kernel_data *src);

static void free_parameter_list(func_parameter_list_t *params);
static void free_kernel_data(cuda_kernel_data *kernel);

static const char *g_cached_nvrtc_options[32] = {0};
static int g_cached_option_count = 0;
static char g_cached_target[64] = {0};
static zend_long g_cached_opt_level = 0;
static zend_bool g_cached_debug = 0;
static zend_bool g_cached_fast_math = 0;

static const char *get_nvrtc_error_string(nvrtcResult result)
{
    switch (result)
    {
    case NVRTC_SUCCESS:
        return "NVRTC_SUCCESS";
    case NVRTC_ERROR_OUT_OF_MEMORY:
        return "NVRTC_ERROR_OUT_OF_MEMORY";
    case NVRTC_ERROR_PROGRAM_CREATION_FAILURE:
        return "NVRTC_ERROR_PROGRAM_CREATION_FAILURE";
    case NVRTC_ERROR_INVALID_INPUT:
        return "NVRTC_ERROR_INVALID_INPUT";
    case NVRTC_ERROR_INVALID_PROGRAM:
        return "NVRTC_ERROR_INVALID_PROGRAM";
    case NVRTC_ERROR_INVALID_OPTION:
        return "NVRTC_ERROR_INVALID_OPTION";
    case NVRTC_ERROR_COMPILATION:
        return "NVRTC_ERROR_COMPILATION";
    case NVRTC_ERROR_BUILTIN_OPERATION_FAILURE:
        return "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE";
    case NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
        return "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
    case NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
        return "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
    case NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
        return "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
    case NVRTC_ERROR_INTERNAL_ERROR:
        return "NVRTC_ERROR_INTERNAL_ERROR";
    default:
        return "Unknown NVRTC error";
    }
}

static const char *get_cuda_error_string(CUresult result)
{
    switch (result)
    {
    case CUDA_SUCCESS:
        return "CUDA_SUCCESS";
    case CUDA_ERROR_INVALID_VALUE:
        return "CUDA_ERROR_INVALID_VALUE";
    case CUDA_ERROR_OUT_OF_MEMORY:
        return "CUDA_ERROR_OUT_OF_MEMORY";
    case CUDA_ERROR_NOT_INITIALIZED:
        return "CUDA_ERROR_NOT_INITIALIZED";
    case CUDA_ERROR_DEINITIALIZED:
        return "CUDA_ERROR_DEINITIALIZED";
    case CUDA_ERROR_PROFILER_DISABLED:
        return "CUDA_ERROR_PROFILER_DISABLED";
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
        return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
    case CUDA_ERROR_PROFILER_ALREADY_STARTED:
        return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
        return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
    default:
        return "Unknown CUDA error";
    }
}

static void ensure_common_headers(cuda_compiler_object *compiler)
{
    static const char *common_headers[] = {
        "#include <cuda_runtime.h>",
        "#include <device_launch_parameters.h>",
        "#include <cuda_fp16.h>",
        NULL};

    for (int i = 0; common_headers[i]; i++)
    {
        zend_string *header = zend_string_init(common_headers[i], strlen(common_headers[i]), 1);

        if (!zend_hash_exists(compiler->headers, header))
        {
            zend_hash_add_ptr(compiler->headers, header, header);
        }
        else
        {
            zend_string_release(header);
        }
    }
}

static void append_math_constants(smart_string *program)
{
    static const char *math_constants =
        "// Math constants for CUDA\n"
        "#ifndef M_PI\n"
        "#define M_PI 3.14159265358979323846f\n"
        "#endif\n\n"
        "#ifndef INFINITY\n"
        "#define INFINITY __int_as_float(0x7f800000)\n"
        "#endif\n\n"
        "#ifndef NAN\n"
        "#define NAN __int_as_float(0x7fffffff)\n"
        "#endif\n\n"
        "#ifndef FLT_MAX\n"
        "#define FLT_MAX 3.402823466e+38f\n"
        "#endif\n\n"
        "#ifndef FLT_MIN\n"
        "#define FLT_MIN 1.175494351e-38f\n"
        "#endif\n\n"
        "#ifndef INF\n"
        "#define INF INFINITY\n"
        "#endif\n\n";

    smart_string_appendl(program, math_constants, strlen(math_constants));
}

static char *read_entire_file(const char *filename, size_t *out_len)
{
    FILE *file = fopen(filename, "r");
    if (!file)
        return NULL;

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *buffer = (char *)emalloc(file_size + 1);
    size_t read_size = fread(buffer, 1, file_size, file);
    buffer[read_size] = '\0';

    fclose(file);

    if (out_len)
        *out_len = read_size;
    return buffer;
}

static char *extract_function_body_for_ast(const char *source, size_t source_len,
                                           uint32_t start_line, uint32_t end_line,
                                           size_t *out_len)
{
    if (start_line == 0 || end_line == 0 || start_line > end_line)
    {
        return NULL;
    }

    const char **line_offsets = (const char **)emalloc(sizeof(char *) * (end_line + 3));
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
    const char *func_end = (end_line < current_line) ? line_offsets[end_line + 1] : source + source_len;

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

    char *output = (char *)emalloc(prefix_len + body_len + 1);
    memcpy(output, prefix, prefix_len);
    memcpy(output + prefix_len, body_start, body_len);
    output[prefix_len + body_len] = '\0';

    if (out_len)
        *out_len = prefix_len + body_len;

    efree(line_offsets);
    return output;
}

static char *build_complete_cuda_program(cuda_compiler_object *compiler, size_t *out_len)
{
    size_t estimated_size = 4096;
    cuda_kernel_data *kernel;

    ZEND_HASH_FOREACH_PTR(compiler->kernels, kernel)
    {
        if (kernel->cuda_code)
        {
            estimated_size += strlen(kernel->cuda_code) + 100;
        }
    }
    ZEND_HASH_FOREACH_END();

    smart_string program = {0};
    smart_string_alloc(&program, estimated_size, 0);

    smart_string_appendl(&program,
                         "#include <cuda_runtime.h>\n"
                         "#include <device_launch_parameters.h>\n"
                         "#include <cuda_fp16.h>\n\n",
                         0);

    append_math_constants(&program);

    zval *header_zv;
    ZEND_HASH_FOREACH_VAL(compiler->headers, header_zv)
    {
        if (Z_TYPE_P(header_zv) == IS_STRING)
        {
            smart_string_appendl(&program, Z_STRVAL_P(header_zv), Z_STRLEN_P(header_zv));
            smart_string_appendl(&program, "\n", 1);
        }
    }
    ZEND_HASH_FOREACH_END();

    cuda_device_object *device;
    ZEND_HASH_FOREACH_PTR(compiler->devices, device)
    {
        if (device && device->name)
        {
            smart_string_appendl(&program, "\n// Device function: ", strlen("\n// Device function: "));
            smart_string_appendl(&program, ZSTR_VAL(device->name), ZSTR_LEN(device->name));
            smart_string_appendl(&program, "\n", 1);
        }
    }
    ZEND_HASH_FOREACH_END();

    ZEND_HASH_FOREACH_PTR(compiler->kernels, kernel)
    {
        if (kernel && kernel->name && kernel->cuda_code)
        {
            smart_string_appendl(&program, "\n// Kernel: ", strlen("\n// Kernel: "));
            smart_string_appendl(&program, ZSTR_VAL(kernel->name), ZSTR_LEN(kernel->name));
            smart_string_appendl(&program, "\n", 1);
            smart_string_appendl(&program, kernel->cuda_code, strlen(kernel->cuda_code));
            smart_string_appendl(&program, "\n", 1);
        }
    }
    ZEND_HASH_FOREACH_END();

    smart_string_0(&program);

    if (out_len)
        *out_len = program.len;
    return program.c;
}

static char *compute_program_hash(cuda_compiler_object *compiler)
{
    smart_string program_hash_content = {0};
    smart_string_alloc(&program_hash_content, 8192, 0);

    cuda_kernel_data *kernel;
    ZEND_HASH_FOREACH_PTR(compiler->kernels, kernel)
    {
        if (kernel->cuda_code)
        {
            smart_string_appendl(&program_hash_content, kernel->cuda_code, strlen(kernel->cuda_code));
            smart_string_appendc(&program_hash_content, ';');
        }
    }
    ZEND_HASH_FOREACH_END();

    char config[256];
    snprintf(config, sizeof(config), "target:%s:opt:%d:debug:%d:fastmath:%d",
             compiler->target_device ? compiler->target_device : "default",
             (int)compiler->optimization_level,
             compiler->debug_mode,
             compiler->fast_math);
    smart_string_appendl(&program_hash_content, config, strlen(config));

    smart_string_0(&program_hash_content);

    PHP_MD5_CTX context;
    unsigned char digest[16];
    char hexdigest[33];

    PHP_MD5Init(&context);
    PHP_MD5Update(&context, (const unsigned char *)program_hash_content.c, program_hash_content.len);
    PHP_MD5Final(digest, &context);

    for (int i = 0; i < 16; i++)
    {
        sprintf(hexdigest + (i * 2), "%02x", digest[i]);
    }
    hexdigest[32] = '\0';

    smart_string_free(&program_hash_content);

    return estrdup(hexdigest);
}

static int get_cached_nvrtc_options(cuda_compiler_object *compiler, const char ***options_out)
{
    const char *current_target = compiler->target_device ? compiler->target_device : "sm_60";
    if (g_cached_option_count > 0 &&
        strcmp(g_cached_target, current_target) == 0 &&
        g_cached_opt_level == compiler->optimization_level &&
        g_cached_debug == compiler->debug_mode &&
        g_cached_fast_math == compiler->fast_math)
    {
        *options_out = g_cached_nvrtc_options;
        return g_cached_option_count;
    }

    for (int i = 0; i < g_cached_option_count; i++)
    {
        if (g_cached_nvrtc_options[i])
        {
            efree((void *)g_cached_nvrtc_options[i]);
            g_cached_nvrtc_options[i] = NULL;
        }
    }
    g_cached_option_count = 0;

    char arch_opt[128];
    snprintf(arch_opt, sizeof(arch_opt), "-arch=%s", current_target);
    g_cached_nvrtc_options[g_cached_option_count++] = estrdup(arch_opt);

    if (compiler->debug_mode)
    {
        g_cached_nvrtc_options[g_cached_option_count++] = estrdup("-G");
        g_cached_nvrtc_options[g_cached_option_count++] = estrdup("-lineinfo");
    }

    if (compiler->fast_math && !compiler->debug_mode)
    {
        g_cached_nvrtc_options[g_cached_option_count++] = estrdup("-use_fast_math");
    }

    g_cached_nvrtc_options[g_cached_option_count++] = estrdup(CUDA_INCLUDE_PATH_STR);
    g_cached_nvrtc_options[g_cached_option_count++] = estrdup(CUDA_CRT_INCLUDE_STR);
    g_cached_nvrtc_options[g_cached_option_count++] = estrdup("-I.");
    g_cached_nvrtc_options[g_cached_option_count++] = estrdup("-std=c++11");
    g_cached_nvrtc_options[g_cached_option_count++] = estrdup("-restrict");

    strncpy(g_cached_target, current_target, sizeof(g_cached_target) - 1);
    g_cached_opt_level = compiler->optimization_level;
    g_cached_debug = compiler->debug_mode;
    g_cached_fast_math = compiler->fast_math;

    *options_out = g_cached_nvrtc_options;
    return g_cached_option_count;
}

static nvrtcResult compile_with_nvrtc(const char *cuda_program, size_t program_len,
                                      const char **options, int option_count,
                                      char **ptx_out, size_t *ptx_size_out)
{
    nvrtcProgram prog;
    nvrtcResult result;

    result = nvrtcCreateProgram(&prog, cuda_program, "kernel.cu", 0, NULL, NULL);
    if (result != NVRTC_SUCCESS)
    {
        return result;
    }

    result = nvrtcCompileProgram(prog, option_count, options);

    size_t log_size;
    nvrtcGetProgramLogSize(prog, &log_size);
    if (log_size > 1)
    {
        char *compile_log = (char *)emalloc(log_size);
        nvrtcGetProgramLog(prog, compile_log);
        if (compile_log[0] != '\0')
        {
            php_printf("NVRTC Compilation Log:\n%s\n", compile_log);
        }
        efree(compile_log);
    }

    if (result != NVRTC_SUCCESS)
    {
        nvrtcDestroyProgram(&prog);
        return result;
    }

    result = nvrtcGetPTXSize(prog, ptx_size_out);
    if (result != NVRTC_SUCCESS)
    {
        nvrtcDestroyProgram(&prog);
        return result;
    }

    *ptx_out = (char *)emalloc(*ptx_size_out + 1);
    result = nvrtcGetPTX(prog, *ptx_out);
    (*ptx_out)[*ptx_size_out] = '\0';

    nvrtcDestroyProgram(&prog);
    return result;
}

static void free_parameter_list(func_parameter_list_t *params)
{
    if (!params)
        return;

    if (params->total < 0 || params->total > 1000)
    {
        efree(params);
        return;
    }

    if (params->parameters)
    {
        for (int i = 0; i < params->total; i++)
        {
            func_parameter *param = params->parameters[i];
            if (param)
            {
                efree(param);
                params->parameters[i] = NULL;
            }
        }

        efree(params->parameters);
        params->parameters = NULL;
    }

    efree(params);
}

static void free_kernel_data(cuda_kernel_data *kernel)
{
    if (!kernel){
        return;
    }

    if (kernel->name)
    {
        zend_string_release(kernel->name);
        kernel->name = NULL;
    }

    if (kernel->parameters != NULL)
    {
        free_parameter_list(kernel->parameters);
        kernel->parameters = NULL;
    }

    if (kernel->cuda_code)
    {
        efree(kernel->cuda_code);
        kernel->cuda_code = NULL;
    }

    if (kernel->used_devices)
    {
        zend_hash_destroy(kernel->used_devices);
        efree(kernel->used_devices);
        kernel->used_devices = NULL;
    }

    efree(kernel);
}

static cuda_kernel_data *copy_kernel_data(cuda_kernel_data *src)
{
    if (!src)
        return NULL;

    cuda_kernel_data *dst = ecalloc(1, sizeof(cuda_kernel_data));
    dst->target = NULL;
    dst->used_devices = NULL;

    if (src->name)
    {
        dst->name = zend_string_copy(src->name);
    }

    if (src->parameters)
    {
        dst->parameters = copy_parameter_list(src->parameters);
    }

    if (src->cuda_code)
    {
        dst->cuda_code = estrdup(src->cuda_code);
    }

    return dst;
}

static func_parameter_list_t *copy_parameter_list(func_parameter_list_t *src)
{
    if (!src) return NULL;

    func_parameter_list_t *dst = ecalloc(1, sizeof(func_parameter_list_t));
    if (src->total < 0 || src->total > 1000) {
        return dst;
    }

    dst->total = src->total;

    if (src->total > 0 && src->parameters) {
        dst->parameters = ecalloc(src->total, sizeof(func_parameter *));

        for (int i = 0; i < src->total; i++) {
            if (src->parameters[i]) {
                dst->parameters[i] = emalloc(sizeof(func_parameter));
                memcpy(dst->parameters[i], src->parameters[i], sizeof(func_parameter));
            }
        }
    }

    return dst;
}

ZEND_METHOD(Compiler, __construct)
{
    cuda_compiler_object *compiler;
    zend_string *target_str = NULL;
    zend_long optimization = 2;
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

    compiler->devices = (HashTable *)emalloc(sizeof(HashTable));
    zend_hash_init(compiler->devices, 8, NULL, NULL, 0);

    compiler->ptx_cache = (HashTable *)emalloc(sizeof(HashTable));
    zend_hash_init(compiler->ptx_cache, 8, NULL, NULL, 0);

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

    ensure_common_headers(compiler);
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
    zend_function *fptr = fcc.function_handler;

    if (!fptr || fptr->type != ZEND_USER_FUNCTION)
    {
        zend_throw_exception_ex(NULL, 0, "Invalid kernel function");
        return;
    }

    if (fptr->op_array.fn_flags & ZEND_ACC_USES_THIS)
    {
        zend_throw_exception_ex(NULL, 0, "Kernel functions cannot use object context");
        return;
    }

    if (fptr->op_array.static_variables != NULL)
    {
        zend_throw_exception_ex(NULL, 0, "CUDA Runtime cannot access outer context variables");
        return;
    }

    cuda_method_attribute_args *fargs = cuda_extract_method_attribute(fptr, cuda_attr_kernel_ce);
    if (!fargs)
    {
        zend_throw_exception_ex(NULL, 0, "Failed to extract kernel attributes");
        return;
    }

    zend_op_array *op_array = &fptr->op_array;
    if (!op_array->filename || op_array->line_start == 0 || op_array->line_end == 0)
    {
        efree(fargs);
        zend_throw_exception_ex(NULL, 0, "Cannot locate kernel source");
        return;
    }

    size_t file_len = 0;
    char *file_content = read_entire_file(ZSTR_VAL(op_array->filename), &file_len);
    if (!file_content)
    {
        efree(fargs);
        zend_throw_exception_ex(NULL, 0, "Cannot read kernel source file");
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
        zend_throw_exception_ex(NULL, 0, "Cannot parse function body");
        efree(fargs);
        return;
    }

    zend_arena *ast_arena = NULL;
    zend_ast *ast = zend_compile_string_to_ast(source_code, &ast_arena, fargs->name);

    func_parameter_list_t *params = cuda_extract_parameters(fptr);
    cuda_compilation_context_t *ctx = create_cuda_context(params, FN_KERNEL, fargs->name, compiler->headers);

    if (compile_ast_to_cuda_fn(ctx, ast) != 1)
    {
        free_cuda_context(ctx);
        zend_string_release(source_code);
        free_parameter_list(params);
        efree(fargs);
        return;
    }

    zend_ast_destroy(ast);
    zend_arena_destroy(ast_arena);
    zend_string_release(source_code);

    smart_string_0(ctx->cuda_code_buffer);

    cuda_kernel_data *kernel = (cuda_kernel_data *)ecalloc(1, sizeof(cuda_kernel_data));
    if (!fargs->name)
    {
        zend_throw_exception_ex(NULL, 0, "Kernel name is NULL");
        efree(fargs);
        efree(kernel);
        return;
    }

    kernel->name = zend_string_dup(fargs->name, 0);
    kernel->parameters = params;
    kernel->used_devices = (HashTable *)emalloc(sizeof(HashTable));
    zend_hash_init(kernel->used_devices, 4, NULL, NULL, 0);

    if (ctx->cuda_code_buffer && ctx->cuda_code_buffer->c)
    {
        kernel->cuda_code = estrdup(ctx->cuda_code_buffer->c);
    }
    else
    {
        kernel->cuda_code = estrdup("");
    }

    zend_hash_update_ptr(compiler->kernels, kernel->name, kernel);
    zend_hash_clean(compiler->ptx_cache);
    free_cuda_context(ctx);

    efree(fargs);

    RETURN_ZVAL(getThis(), 1, 0);
}

ZEND_METHOD(Compiler, device)
{
    cuda_compiler_object *compiler;
    zend_fcall_info fci;
    zend_fcall_info_cache fcc;

    ZEND_PARSE_PARAMETERS_START(1, 1)
    Z_PARAM_FUNC(fci, fcc)
    ZEND_PARSE_PARAMETERS_END();

    compiler = Z_CUDA_COMPILER_P(ZEND_THIS);
    zend_function *fptr = fcc.function_handler;

    if (!fptr || fptr->type != ZEND_USER_FUNCTION)
    {
        zend_throw_exception_ex(NULL, 0, "Invalid device function");
        return;
    }

    cuda_method_attribute_args *fargs = cuda_extract_method_attribute(fptr, cuda_attr_device_ce);
    if (!fargs)
    {
        zend_throw_exception_ex(NULL, 0, "Failed to extract device attributes");
        return;
    }

    zend_string *device_class_name = zend_string_init("Cuda\\Device", strlen("Cuda\\Device"), 0);
    zend_class_entry *device_ce = zend_lookup_class(device_class_name);
    zend_string_release(device_class_name);

    if (!device_ce)
    {
        efree(fargs);
        zend_throw_exception_ex(NULL, 0, "Class Cuda\\Device not found");
        RETURN_NULL();
    }

    zval device_zv;
    object_init_ex(&device_zv, device_ce);
    cuda_device_object *device = Z_CUDA_DEVICE_P(&device_zv);

    device->name = zend_string_copy(fargs->name);

    zend_string *key_name = zend_string_copy(device->name);
    zend_hash_add_ptr(compiler->devices, key_name, device);

    zend_hash_clean(compiler->ptx_cache);

    zend_string_release(fargs->name);
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

    if (target != NULL)
    {
        if (compiler->target_device)
        {
            efree(compiler->target_device);
        }
        compiler->target_device = estrndup(target, target_len);
    }

    char *program_hash = compute_program_hash(compiler);
    zend_string *hash_zstr = zend_string_init(program_hash, strlen(program_hash), 0);
    efree(program_hash);

    cached_ptx_t *cached = zend_hash_find_ptr(compiler->ptx_cache, hash_zstr);
    if (cached)
    {
        if (!cached->ptx || cached->ptx_size == 0)
        {
            zend_string_release(hash_zstr);
            zend_throw_exception_ex(NULL, 0, "Invalid cached PTX data");
            RETURN_NULL();
        }

        zend_string *module_class_name = zend_string_init("Cuda\\CompiledModule",
                                                          strlen("Cuda\\CompiledModule"), 0);
        zend_class_entry *module_ce = zend_lookup_class(module_class_name);
        zend_string_release(module_class_name);

        if (!module_ce)
        {
            zend_string_release(hash_zstr);
            zend_throw_exception_ex(NULL, 0, "CompiledModule class not found");
            RETURN_NULL();
        }

        zval module_zv;
        object_init_ex(&module_zv, module_ce);
        cuda_module_object *module = Z_CUDA_MODULE_P(&module_zv);
        module->ptx_code = estrdup(cached->ptx);
        module->ptx_size = cached->ptx_size;
        module->kernel_functions = (HashTable *)emalloc(sizeof(HashTable));
        zend_hash_init(module->kernel_functions, 8, NULL, NULL, 0);

        cuda_kernel_data *src_kernel;
        ZEND_HASH_FOREACH_PTR(compiler->kernels, src_kernel)
        {
            if (!src_kernel || !src_kernel->name)
                continue;

            cuda_kernel_data *kernel_copy = copy_kernel_data(src_kernel);

            zend_string *key_copy = zend_string_copy(kernel_copy->name);
            zend_hash_add_ptr(module->kernel_functions, key_copy, kernel_copy);
            zend_string_release(key_copy);
        }
        ZEND_HASH_FOREACH_END();

        zend_string_release(hash_zstr);
        RETURN_ZVAL(&module_zv, 1, 0);
    }

    size_t program_len;
    char *cuda_program = build_complete_cuda_program(compiler, &program_len);
    if (!cuda_program || program_len == 0)
    {
        if (cuda_program)
        {
            efree(cuda_program);
        }
        zend_string_release(hash_zstr);
        zend_throw_exception_ex(NULL, 0, "Failed to build CUDA program");
        RETURN_NULL();
    }

    const char **options = NULL;
    int option_count = get_cached_nvrtc_options(compiler, &options);

    char *ptx_code = NULL;
    size_t ptx_size = 0;
    nvrtcResult nvrtc_result = compile_with_nvrtc(cuda_program, program_len,
                                                  options, option_count,
                                                  &ptx_code, &ptx_size);
    efree(cuda_program);

    if (nvrtc_result != NVRTC_SUCCESS)
    {
        if (ptx_code)
        {
            free(ptx_code);
        }

        zend_string_release(hash_zstr);
        zend_throw_exception_ex(NULL, 0,
                                "NVRTC compilation failed: %s (code: %d)",
                                get_nvrtc_error_string(nvrtc_result), nvrtc_result);
        RETURN_NULL();
    }

    zend_string *module_class_name = zend_string_init("Cuda\\CompiledModule",
                                                      strlen("Cuda\\CompiledModule"), 0);
    zend_class_entry *module_ce = zend_lookup_class(module_class_name);
    zend_string_release(module_class_name);

    if (!module_ce)
    {
        free(ptx_code);
        zend_string_release(hash_zstr);
        zend_throw_exception_ex(NULL, 0, "CompiledModule class not found");
        RETURN_NULL();
    }

    zval module_zv;
    object_init_ex(&module_zv, module_ce);
    cuda_module_object *module = Z_CUDA_MODULE_P(&module_zv);

    module->ptx_code = estrndup(ptx_code, ptx_size);
    module->ptx_size = ptx_size;

    module->kernel_functions = (HashTable *)emalloc(sizeof(HashTable));
    zend_hash_init(module->kernel_functions, 8, NULL, NULL, 0);

    cuda_kernel_data *src_kernel;
    ZEND_HASH_FOREACH_PTR(compiler->kernels, src_kernel)
    {
        if (!src_kernel || !src_kernel->name)
        {
            continue;
        }

        cuda_kernel_data *kernel_copy = copy_kernel_data(src_kernel);

        zend_string *key_copy = zend_string_copy(kernel_copy->name);
        zend_hash_add_ptr(module->kernel_functions, key_copy, kernel_copy);
        zend_string_release(key_copy);
    }
    ZEND_HASH_FOREACH_END();

    cached_ptx_t *new_cache = (cached_ptx_t *)emalloc(sizeof(cached_ptx_t));
    new_cache->ptx = estrdup(module->ptx_code);
    new_cache->ptx_size = module->ptx_size;
    new_cache->timestamp = time(NULL);

    zend_hash_add_ptr(compiler->ptx_cache, zend_string_copy(hash_zstr), new_cache);

    zend_string_release(hash_zstr);

    RETURN_ZVAL(&module_zv, 1, 0);
}

ZEND_METHOD(Compiler, clearCache)
{
    cuda_compiler_object *compiler = Z_CUDA_COMPILER_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_NONE();

    zend_hash_clean(compiler->ptx_cache);
    RETURN_TRUE;
}

ZEND_METHOD(Compiler, getCacheStats)
{
    cuda_compiler_object *compiler = Z_CUDA_COMPILER_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_NONE();

    array_init(return_value);

    add_assoc_long(return_value, "cache_entries", zend_hash_num_elements(compiler->ptx_cache));

    size_t total_size = 0;
    cached_ptx_t *cached;

    ZEND_HASH_FOREACH_PTR(compiler->ptx_cache, cached)
    {
        if (cached)
        {
            total_size += cached->ptx_size;
        }
    }
    ZEND_HASH_FOREACH_END();

    add_assoc_long(return_value, "total_cache_size", total_size);
    add_assoc_string(return_value, "target_device",
                     compiler->target_device ? compiler->target_device : "default");
    add_assoc_long(return_value, "optimization_level", compiler->optimization_level);
    add_assoc_bool(return_value, "debug_mode", compiler->debug_mode);
    add_assoc_bool(return_value, "fast_math", compiler->fast_math);
}

ZEND_METHOD(Compiler, getKernels)
{
    cuda_compiler_object *compiler = Z_CUDA_COMPILER_P(ZEND_THIS);
    array_init(return_value);

    cuda_kernel_data *kernel;
    ZEND_HASH_FOREACH_PTR(compiler->kernels, kernel)
    {
        if (!kernel)
            continue;

        zval kernel_info;
        array_init(&kernel_info);

        add_assoc_str(&kernel_info, "name", zend_string_copy(kernel->name));
        if (kernel->target)
        {
            add_assoc_str(&kernel_info, "target", zend_string_copy(kernel->target));
        }
        if (kernel->cuda_code)
        {
            add_assoc_stringl(&kernel_info, "cuda_code", kernel->cuda_code, strlen(kernel->cuda_code));
        }

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
        if (!device)
        {
            continue;
        }

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
        compiler->target_device = NULL;
    }
    if (compiler->kernels)
    {

        cuda_kernel_data *kernel;
        ZEND_HASH_FOREACH_PTR(compiler->kernels, kernel)
        {
            free_kernel_data(kernel);
        }
        ZEND_HASH_FOREACH_END();

        zend_hash_destroy(compiler->kernels);
        efree(compiler->kernels);
        compiler->kernels = NULL;
    }

    if (compiler->devices)
    {
        zend_hash_destroy(compiler->devices);
        efree(compiler->devices);
        compiler->devices = NULL;
    }

    if (compiler->headers)
    {
        zend_hash_destroy(compiler->headers);
        efree(compiler->headers);
        compiler->headers = NULL;
    }

    if (compiler->ptx_cache)
    {
        cached_ptx_t *cached;
        ZEND_HASH_FOREACH_PTR(compiler->ptx_cache, cached)
        {
            if (cached)
            {
                if (cached->ptx)
                    efree(cached->ptx);
                efree(cached);
            }
        }
        ZEND_HASH_FOREACH_END();
        zend_hash_destroy(compiler->ptx_cache);
        efree(compiler->ptx_cache);
        compiler->ptx_cache = NULL;
    }

    zend_object_std_dtor(&compiler->std);
}

static zend_object *compiler_create_object(zend_class_entry *class_type)
{
    cuda_compiler_object *compiler = (cuda_compiler_object *)ecalloc(1, sizeof(cuda_compiler_object));

    zend_object_std_init(&compiler->std, class_type);
    compiler->std.handlers = &compiler_handlers;

    compiler->headers = (HashTable *)emalloc(sizeof(HashTable));
    zend_hash_init(compiler->headers, 8, NULL, NULL, 0);

    compiler->ptx_cache = (HashTable *)emalloc(sizeof(HashTable));
    zend_hash_init(compiler->ptx_cache, 8, NULL, NULL, 0);

    compiler->kernels = (HashTable *)emalloc(sizeof(HashTable));
    zend_hash_init(compiler->kernels, 8, NULL, NULL, 0);

    compiler->target_device = NULL;
    compiler->optimization_level = 2;
    compiler->debug_mode = 0;
    compiler->fast_math = 1;

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