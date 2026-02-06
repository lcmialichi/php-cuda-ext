#ifndef AST_CUDA_BUILTINS_H
#define AST_CUDA_BUILTINS_H

#include "ast_cuda_types.h"
#include "php.h"

typedef enum
{
    FUNC_CATEGORY_MATH,
    FUNC_CATEGORY_SYSTEM,
    FUNC_CATEGORY_ATOMIC,
    FUNC_CATEGORY_MEMORY,
    FUNC_CATEGORY_SYNC,
    FUNC_CATEGORY_WARP,
    FUNC_CATEGORY_OTHER
} cuda_func_category_t;

typedef struct
{
    const char *php_name;
    const char *cuda_name_f32;
    const char *cuda_name_f64;
    const char *cuda_name_i32;
    dtype_t return_type_f32;
    dtype_t return_type_f64;
    dtype_t return_type_i32;
    uint32_t num_params;
    dtype_t param_types_f32[4];
    dtype_t param_types_f64[4];
    dtype_t param_types_i32[4];
    const char *header;
    cuda_func_category_t category;
    zend_bool requires_this;
} cuda_function_info_t;

const cuda_function_info_t *find_cuda_function_by_category(const char *php_name, cuda_func_category_t category);
const cuda_function_info_t *find_cuda_function(const char *php_name);
cuda_function_match_t find_cuda_function_by_type(const char *php_name, dtype_t arg_types[], uint32_t num_args);

func_parameter *find_kernel_parameter(func_parameter_list_t *list, const char *name);
zend_bool types_are_compatible(dtype_t t1, dtype_t t1_second, dtype_t t2, dtype_t t2_second);

dtype_t determine_dominant_type(dtype_t arg_types[], uint32_t num_args);

const char *get_cuda_type_str(dtype_t type, dtype_t second_dtype);
const char *get_ast_kind_name(zend_ast_kind kind);
const char *get_binary_op_symbol(uint32_t op_type);
const char *get_assign_op_symbol(uint32_t op_type);
const char *get_cuda_object_name(int obj_type);
const char *get_unary_op_symbol(uint32_t op_type);
#endif
