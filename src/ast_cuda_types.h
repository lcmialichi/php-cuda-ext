#ifndef AST_CUDA_TYPES_H
#define AST_CUDA_TYPES_H
#include "php.h"
#include "kernel_types.h"
#include "data_types.h"

#define MAX_P_NAME_LEN 32

typedef struct
{
    zend_string *name;
    dtype_t dtype;
    dtype_t second_dtype;
    int array_dimensions;
    int level;
    enum
    {
        VAR_LOCAL,
        VAR_LOCAL_SHARED
    } var_type;

} local_variable_t;

typedef enum
{
    INPUT,
    OUTPUT,
    PARAMETER
} parameter_type_t;

typedef struct
{
    char name[MAX_P_NAME_LEN];
    dtype_t dtype;
    dtype_t second_dtype;
    parameter_type_t type;
} func_parameter;

typedef struct
{
    int total;
    func_parameter **parameters;
} func_parameter_list_t;

typedef struct
{
    zend_string *filename;
    uint32_t start_line;
    uint32_t end_line;
    zend_string *method_name;
    zend_function *fptr;
} method_source_info_t;

typedef enum
{
    FN_KERNEL,
    FN_DEVICE,
    FN_GLOBAL

} cuda_fn_type;

typedef struct
{
    enum
    {
        CUDA_OBJ_NONE,
        CUDA_OBJ_CUDA,
        CUDA_OBJ_MATH,
        CUDA_OBJ_ATOMIC,
        CUDA_OBJ_SYNC,
        CUDA_OBJ_WARP,
        CUDA_OBJ_THREADIDX,
        CUDA_OBJ_BLOCKIDX,
        CUDA_OBJ_BLOCKDIM,
        CUDA_OBJ_GRIDDIM
    } current_cuda_object;
    HashTable *headers;
    cuda_fn_type fn_type;
    zend_string *name;
    func_parameter_list_t *parameters;
    HashTable local_variables;
    HashTable shared_memory_vars;
    smart_string *cuda_code_buffer;
    int dim_access;
    dtype_t last_evaluated_first_dtype;
    dtype_t last_evaluated_second_dtype;
    dtype_t return_dtype;
    int loop_depth;
    int uses_shared_memory;
    int uses_static_shared_memory;
    int shared_memory_declared;
    int current_line;
} cuda_compilation_context_t;

typedef struct _cuda_kernel_data
{
    zend_string *name;
    char *cuda_code;
    func_parameter_list_t *parameters;
} cuda_kernel_data;

#endif