
#ifndef KERNEL_TYPES_H
#define KERNEL_TYPES_H

#include "php.h"
#include "zend_compile.h"
#include "data_types.h"
typedef struct
{
    zend_string *name;
    dtype_t dtype;

} local_variable_t;


typedef struct {
    const char *cuda_name;
    dtype_t return_type;
} cuda_function_match_t;

typedef struct _kernel_obj {
    zend_object obj;
} kernel_obj;

typedef enum
{
    INPUT,
    OUTPUT,
    PARAMETER
} parameter_type_t;

typedef struct
{
    char name[32];
    dtype_t dtype;
    parameter_type_t type;
} func_parameter;

typedef struct
{
    int total;
    func_parameter **parameters;
} func_parameter_list_t;

typedef struct {
    zend_string *filename;
    uint32_t start_line;
    uint32_t end_line;
    zend_string *method_name;
    zend_function *fptr;
} method_source_info_t;

typedef struct
{
    func_parameter_list_t *parameters;
    HashTable local_variables;
    smart_string *cuda_code_buffer;
    dtype_t last_evaluated_dtype;
    dtype_t return_dtype;
    int loop_depth;
} cuda_compilation_context_t;

typedef struct _cuda_compiler_object {
    zend_object std;
    char *target_device;
    int optimization_level;
    zend_bool debug_mode;
    zend_bool fast_math;
    HashTable *kernels;
    HashTable *devices;
    cuda_compilation_context_t *compilation_context;
} cuda_compiler_object;

typedef struct _cuda_kernel_data {
    zend_string *name;
    zend_string *target;
    int grid[3];
    int block[3];
    zend_fcall_info fci;
    zend_fcall_info_cache fcc;
    zend_ast *ast;
    zend_arena *ast_arena;
    zend_string *source_code;
    HashTable *used_devices;
    func_parameter_list_t *parameters;
} cuda_kernel_data;

typedef struct _cuda_kernel_object {
    zend_object std;
    zend_fcall_info fci;
    zend_fcall_info_cache fcc;
    zend_string *name;
    zend_string *target;
    int grid[3];    
    int block[3];
    zend_ast *ast;
    zend_arena *ast_arena;
    HashTable *used_devices; 
    func_parameter_list_t *parameters;
} cuda_kernel_object;

typedef struct _cuda_device_object {
    zend_object std;
    zend_string *name;
    zend_string *target;
    zend_fcall_info fci;
    zend_fcall_info_cache fcc;
    HashTable *attributes;
    zend_ast *ast;
    zend_arena *ast_arena;
} cuda_device_object;

typedef struct _cuda_module_object {
    zend_object std;
    char *ptx_code;
    size_t ptx_size;
    HashTable *functions;
    HashTable *kernel_functions;
} cuda_module_object;

#define Z_CUDA_KERNEL_P(zv) ((cuda_kernel_object*)((char*)Z_OBJ_P(zv) - XtOffsetOf(cuda_kernel_object, std)))
#define Z_CUDA_KERNEL_FROM_OBJ(obj) ((cuda_kernel_object*)((char*)(obj) - XtOffsetOf(cuda_kernel_object, std)))
#define Z_CUDA_DEVICE_P(zv) ((cuda_device_object*)((char*)Z_OBJ_P(zv) - XtOffsetOf(cuda_device_object, std)))
#define Z_CUDA_DEVICE_FROM_OBJ(obj) ((cuda_device_object*)((char*)(obj) - XtOffsetOf(cuda_device_object, std)))
#define Z_CUDA_MODULE_P(zv) ((cuda_module_object*)((char*)Z_OBJ_P(zv) - XtOffsetOf(cuda_module_object, std)))
#define Z_CUDA_MODULE_FROM_OBJ(obj) ((cuda_module_object*)((char*)(obj) - XtOffsetOf(cuda_module_object, std)))
#define Z_CUDA_COMPILER_P(zv) ((cuda_compiler_object*)((char*)Z_OBJ_P(zv) - XtOffsetOf(cuda_compiler_object, std)))
#define Z_CUDA_COMPILER_FROM_OBJ(obj) ((cuda_compiler_object*)((char*)(obj) - XtOffsetOf(cuda_compiler_object, std)))

#endif