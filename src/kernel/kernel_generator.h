#ifndef KERNEL_GENERATOR_H
#define KERNEL_GENERATOR_H

#include "operations.h"

typedef enum {
    KERNEL_TYPE_ELEMENTWISE, 
    KERNEL_TYPE_REDUCTION,
    KERNEL_TYPE_MIXED, 
} kernel_type_t;

typedef struct
{
    tensor_t **tensors;
    int count;
    int capacity;
} tensor_list_t;

typedef struct _kernel_generator {
    fusion_context_t *context;
    
    int block_size;
    int grid_size; 
    kernel_type_t kernel_type;
    
    char *header_code;
    char *device_code;
    char *kernel_code; 
    char *launch_code;
    
    int total_threads;
    int memory_bytes;
    int num_params;
    
} kernel_generator_t;

kernel_generator_t *kernel_generator_create(fusion_context_t *context);
void kernel_generator_destroy(kernel_generator_t *gen);
bool kernel_generator_analyze(kernel_generator_t *gen);
bool kernel_generator_generate(kernel_generator_t *gen);
void kernel_generator_print(kernel_generator_t *gen);
bool kernel_generator_compile(kernel_generator_t *gen);
bool kernel_generator_execute(kernel_generator_t *gen);

#endif