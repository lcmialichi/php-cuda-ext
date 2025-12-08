#ifndef KERNEL_H
#define KERNEL_H

#include "php.h"

extern zend_class_entry *kernel_ce;

typedef struct _kernel_obj {
    zend_object obj;
    int is_compiled;
} kernel_obj;

typedef struct cuda_context {
    char *code;
    size_t size;
    size_t capacity;
    int indent_level;
    int temp_var_counter;
    zend_bool in_device_function;
} cuda_context_t;


int kernel_init();

ZEND_METHOD(Kernel, __construct);

#endif
