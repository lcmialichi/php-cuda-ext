#ifndef KERNEL_H
#define KERNEL_H

#include "php.h"

extern zend_class_entry *kernel_ce;

typedef struct _kernel_obj {
    zend_object obj;
    int is_compiled;
} kernel_obj;


int kernel_init();

ZEND_METHOD(Kernel, __construct);

#endif
