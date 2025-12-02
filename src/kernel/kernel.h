#ifndef KERNEL_H
#define KERNEL_H

#include "php.h"

extern zend_class_entry *kernel_ce;

typedef struct
{
    zend_object obj;

} kernel_obj;


int kernel_init();

ZEND_METHOD(Kernel, fusion);

#endif
