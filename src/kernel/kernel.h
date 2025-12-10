#ifndef KERNEL_H
#define KERNEL_H

#include "php.h"

extern zend_class_entry *kernel_ce;

int kernel_init();

ZEND_METHOD(Kernel, __construct);
ZEND_METHOD(Kernel, fn);

#endif
