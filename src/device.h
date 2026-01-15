#ifndef DEVICE_H
#define DEVICE_H

#include "php.h"
#include "kernel_types.h"

extern zend_class_entry *cuda_device_ce;

int device_init(void);

ZEND_METHOD(Device, __construct);
ZEND_METHOD(Device, getName);
ZEND_METHOD(Device, fn);

#endif