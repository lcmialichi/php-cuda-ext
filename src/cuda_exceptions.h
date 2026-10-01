#ifndef CUDA_EXCEPTIONS_H
#define CUDA_EXCEPTIONS_H

#include "php.h"
#include "Zend/zend_exceptions.h"

extern zend_class_entry *cuda_exception_ce;
extern zend_class_entry *cuda_runtime_exception_ce;
extern zend_class_entry *cuda_invalid_argument_exception_ce;
extern zend_class_entry *cuda_out_of_memory_exception_ce;
extern zend_class_entry *cuda_compilation_exception_ce;

#define CUDA_THROW_AS(class_entry, ...) \
	do { \
		if (!EG(exception)) { \
			zend_throw_exception_ex(class_entry, 0, __VA_ARGS__); \
		} \
	} while (0)

#define CUDA_THROW_RUNTIME(...) CUDA_THROW_AS(cuda_runtime_exception_ce, __VA_ARGS__)
#define CUDA_THROW_INVALID(...) CUDA_THROW_AS(cuda_invalid_argument_exception_ce, __VA_ARGS__)
#define CUDA_THROW_OOM(...) CUDA_THROW_AS(cuda_out_of_memory_exception_ce, __VA_ARGS__)
#define CUDA_THROW_COMPILATION(...) CUDA_THROW_AS(cuda_compilation_exception_ce, __VA_ARGS__)

void cuda_register_exceptions(void);

#endif