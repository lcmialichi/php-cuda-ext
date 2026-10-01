#include "cuda_exceptions.h"

zend_class_entry *cuda_exception_ce;
zend_class_entry *cuda_runtime_exception_ce;
zend_class_entry *cuda_invalid_argument_exception_ce;
zend_class_entry *cuda_out_of_memory_exception_ce;
zend_class_entry *cuda_compilation_exception_ce;

void cuda_register_exceptions(void)
{
    zend_class_entry ce;

    INIT_NS_CLASS_ENTRY(ce, "Cuda", "Exception", NULL);
    cuda_exception_ce = zend_register_internal_class_ex(&ce, zend_ce_exception);

    INIT_NS_CLASS_ENTRY(ce, "Cuda", "RuntimeException", NULL);
    cuda_runtime_exception_ce = zend_register_internal_class_ex(&ce, cuda_exception_ce);

    INIT_NS_CLASS_ENTRY(ce, "Cuda", "InvalidArgumentException", NULL);
    cuda_invalid_argument_exception_ce = zend_register_internal_class_ex(&ce, cuda_exception_ce);

    INIT_NS_CLASS_ENTRY(ce, "Cuda", "OutOfMemoryException", NULL);
    cuda_out_of_memory_exception_ce = zend_register_internal_class_ex(&ce, cuda_runtime_exception_ce);

    INIT_NS_CLASS_ENTRY(ce, "Cuda", "CompilationException", NULL);
    cuda_compilation_exception_ce = zend_register_internal_class_ex(&ce, cuda_runtime_exception_ce);
}