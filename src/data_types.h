#ifndef DATA_TYPES_H
#define DATA_TYPES_H
typedef enum
{
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT8,
    DTYPE_INT16,
    DTYPE_INT32,
    DTYPE_INT64,
    DTYPE_UINT8,
    DTYPE_UINT16,
    DTYPE_UINT32,
    DTYPE_UINT64,
    DTYPE_BOOL,
    DTYPE_UNKNOWN,
    DTYPE_COUNT,
    DTYPE_LIST,
    DTYPE_VOID
} dtype_t;

#endif