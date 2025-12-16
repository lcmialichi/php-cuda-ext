#ifndef DATA_TYPES_H
#define DATA_TYPES_H

typedef enum
{
    FLOAT32,
    FLOAT64,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    BOOL,
    DTYPE_UNKNOWN,
    DTYPE_COUNT,
    LIST,
    VOID
} dtype_t;

#endif