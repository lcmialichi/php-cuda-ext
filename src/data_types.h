#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <string.h>
#include <stdint.h>

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

typedef struct {
    const char* name;
    size_t size;
    int is_floating;
    int is_integer;
    int is_signed;
    int is_unsigned;
    int is_boolean;
    int default_device;
} dtype_info_t;

extern const dtype_info_t dtype_infos[DTYPE_COUNT];

const char* dtype_to_string(dtype_t dtype);
size_t dtype_size(dtype_t dtype);
int dtype_is_floating(dtype_t dtype);
int dtype_is_integer(dtype_t dtype);
int dtype_is_signed(dtype_t dtype);
int dtype_is_boolean(dtype_t dtype);

dtype_t dtype_from_string(const char* type_str);
int is_valid_dtype_string(const char* type_str);

dtype_t promote_types(dtype_t a, dtype_t b);
dtype_t promote_types_for_arithmetic(dtype_t a, dtype_t b);
dtype_t promote_types_for_comparison(dtype_t a, dtype_t b);
dtype_t promote_types_for_logical(dtype_t a, dtype_t b);
int can_safely_cast_to(dtype_t from, dtype_t to);

#endif