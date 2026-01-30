#include "data_types.h"
#include <ctype.h>
#include <string.h>

const dtype_info_t dtype_infos[DTYPE_COUNT] = {
    [DTYPE_FLOAT32] = {
        .name = "float32",
        .size = sizeof(float),
        .is_floating = 1,
        .is_integer = 0,
        .is_signed = 1,
        .is_unsigned = 0,
        .is_boolean = 0,
        .default_device = 0
    },
    [DTYPE_FLOAT64] = {
        .name = "float64",
        .size = sizeof(double),
        .is_floating = 1,
        .is_integer = 0,
        .is_signed = 1,
        .is_unsigned = 0,
        .is_boolean = 0,
        .default_device = 0
    },
    [DTYPE_INT8] = {
        .name = "int8",
        .size = sizeof(int8_t),
        .is_floating = 0,
        .is_integer = 1,
        .is_signed = 1,
        .is_unsigned = 0,
        .is_boolean = 0,
        .default_device = 0
    },
    [DTYPE_INT16] = {
        .name = "int16",
        .size = sizeof(int16_t),
        .is_floating = 0,
        .is_integer = 1,
        .is_signed = 1,
        .is_unsigned = 0,
        .is_boolean = 0,
        .default_device = 0
    },
    [DTYPE_INT32] = {
        .name = "int32",
        .size = sizeof(int32_t),
        .is_floating = 0,
        .is_integer = 1,
        .is_signed = 1,
        .is_unsigned = 0,
        .is_boolean = 0,
        .default_device = 0
    },
    [DTYPE_INT64] = {
        .name = "int64",
        .size = sizeof(int64_t),
        .is_floating = 0,
        .is_integer = 1,
        .is_signed = 1,
        .is_unsigned = 0,
        .is_boolean = 0,
        .default_device = 0
    },
    [DTYPE_UINT8] = {
        .name = "uint8",
        .size = sizeof(uint8_t),
        .is_floating = 0,
        .is_integer = 1,
        .is_signed = 0,
        .is_unsigned = 1,
        .is_boolean = 0,
        .default_device = 0
    },
    [DTYPE_UINT16] = {
        .name = "uint16",
        .size = sizeof(uint16_t),
        .is_floating = 0,
        .is_integer = 1,
        .is_signed = 0,
        .is_unsigned = 1,
        .is_boolean = 0,
        .default_device = 0
    },
    [DTYPE_UINT32] = {
        .name = "uint32",
        .size = sizeof(uint32_t),
        .is_floating = 0,
        .is_integer = 1,
        .is_signed = 0,
        .is_unsigned = 1,
        .is_boolean = 0,
        .default_device = 0
    },
    [DTYPE_UINT64] = {
        .name = "uint64",
        .size = sizeof(uint64_t),
        .is_floating = 0,
        .is_integer = 1,
        .is_signed = 0,
        .is_unsigned = 1,
        .is_boolean = 0,
        .default_device = 0
    },
    [DTYPE_BOOL] = {
        .name = "bool",
        .size = sizeof(uint8_t),
        .is_floating = 0,
        .is_integer = 0,
        .is_signed = 0,
        .is_unsigned = 0,
        .is_boolean = 1,
        .default_device = 0
    }
};

static void to_lower_case(char* dest, const char* src, size_t max_len) {
    size_t i;
    for (i = 0; src[i] && i < max_len - 1; i++) {
        dest[i] = tolower(src[i]);
    }
    dest[i] = '\0';
}

dtype_t dtype_from_string(const char* type_str) {
    if (!type_str || type_str[0] == '\0') {
        return DTYPE_FLOAT32;
    }
    
    char lower[32];
    to_lower_case(lower, type_str, sizeof(lower));
    
    if (strcmp(lower, "float32") == 0 || strcmp(lower, "float") == 0 || 
        strcmp(lower, "f32") == 0 || strcmp(lower, "single") == 0) {
        return DTYPE_FLOAT32;
    }
    if (strcmp(lower, "float64") == 0 || strcmp(lower, "double") == 0 || 
        strcmp(lower, "f64") == 0) {
        return DTYPE_FLOAT64;
    }
    if (strcmp(lower, "int8") == 0 || strcmp(lower, "char") == 0 || 
        strcmp(lower, "i8") == 0) {
        return DTYPE_INT8;
    }
    if (strcmp(lower, "int16") == 0 || strcmp(lower, "short") == 0 || 
        strcmp(lower, "i16") == 0) {
        return DTYPE_INT16;
    }
    if (strcmp(lower, "int32") == 0 || strcmp(lower, "int") == 0 || 
        strcmp(lower, "i32") == 0) {
        return DTYPE_INT32;
    }
    if (strcmp(lower, "int64") == 0 || strcmp(lower, "long") == 0 || 
        strcmp(lower, "i64") == 0 || strcmp(lower, "longlong") == 0) {
        return DTYPE_INT64;
    }
    if (strcmp(lower, "uint8") == 0 || strcmp(lower, "uchar") == 0 || 
        strcmp(lower, "u8") == 0 || strcmp(lower, "byte") == 0) {
        return DTYPE_UINT8;
    }
    if (strcmp(lower, "uint16") == 0 || strcmp(lower, "ushort") == 0 || 
        strcmp(lower, "u16") == 0) {
        return DTYPE_UINT16;
    }
    if (strcmp(lower, "uint32") == 0 || strcmp(lower, "uint") == 0 || 
        strcmp(lower, "u32") == 0) {
        return DTYPE_UINT32;
    }
    if (strcmp(lower, "uint64") == 0 || strcmp(lower, "ulong") == 0 || 
        strcmp(lower, "u64") == 0) {
        return DTYPE_UINT64;
    }
    if (strcmp(lower, "bool") == 0 || strcmp(lower, "boolean") == 0) {
        return DTYPE_BOOL;
    }
    
    return DTYPE_UNKNOWN;
}

int is_valid_dtype_string(const char* type_str) {
    if (!type_str || type_str[0] == '\0') {
        return 0;
    }
    
    dtype_t dtype = dtype_from_string(type_str);
    return (dtype != DTYPE_UNKNOWN && dtype != DTYPE_COUNT);
}

const char* dtype_to_string(dtype_t dtype) {
    if (dtype >= DTYPE_COUNT) return "unknown";
    return dtype_infos[dtype].name;
}

size_t dtype_size(dtype_t dtype) {
    if (dtype >= DTYPE_COUNT) return 0;
    return dtype_infos[dtype].size;
}

int dtype_is_floating(dtype_t dtype) {
    if (dtype >= DTYPE_COUNT) return 0;
    return dtype_infos[dtype].is_floating;
}

int dtype_is_integer(dtype_t dtype) {
    if (dtype >= DTYPE_COUNT) return 0;
    return dtype_infos[dtype].is_integer;
}

int dtype_is_signed(dtype_t dtype) {
    if (dtype >= DTYPE_COUNT) return 0;
    return dtype_infos[dtype].is_signed;
}

int dtype_is_boolean(dtype_t dtype) {
    if (dtype >= DTYPE_COUNT) return 0;
    return dtype_infos[dtype].is_boolean;
}

static const dtype_t type_hierarchy[] = {
    DTYPE_BOOL,
    DTYPE_UINT8,
    DTYPE_UINT16,
    DTYPE_UINT32,
    DTYPE_UINT64,
    DTYPE_INT8,
    DTYPE_INT16,
    DTYPE_INT32,
    DTYPE_INT64,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
};

static const int type_hierarchy_size = sizeof(type_hierarchy) / sizeof(type_hierarchy[0]);

static int type_index(dtype_t dtype) {
    for (int i = 0; i < type_hierarchy_size; i++) {
        if (type_hierarchy[i] == dtype) {
            return i;
        }
    }
    return -1;
}

dtype_t promote_types(dtype_t a, dtype_t b) {
    if (a == b) return a;
    if (a == DTYPE_BOOL) return b;
    if (b == DTYPE_BOOL) return a;
    if (dtype_is_floating(a) && dtype_is_integer(b)) {
        return a;
    }
    if (dtype_is_integer(a) && dtype_is_floating(b)) {
        return b;
    }
    
    int idx_a = type_index(a);
    int idx_b = type_index(b);
    
    if (idx_a == -1 || idx_b == -1) {
        return DTYPE_FLOAT32;
    }
    
    return (idx_a > idx_b) ? a : b;
}

dtype_t promote_types_for_arithmetic(dtype_t a, dtype_t b) {
    dtype_t promoted = promote_types(a, b);
    
    if (dtype_is_integer(a) && dtype_is_integer(b)) {
        int a_signed = dtype_is_signed(a);
        int b_signed = dtype_is_signed(b);
        
        if (a_signed != b_signed) {
            if (a_signed) {
                if (type_index(a) > type_index(DTYPE_INT32)) {
                    return a;
                }
                return DTYPE_INT64;
            } else {
                if (type_index(b) > type_index(DTYPE_INT32)) {
                    return b;
                }
                return DTYPE_INT64;
            }
        }
    }
    
    return promoted;
}

dtype_t promote_types_for_comparison(dtype_t a, dtype_t b) {
    return promote_types(a, b);
}

dtype_t promote_types_for_logical(dtype_t a, dtype_t b) {
    (void)b;
    return DTYPE_BOOL;
}

int can_safely_cast_to(dtype_t from, dtype_t to) {
    if (from == to) return 1;
    if (from == DTYPE_BOOL) return 1;
    
    if (dtype_is_floating(from) && dtype_is_integer(to)) {
        return 0;
    }
    
    if (dtype_is_integer(from) && dtype_is_floating(to)) {
        size_t from_size = dtype_size(from);
        size_t to_size = dtype_size(to);
        
        if (to == DTYPE_FLOAT32 && from_size <= 3) return 1;
        if (to == DTYPE_FLOAT64 && from_size <= 6) return 1;
        return 0;
    }
    
    if (dtype_is_integer(from) && dtype_is_integer(to)) {
        size_t from_size = dtype_size(from);
        size_t to_size = dtype_size(to);
        
        if (dtype_is_signed(from) == dtype_is_signed(to)) {
            return to_size >= from_size;
        }
        
        if (!dtype_is_signed(from) && dtype_is_signed(to)) {
            return to_size > from_size;
        }
        
        return 0;
    }
    
    if (dtype_is_floating(from) && dtype_is_floating(to)) {
        return dtype_size(to) >= dtype_size(from);
    }
    
    return 0;
}