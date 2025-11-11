#ifndef CONDITIONAL_H
#define CONDITIONAL_H

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum {
    OP_GREATER,
    OP_LESS,
    OP_EQUAL,
    OP_NOT_EQUAL,
    OP_GREATER_EQUAL,
    OP_LESS_EQUAL
} condition_op_t;

typedef enum {
    LOGICAL_AND,
    LOGICAL_OR
} logical_op_t;

typedef struct {
    condition_op_t op;
    logical_op_t logical;
    float value;
} condition_t;

typedef struct {
    condition_t *conditions;
    int count;
} condition_config_t;

#ifdef __cplusplus
}
#endif

#endif