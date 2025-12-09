
#ifndef KERNEL_TYPES_H
#define KERNEL_TYPES_H

#include "php.h"
#include "zend_compile.h"
#include "data_types.h"

typedef enum
{
    INPUT,
    OUTPUT,
    PARAMETER
} parameter_type_t;

typedef struct
{
    char name[32];
    dtype_t dtype;
    parameter_type_t type;
} func_parameter;

typedef struct
{
    int total;
    func_parameter **parameters;
} func_parameter_list_t;


#define KERNEL_PLIST_FIND(list, name, var)                              \
    do                                                                  \
    {                                                                   \
        (var) = NULL;                                                   \
        if ((list) && (list)->parameters)                               \
        {                                                               \
            for (int __i = 0; __i < (list)->total; __i++)               \
            {                                                           \
                if (strcmp((list)->parameters[__i]->name, (name)) == 0) \
                {                                                       \
                    (var) = (list)->parameters[__i];                    \
                    break;                                              \
                }                                                       \
            }                                                           \
        }                                                               \
    } while (0)

/**
 * @param dtype1
 * @param dtype2
 */
#define KERNEL_DTYPE_EQUAL(dtype1, dtype2) \
    ((dtype1) == (dtype2))

/**
 * @param param
 */
#define KERNEL_GET_DTYPE(param) \
    ((param)->dtype)

/**
 * @param param
 */
#define KERNEL_GET_TYPE(param) \
    ((param)->type)

#endif