#include "php_helpers.h"

#ifdef __cplusplus
extern "C"
{
#endif

    void extract_shape_from_array(zval *data, int *shape, int *ndims)
    {
        *ndims = 0;

        void extract_shape_recursive(zval * arr, int current_depth)
        {
            if (Z_TYPE_P(arr) != IS_ARRAY)
                return;
            if (current_depth >= 10)
                return;

            HashTable *arr_ht = Z_ARRVAL_P(arr);
            int count = zend_array_count(arr_ht);

            if (count == 0)
                return;

            shape[current_depth] = count;
            if (current_depth >= *ndims)
            {
                *ndims = current_depth + 1;
            }

            if (count > 0)
            {
                zval *first = zend_hash_index_find(arr_ht, 0);
                if (first != NULL)
                {
                    extract_shape_recursive(first, current_depth + 1);
                }
            }
        }

        extract_shape_recursive(data, 0);
    }

    void flatten_php_array(zval *data, float *flat_array, int *index)
    {
        if (Z_TYPE_P(data) != IS_ARRAY)
        {
            if (Z_TYPE_P(data) == IS_LONG)
            {
                flat_array[(*index)++] = (float)Z_LVAL_P(data);
            }
            else if (Z_TYPE_P(data) == IS_DOUBLE)
            {
                flat_array[(*index)++] = (float)Z_DVAL_P(data);
            }
            else if (Z_TYPE_P(data) == IS_TRUE)
            {
                flat_array[(*index)++] = 1.0f;
            }
            else if (Z_TYPE_P(data) == IS_FALSE)
            {
                flat_array[(*index)++] = 0.0f;
            }
            return;
        }

        HashTable *ht = Z_ARRVAL_P(data);
        zval *current;
        ZEND_HASH_FOREACH_VAL(ht, current)
        {
            flatten_php_array(current, flat_array, index);
        }
        ZEND_HASH_FOREACH_END();
    }

    size_t calculate_total_size(zval *data)
    {
        if (Z_TYPE_P(data) != IS_ARRAY)
        {
            return 1;
        }

        size_t total = 1;
        HashTable *ht = Z_ARRVAL_P(data);
        zval *first = zend_hash_index_find(ht, 0);

        if (first != NULL)
        {
            total = zend_array_count(ht) * calculate_total_size(first);
        }

        return total;
    }

    int parse_slice_parameter(zval *param, slice_info_t *slice)
    {
        memset(slice, 0, sizeof(slice_info_t));

        if (Z_TYPE_P(param) == IS_NULL)
        {
            slice->type = SLICE_ALL;
            return 1;
        }

        if (Z_TYPE_P(param) == IS_LONG)
        {
            slice->type = SLICE_INDEX;
            slice->data.index = Z_LVAL_P(param);
            return 1;
        }

        if (Z_TYPE_P(param) == IS_ARRAY)
        {
            HashTable *ht = Z_ARRVAL_P(param);
            if (zend_array_count(ht) == 2)
            {
                zval *start_val = zend_hash_index_find(ht, 0);
                zval *end_val = zend_hash_index_find(ht, 1);

                if (start_val && end_val &&
                    Z_TYPE_P(start_val) == IS_LONG &&
                    Z_TYPE_P(end_val) == IS_LONG)
                {

                    slice->type = SLICE_RANGE;
                    slice->data.range.start = Z_LVAL_P(start_val);
                    slice->data.range.end = Z_LVAL_P(end_val);
                    return 1;
                }
            }
        }

        return 0;
    }

#ifdef __cplusplus
    extern "C"
}
#endif