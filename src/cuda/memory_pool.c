#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <math.h>
#include "php.h"
#include "memory_pool.h"

static void *base_ptr = NULL;
static size_t pool_size = 0;
static FreeBlock *free_list = NULL;
static AllocatedBlock *allocated_list = NULL;

static CachedBlock *cached_list = NULL;
static size_t cached_block_count = 0;

static pthread_mutex_t central_mutex;
static int initialized = 0;

#define ALIGNMENT 512

static size_t align_size(size_t s)
{
    return (s + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

static FreeBlock *__bfc_find_best_fit(size_t aligned_size)
{
    FreeBlock *curr = free_list;
    FreeBlock *prev = NULL;
    FreeBlock *best_fit = NULL;
    FreeBlock *best_fit_prev = NULL;
    size_t min_diff = (size_t)-1;

    while (curr)
    {
        if (curr->size >= aligned_size)
        {
            size_t diff = curr->size - aligned_size;

            if (diff < min_diff)
            {
                min_diff = diff;
                best_fit = curr;
                best_fit_prev = prev;
                if (diff == 0)
                    break;
            }
        }
        prev = curr;
        curr = curr->next;
    }

    if (best_fit)
    {
        if (best_fit_prev)
        {
            best_fit_prev->next = best_fit->next;
        }
        else
        {
            free_list = best_fit->next;
        }
        best_fit->next = NULL;
    }
    return best_fit;
}

static void __bfc_insert_and_coalesce_free_block(FreeBlock *new_block)
{
    FreeBlock **curr_ptr = &free_list;

    while (*curr_ptr && (*curr_ptr)->ptr < new_block->ptr)
    {
        curr_ptr = &(*curr_ptr)->next;
    }

    new_block->next = *curr_ptr;
    *curr_ptr = new_block;

    FreeBlock *prev_block = NULL;
    if (curr_ptr != &free_list)
    {
        FreeBlock **scan_ptr = &free_list;
        while (*scan_ptr != new_block)
        {
            prev_block = *scan_ptr;
            scan_ptr = &(*scan_ptr)->next;
        }
    }

    if (prev_block && (char *)prev_block->ptr + prev_block->size == (char *)new_block->ptr)
    {
        prev_block->size += new_block->size;
        prev_block->next = new_block->next;
        pefree(new_block, 1);
        new_block = prev_block;
    }

    FreeBlock *next_block = new_block->next;
    if (next_block && (char *)new_block->ptr + new_block->size == (char *)next_block->ptr)
    {
        new_block->size += next_block->size;
        new_block->next = next_block->next;
        pefree(next_block, 1);
    }
}

static void __allocated_add_block(void *ptr, size_t size)
{
    AllocatedBlock *ab = (AllocatedBlock *)pemalloc(sizeof(AllocatedBlock), 1);
    if (!ab)
        return;
    ab->ptr = ptr;
    ab->size = size;
    ab->next = allocated_list;
    allocated_list = ab;
}

static AllocatedBlock *__allocated_remove_block(void *ptr)
{
    AllocatedBlock *curr = allocated_list;
    AllocatedBlock *prev = NULL;

    while (curr)
    {
        if (curr->ptr == ptr)
        {
            if (prev)
            {
                prev->next = curr->next;
            }
            else
            {
                allocated_list = curr->next;
            }
            curr->next = NULL;
            return curr;
        }
        prev = curr;
        curr = curr->next;
    }
    return NULL;
}

static CachedBlock *__cache_find_best_fit(size_t aligned_size)
{
    CachedBlock *curr = cached_list;
    CachedBlock *prev = NULL;
    CachedBlock *best_fit = NULL;
    CachedBlock *best_fit_prev = NULL;

    size_t min_diff = (size_t)-1;

    while (curr)
    {
        if (curr->size >= aligned_size)
        {
            size_t diff = curr->size - aligned_size;

            if (diff < min_diff)
            {
                min_diff = diff;
                best_fit = curr;
                best_fit_prev = prev;
            }
        }
        prev = curr;
        curr = curr->next;
    }

    if (best_fit)
    {
        if (best_fit_prev)
        {
            best_fit_prev->next = best_fit->next;
        }
        else
        {
            cached_list = best_fit->next;
        }
        cached_block_count--;
        return best_fit;
    }
    return NULL;
}

static void __cache_release_block(void *ptr, size_t size)
{
    FreeBlock *fb = (FreeBlock *)pemalloc(sizeof(FreeBlock), 1);
    if (!fb)
    {
        cudaFree(ptr);
        php_error_docref(NULL, E_WARNING,
                         "VRAM Manager: Failed to track free block (Host memory exhaustion). GPU block released to CUDA.");
        return;
    }
    fb->ptr = ptr;
    fb->size = size;
    fb->next = NULL;

    __bfc_insert_and_coalesce_free_block(fb);
}

static void __cache_add_block(void *ptr, size_t size)
{
    if (cached_block_count >= MAX_CACHED_BLOCKS)
    {
        __cache_release_block(ptr, size);
        return;
    }

    CachedBlock *cb = (CachedBlock *)pemalloc(sizeof(CachedBlock), 1);
    if (!cb)
    {
        __cache_release_block(ptr, size);
        return;
    }

    cb->ptr = ptr;
    cb->size = size;

    cb->next = cached_list;
    cached_list = cb;
    cached_block_count++;
}

int tensor_mem_init(size_t size)
{
    if (initialized)
        return 1;

    if (pthread_mutex_init(&central_mutex, NULL) != 0)
    {
        return 0;
    }

    size_t aligned_size = align_size(size);

    cudaError_t err = cudaMalloc(&base_ptr, aligned_size);
    if (err != cudaSuccess)
    {
        pthread_mutex_destroy(&central_mutex);
        return 0;
    }

    pool_size = aligned_size;

    FreeBlock *initial_block = (FreeBlock *)pemalloc(sizeof(FreeBlock), 1);
    if (!initial_block)
    {
        cudaFree(base_ptr);
        pthread_mutex_destroy(&central_mutex);
        return 0;
    }

    initial_block->ptr = base_ptr;
    initial_block->size = pool_size;
    initial_block->next = NULL;
    free_list = initial_block;

    initialized = 1;
    return 1;
}

void *tensor_mem_alloc(size_t size)
{
    if (!initialized || size == 0)
        return NULL;

    pthread_mutex_lock(&central_mutex);

    size_t aligned_size = align_size(size);
    void *ptr = NULL;

    CachedBlock *cached_block = __cache_find_best_fit(aligned_size);

    if (cached_block)
    {
        ptr = cached_block->ptr;
        size_t actual_size = cached_block->size;
        pefree(cached_block, 1);

        __allocated_add_block(ptr, actual_size);
    }
    else
    {
        FreeBlock *free_block = __bfc_find_best_fit(aligned_size);

        if (free_block)
        {
            ptr = free_block->ptr;
            size_t actual_size = free_block->size;

            size_t remainder_size = actual_size - aligned_size;

            if (remainder_size > ALIGNMENT)
            {
                FreeBlock *remainder_block = (FreeBlock *)pemalloc(sizeof(FreeBlock), 1);
                if (remainder_block)
                {
                    remainder_block->ptr = (char *)ptr + aligned_size;
                    remainder_block->size = remainder_size;
                    remainder_block->next = NULL;
                    __bfc_insert_and_coalesce_free_block(remainder_block);
                }
                else
                {
                    __bfc_insert_and_coalesce_free_block(free_block);
                    ptr = NULL;
                    actual_size = 0;
                    free_block = NULL;
                }

                actual_size = aligned_size;
            }

            if (ptr)
            {
                __allocated_add_block(ptr, actual_size);
                if (free_block)
                    pefree(free_block, 1);
            }
        }
    }

    pthread_mutex_unlock(&central_mutex);
    return ptr;
}

void tensor_mem_free(void *ptr)
{
    if (!initialized || !ptr)
        return;

    pthread_mutex_lock(&central_mutex);

    AllocatedBlock *ab = __allocated_remove_block(ptr);

    if (ab)
    {
        size_t size_to_free = ab->size;
        pefree(ab, 1);

        __cache_add_block(ptr, size_to_free);
    }

    pthread_mutex_unlock(&central_mutex);
}

void tensor_mem_destroy()
{
    if (!initialized)
        return;

    pthread_mutex_lock(&central_mutex);

    CachedBlock *curr_cache = cached_list;
    while (curr_cache)
    {
        CachedBlock *next = curr_cache->next;
        __cache_release_block(curr_cache->ptr, curr_cache->size);
        pefree(curr_cache, 1);
        curr_cache = next;
    }
    cached_list = NULL;
    cached_block_count = 0;

    AllocatedBlock *curr_alloc = allocated_list;
    while (curr_alloc)
    {
        AllocatedBlock *next = curr_alloc->next;
        pefree(curr_alloc, 1);
        curr_alloc = next;
    }
    allocated_list = NULL;

    FreeBlock *curr_free = free_list;
    while (curr_free)
    {
        FreeBlock *next = curr_free->next;
        pefree(curr_free, 1);
        curr_free = next;
    }
    free_list = NULL;

    if (base_ptr)
    {
        cudaFree(base_ptr);
    }
    base_ptr = NULL;
    pool_size = 0;

    pthread_mutex_unlock(&central_mutex);
    pthread_mutex_destroy(&central_mutex);
    initialized = 0;
}
