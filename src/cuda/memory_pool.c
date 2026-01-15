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
FreeBlock *free_list = NULL;
AllocatedBlock *allocated_list = NULL;
CachedBlock *cached_list = NULL;
size_t cached_block_count = 0;
pthread_mutex_t central_mutex;
int initialized = 0;

static size_t max_pool_size = 0;
static size_t current_allocated = 0;
static size_t peak_usage = 0;

static void *small_pool_ptr = NULL;
static size_t small_pool_size = 0;
static FreeBlock *small_free_list = NULL;
static const size_t SMALL_POOL_SIZE = 16 * 1024 * 1024;
static const size_t SMALL_BLOCK_THRESHOLD = 1024 * 1024;

static size_t align_size(size_t s);
static int should_use_small_pool(size_t size);
static void *allocate_from_pool(FreeBlock **pool_head, size_t aligned_size, int is_small_pool);
static FreeBlock *find_best_fit_in_pool(FreeBlock *head, size_t aligned_size);
static void remove_block_from_pool(FreeBlock **head, FreeBlock *block);
static void insert_into_pool(FreeBlock **head, FreeBlock *new_block);
static void __allocated_add_block(void *ptr, size_t size);
static AllocatedBlock *__allocated_remove_block(void *ptr);
static CachedBlock *__cache_find_best_fit(size_t aligned_size);
static void __cache_add_block(void *ptr, size_t size);
static void __cache_release_block(void *ptr, size_t size);
static void *expand_pool_if_needed(size_t aligned_size);
static int can_allocate_more(size_t requested_size);

static size_t align_size(size_t s)
{
    return (s + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

static int should_use_small_pool(size_t size)
{
    return (size <= SMALL_BLOCK_THRESHOLD) && (small_pool_ptr != NULL);
}

static int can_allocate_more(size_t requested_size)
{
    return (current_allocated + requested_size <= max_pool_size);
}

int tensor_mem_init(size_t size)
{
    if (initialized)
        return 1;

    if (pthread_mutex_init(&central_mutex, NULL) != 0)
    {
        return 0;
    }

    max_pool_size = align_size(size);
    current_allocated = 0;
    peak_usage = 0;

    cudaError_t err = cudaMalloc(&small_pool_ptr, SMALL_POOL_SIZE);
    if (err == cudaSuccess)
    {
        small_pool_size = SMALL_POOL_SIZE;
        FreeBlock *small_initial = (FreeBlock *)pemalloc(sizeof(FreeBlock), 1);
        if (small_initial)
        {
            small_initial->ptr = small_pool_ptr;
            small_initial->size = SMALL_POOL_SIZE;
            small_initial->next = NULL;
            small_free_list = small_initial;
        }
    }
    else
    {
        return 0;
    }

    initialized = 1;
    return 1;
}

static void *expand_pool_if_needed(size_t aligned_size)
{
    if (base_ptr != NULL)
    {
        return NULL;
    }

    if (!can_allocate_more(aligned_size))
    {
        return NULL;
    }

    size_t initial_pool_size = aligned_size * 4;
    initial_pool_size = align_size(initial_pool_size);

    size_t available = max_pool_size - current_allocated;
    if (initial_pool_size > available)
    {
        initial_pool_size = available;
    }

    if (initial_pool_size < 1024 * 1024)
    {
        return NULL;
    }

    cudaError_t err = cudaMalloc(&base_ptr, initial_pool_size);
    if (err != cudaSuccess)
    {
        base_ptr = NULL;
        return NULL;
    }

    pool_size = initial_pool_size;
    current_allocated += initial_pool_size;

    FreeBlock *initial_block = (FreeBlock *)pemalloc(sizeof(FreeBlock), 1);
    if (!initial_block)
    {
        cudaFree(base_ptr);
        base_ptr = NULL;
        pool_size = 0;
        current_allocated -= initial_pool_size;
        return NULL;
    }

    initial_block->ptr = base_ptr;
    initial_block->size = pool_size;
    initial_block->next = NULL;
    free_list = initial_block;
    return base_ptr;
}

void *tensor_mem_alloc(size_t size)
{
    if (!initialized || size == 0)
        return NULL;

    pthread_mutex_lock(&central_mutex);
    void *ptr = NULL;
    size_t aligned_size = align_size(size);

    if (current_allocated + aligned_size > peak_usage)
    {
        peak_usage = current_allocated + aligned_size;
    }

    if (!can_allocate_more(aligned_size))
    {
        pthread_mutex_unlock(&central_mutex);
        return NULL;
    }

    if (should_use_small_pool(aligned_size))
    {
        ptr = allocate_from_pool(&small_free_list, aligned_size, 1);
    }

    if (!ptr && base_ptr)
    {
        ptr = allocate_from_pool(&free_list, aligned_size, 0);
    }

    if (!ptr && !base_ptr)
    {
        expand_pool_if_needed(aligned_size);
        if (base_ptr)
        {
            ptr = allocate_from_pool(&free_list, aligned_size, 0);
        }
    }

    if (!ptr)
    {
        if (can_allocate_more(aligned_size))
        {
            cudaError_t err = cudaMalloc(&ptr, aligned_size);
            if (err == cudaSuccess)
            {
                __allocated_add_block(ptr, aligned_size);
                current_allocated += aligned_size;
            }
            else
            {
                ptr = NULL;
            }
        }
    }

    pthread_mutex_unlock(&central_mutex);
    return ptr;
}

static void *allocate_from_pool(FreeBlock **pool_head, size_t aligned_size, int is_small_pool)
{
    CachedBlock *cached_block = __cache_find_best_fit(aligned_size);
    if (cached_block)
    {
        void *ptr = cached_block->ptr;
        size_t actual_size = cached_block->size;
        pefree(cached_block, 1);
        __allocated_add_block(ptr, actual_size);
        return ptr;
    }

    FreeBlock *free_block = find_best_fit_in_pool(*pool_head, aligned_size);
    if (!free_block)
        return NULL;

    remove_block_from_pool(pool_head, free_block);

    void *ptr = free_block->ptr;
    size_t actual_size = free_block->size;

    if (!is_small_pool && (actual_size - aligned_size) > ALIGNMENT)
    {
        FreeBlock *remainder = (FreeBlock *)pemalloc(sizeof(FreeBlock), 1);
        if (remainder)
        {
            remainder->ptr = (char *)ptr + aligned_size;
            remainder->size = actual_size - aligned_size;
            insert_into_pool(pool_head, remainder);
            actual_size = aligned_size;
        }
    }

    __allocated_add_block(ptr, actual_size);
    pefree(free_block, 1);
    return ptr;
}

static FreeBlock *find_best_fit_in_pool(FreeBlock *head, size_t aligned_size)
{
    FreeBlock *curr = head;
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

    if (best_fit && best_fit_prev)
    {
        best_fit_prev->next = best_fit->next;
    }
    else if (best_fit)
    {
        FreeBlock **head_ptr = &head;
        *head_ptr = best_fit->next;
    }

    if (best_fit)
    {
        best_fit->next = NULL;
    }

    return best_fit;
}

static void remove_block_from_pool(FreeBlock **head, FreeBlock *block)
{
    FreeBlock *curr = *head;
    FreeBlock *prev = NULL;

    while (curr)
    {
        if (curr == block)
        {
            if (prev)
            {
                prev->next = curr->next;
            }
            else
            {
                *head = curr->next;
            }
            curr->next = NULL;
            return;
        }
        prev = curr;
        curr = curr->next;
    }
}

static void insert_into_pool(FreeBlock **head, FreeBlock *new_block)
{
    FreeBlock **curr_ptr = head;
    while (*curr_ptr && (*curr_ptr)->ptr < new_block->ptr)
    {
        curr_ptr = &(*curr_ptr)->next;
    }
    new_block->next = *curr_ptr;
    *curr_ptr = new_block;
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

static void __cache_release_block(void *ptr, size_t size)
{
    int should_free_to_cuda = 0;

    if (should_use_small_pool(size))
    {
        FreeBlock *fb = (FreeBlock *)pemalloc(sizeof(FreeBlock), 1);
        if (fb)
        {
            fb->ptr = ptr;
            fb->size = size;
            fb->next = NULL;
            insert_into_pool(&small_free_list, fb);
            return;
        }
        else
        {
            should_free_to_cuda = 1;
        }
    }
    else
    {
        FreeBlock *fb = (FreeBlock *)pemalloc(sizeof(FreeBlock), 1);
        if (fb)
        {
            fb->ptr = ptr;
            fb->size = size;
            fb->next = NULL;

            FreeBlock **curr_ptr = &free_list;
            while (*curr_ptr && (*curr_ptr)->ptr < fb->ptr)
            {
                curr_ptr = &(*curr_ptr)->next;
            }
            fb->next = *curr_ptr;
            *curr_ptr = fb;
            return;
        }
        else
        {
            should_free_to_cuda = 1;
        }
    }

    if (should_free_to_cuda)
    {
        cudaFree(ptr);
        current_allocated -= size;
    }
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
        if (cached_block_count < MAX_CACHED_BLOCKS && should_use_small_pool(size_to_free))
        {
            __cache_add_block(ptr, size_to_free);
        }
        else
        {
            __cache_release_block(ptr, size_to_free);
        }

        pefree(ab, 1);
    }

    pthread_mutex_unlock(&central_mutex);
}

void tensor_mem_destroy()
{
    if (!initialized)
        return;

    pthread_mutex_lock(&central_mutex);

    FreeBlock *curr_small = small_free_list;
    while (curr_small)
    {
        FreeBlock *next = curr_small->next;
        pefree(curr_small, 1);
        curr_small = next;
    }
    small_free_list = NULL;

    if (small_pool_ptr)
    {
        cudaFree(small_pool_ptr);
        small_pool_ptr = NULL;
        small_pool_size = 0;
    }

    CachedBlock *curr_cache = cached_list;
    while (curr_cache)
    {
        CachedBlock *next = curr_cache->next;
        cudaFree(curr_cache->ptr);
        pefree(curr_cache, 1);
        curr_cache = next;
    }
    cached_list = NULL;
    cached_block_count = 0;

    AllocatedBlock *curr_alloc = allocated_list;
    while (curr_alloc)
    {
        AllocatedBlock *next = curr_alloc->next;
        cudaFree(curr_alloc->ptr);
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
        base_ptr = NULL;
    }

    pool_size = 0;
    current_allocated = 0;
    max_pool_size = 0;
    peak_usage = 0;

    pthread_mutex_unlock(&central_mutex);
    pthread_mutex_destroy(&central_mutex);
    initialized = 0;
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