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

static size_t max_pool_size = 0;
static size_t current_allocated = 0;
static size_t peak_usage = 0;

static void *small_pool_ptr = NULL;
static size_t small_pool_size = 0;
static FreeBlock *small_free_list = NULL;
static const size_t SMALL_POOL_CAPACITY = 16 * 1024 * 1024; // 16MB fixos para metadados

// --- Protótipos de Funções Internas ---
static size_t align_size(size_t s);
static int should_use_small_pool(size_t size);
static void *allocate_from_pool_logic(FreeBlock **pool_head, size_t aligned_size, int is_small_pool);
static void insert_into_pool_and_merge(FreeBlock **head, FreeBlock *new_block);
static void coalesce_free_list(FreeBlock **head);
static void __allocated_add_block(void *ptr, size_t size);
static AllocatedBlock *__allocated_remove_block(void *ptr);
static CachedBlock *__cache_find_best_fit(size_t aligned_size);
static void __cache_add_block(void *ptr, size_t size);
static void __cache_release_block(void *ptr, size_t size);
static void *expand_pool_if_needed(size_t aligned_size);
static int can_allocate_more(size_t requested_size);

static size_t align_size(size_t s) {
    return (s + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

static int should_use_small_pool(size_t size) {
    return (size <= SMALL_BLOCK_THRESHOLD) && (small_pool_ptr != NULL);
}

static int can_allocate_more(size_t requested_size) {
    return (current_allocated + requested_size <= max_pool_size);
}

static void coalesce_free_list(FreeBlock **head) {
    if (!head || !*head) return;
    FreeBlock *curr = *head;
    while (curr && curr->next) {
        if ((char *)curr->ptr + curr->size == (char *)curr->next->ptr) {
            FreeBlock *temp = curr->next;
            curr->size += temp->size;
            curr->next = temp->next;
            pefree(temp, 1);
        } else {
            curr = curr->next;
        }
    }
}

static void insert_into_pool_and_merge(FreeBlock **head, FreeBlock *new_block) {
    FreeBlock **curr_ptr = head;
    while (*curr_ptr && (*curr_ptr)->ptr < new_block->ptr) {
        curr_ptr = &(*curr_ptr)->next;
    }
    new_block->next = *curr_ptr;
    *curr_ptr = new_block;
    coalesce_free_list(head);
}

int tensor_mem_init(size_t size) {
    if (initialized) return 1;

    if (pthread_mutex_init(&central_mutex, NULL) != 0) return 0;

    max_pool_size = align_size(size);
    current_allocated = 0;
    peak_usage = 0;

    cudaError_t err = cudaMalloc(&small_pool_ptr, SMALL_POOL_CAPACITY);
    if (err == cudaSuccess) {
        small_pool_size = SMALL_POOL_CAPACITY;
        FreeBlock *small_initial = (FreeBlock *)pemalloc(sizeof(FreeBlock), 1);
        if (small_initial) {
            small_initial->ptr = small_pool_ptr;
            small_initial->size = SMALL_POOL_CAPACITY;
            small_initial->next = NULL;
            small_free_list = small_initial;
        }
    } else {
        return 0;
    }

    initialized = 1;
    return 1;
}

static void *expand_pool_if_needed(size_t aligned_size) {
    if (base_ptr != NULL || !can_allocate_more(aligned_size)) return NULL;

    size_t initial_pool_size = align_size(aligned_size * 4);
    size_t available = max_pool_size - current_allocated;
    if (initial_pool_size > available) initial_pool_size = available;

    if (initial_pool_size < 1024 * 1024) return NULL;

    cudaError_t err = cudaMalloc(&base_ptr, initial_pool_size);
    if (err != cudaSuccess) return NULL;

    pool_size = initial_pool_size;
    current_allocated += initial_pool_size;

    FreeBlock *initial_block = (FreeBlock *)pemalloc(sizeof(FreeBlock), 1);
    initial_block->ptr = base_ptr;
    initial_block->size = pool_size;
    initial_block->next = NULL;
    free_list = initial_block;
    
    return base_ptr;
}

void *cuda_mem_alloc(size_t size) {
    if (!initialized || size == 0) return NULL;

    pthread_mutex_lock(&central_mutex);
    void *ptr = NULL;
    size_t aligned_size = align_size(size);

    if (current_allocated + aligned_size > peak_usage) {
        peak_usage = current_allocated + aligned_size;
    }

    CachedBlock *cached_block = __cache_find_best_fit(aligned_size);
    if (cached_block) {
        ptr = cached_block->ptr;
        size_t actual_size = cached_block->size;
        pefree(cached_block, 1);
        __allocated_add_block(ptr, actual_size);
        pthread_mutex_unlock(&central_mutex);
        return ptr;
    }

    if (should_use_small_pool(aligned_size)) {
        ptr = allocate_from_pool_logic(&small_free_list, aligned_size, 1);
    }

    if (!ptr) {
        if (!base_ptr) expand_pool_if_needed(aligned_size);
        if (base_ptr) ptr = allocate_from_pool_logic(&free_list, aligned_size, 0);
    }

    if (!ptr && can_allocate_more(aligned_size)) {
        cudaError_t err = cudaMalloc(&ptr, aligned_size);
        if (err == cudaSuccess) {
            __allocated_add_block(ptr, aligned_size);
            current_allocated += aligned_size;
        }
    }

    pthread_mutex_unlock(&central_mutex);
    return ptr;
}

static void *allocate_from_pool_logic(FreeBlock **pool_head, size_t aligned_size, int is_small_pool) {
    FreeBlock *curr = *pool_head;
    FreeBlock *prev = NULL;
    FreeBlock *best_fit = NULL;
    FreeBlock *best_fit_prev = NULL;
    size_t min_diff = (size_t)-1;

    while (curr) {
        if (curr->size >= aligned_size) {
            size_t diff = curr->size - aligned_size;
            if (diff < min_diff) {
                min_diff = diff;
                best_fit = curr;
                best_fit_prev = prev;
                if (diff == 0) break;
            }
        }
        prev = curr;
        curr = curr->next;
    }

    if (!best_fit) return NULL;

    void *ptr = best_fit->ptr;
    size_t actual_size = best_fit->size;

    if (actual_size > aligned_size + ALIGNMENT) {
        best_fit->ptr = (char *)ptr + aligned_size;
        best_fit->size -= aligned_size;
        actual_size = aligned_size;
    } else {
        if (best_fit_prev) best_fit_prev->next = best_fit->next;
        else *pool_head = best_fit->next;
        pefree(best_fit, 1);
    }

    __allocated_add_block(ptr, actual_size);
    return ptr;
}

void cuda_mem_free(void *ptr) {
    if (!initialized || !ptr) return;

    pthread_mutex_lock(&central_mutex);

    AllocatedBlock *ab = __allocated_remove_block(ptr);
    if (ab) {
        size_t size_to_free = ab->size;

        int is_in_small = (small_pool_ptr && (char *)ptr >= (char *)small_pool_ptr && (char *)ptr < (char *)small_pool_ptr + SMALL_POOL_CAPACITY);
        int is_in_main = (base_ptr && (char *)ptr >= (char *)base_ptr && (char *)ptr < (char *)base_ptr + pool_size);

        if (is_in_small) {
            FreeBlock *fb = (FreeBlock *)pemalloc(sizeof(FreeBlock), 1);
            fb->ptr = ptr; fb->size = size_to_free;
            insert_into_pool_and_merge(&small_free_list, fb);
        } 
        else if (is_in_main) {
            FreeBlock *fb = (FreeBlock *)pemalloc(sizeof(FreeBlock), 1);
            fb->ptr = ptr; fb->size = size_to_free;
            insert_into_pool_and_merge(&free_list, fb);
        }
        else {
            if (cached_block_count < MAX_CACHED_BLOCKS) {
                __cache_add_block(ptr, size_to_free);
            } else {
                __cache_release_block(ptr, size_to_free);
            }
        }
        pefree(ab, 1);
    }

    pthread_mutex_unlock(&central_mutex);
}

void tensor_mem_destroy() {
    if (!initialized) return;

    pthread_mutex_lock(&central_mutex);

    FreeBlock *curr_small = small_free_list;
    while (curr_small) {
        FreeBlock *next = curr_small->next;
        pefree(curr_small, 1);
        curr_small = next;
    }
    if (small_pool_ptr) cudaFree(small_pool_ptr);

    CachedBlock *curr_cache = cached_list;
    while (curr_cache) {
        CachedBlock *next = curr_cache->next;
        cudaFree(curr_cache->ptr);
        pefree(curr_cache, 1);
        curr_cache = next;
    }

    AllocatedBlock *curr_alloc = allocated_list;
    while (curr_alloc) {
        AllocatedBlock *next = curr_alloc->next;
        cudaFree(curr_alloc->ptr);
        pefree(curr_alloc, 1);
        curr_alloc = next;
    }

    FreeBlock *curr_free = free_list;
    while (curr_free) {
        FreeBlock *next = curr_free->next;
        pefree(curr_free, 1);
        curr_free = next;
    }

    if (base_ptr) cudaFree(base_ptr);

    small_free_list = NULL;
    cached_list = NULL;
    allocated_list = NULL;
    free_list = NULL;
    base_ptr = NULL;
    small_pool_ptr = NULL;

    initialized = 0;
    pthread_mutex_unlock(&central_mutex);
    pthread_mutex_destroy(&central_mutex);
}

static void __allocated_add_block(void *ptr, size_t size) {
    AllocatedBlock *ab = (AllocatedBlock *)pemalloc(sizeof(AllocatedBlock), 1);
    if (!ab) return;
    ab->ptr = ptr; ab->size = size;
    ab->next = allocated_list;
    allocated_list = ab;
}

static AllocatedBlock *__allocated_remove_block(void *ptr) {
    AllocatedBlock *curr = allocated_list, *prev = NULL;
    while (curr) {
        if (curr->ptr == ptr) {
            if (prev) prev->next = curr->next;
            else allocated_list = curr->next;
            return curr;
        }
        prev = curr; curr = curr->next;
    }
    return NULL;
}

static CachedBlock *__cache_find_best_fit(size_t aligned_size) {
    CachedBlock *curr = cached_list, *prev = NULL, *best = NULL, *best_prev = NULL;
    size_t min_diff = (size_t)-1;
    while (curr) {
        if (curr->size >= aligned_size) {
            size_t diff = curr->size - aligned_size;
            if (diff < min_diff) {
                min_diff = diff; best = curr; best_prev = prev;
            }
        }
        prev = curr; curr = curr->next;
    }
    if (best) {
        if (best_prev) best_prev->next = best->next;
        else cached_list = best->next;
        cached_block_count--;
        return best;
    }
    return NULL;
}

static void __cache_add_block(void *ptr, size_t size) {
    CachedBlock *cb = (CachedBlock *)pemalloc(sizeof(CachedBlock), 1);
    if (!cb) { __cache_release_block(ptr, size); return; }
    cb->ptr = ptr; cb->size = size;
    cb->next = cached_list;
    cached_list = cb;
    cached_block_count++;
}

static void __cache_release_block(void *ptr, size_t size) {
    cudaFree(ptr);
    current_allocated -= size;
}