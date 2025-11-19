#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#define MAX_CACHED_BLOCKS 100

typedef struct AllocatedBlock {
    void* ptr;
    size_t size; 
    struct AllocatedBlock* next;
} AllocatedBlock;

typedef struct FreeBlock {
    void* ptr;
    size_t size;
    struct FreeBlock* next;
} FreeBlock;

typedef struct CachedBlock {
    void* ptr;
    size_t size;       
    struct CachedBlock* next;
} CachedBlock;

int tensor_mem_init(size_t size);
void *tensor_mem_alloc(size_t size);
void tensor_mem_free(void *ptr);
void tensor_mem_destroy();

#endif