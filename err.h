#ifndef __ERROR_H__
#define __ERROR_H__

#include <stdio.h>

#define CUDA_CHECK_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

#define CUDA_CHECK_LAST_ERROR() \
    { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    }

#endif // __ERROR_H__ 