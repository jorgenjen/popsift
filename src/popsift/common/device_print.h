#pragma once

#include "popsift/common/assist.h"

#include <cuda_runtime.h>

namespace popsift {
namespace print {

__global__ static void printSurface(
  cudaSurfaceObject_t obj, const char* identifier, int start_x, int end_x, int start_y, int end_y, int level)
{
    printf("\n%s --> SURFACE PRINT OUT: region y(%d -> %d) x(%d -> %d)\n", identifier, start_y, end_y, start_x, end_x);
    float val = {};
    for(int y = start_y; y < end_y; ++y)
    {
        for(int x = start_x; x < end_x; ++x)
        {
            surf2DLayeredread(&val, obj, x * 4, y, level, cudaBoundaryModeZero);
            printf("%10.6f ", val);
        }
        printf("\n");
    }
    printf("\n\n");
}

__global__ static void printTexture(
  cudaSurfaceObject_t obj, const char* identifier, int start_x, int end_x, int start_y, int end_y, int level)
{
    printf("\n%s --> TEXTURE PRINT OUT: region y(%d -> %d) x(%d -> %d)\n", identifier, start_y, end_y, start_x, end_x);
    float val = {};
    for(int y = start_y; y < end_y; ++y)
    {
        for(int x = start_x; x < end_x; ++x)
        {
            val = popsift::readTex(obj, x, y, level);
            printf("%10.6f ", val);
        }
        printf("\n");
    }
    printf("\n\n");
}

template<bool is_tex, typename T>
void print_region(T obj, const char* identifier, int start_x, int end_x, int start_y, int end_y, int level)
{
    size_t str_len = strlen(identifier) + 1;
    char* dev_msg;
    cudaMalloc(&dev_msg, str_len);
    cudaMemcpy(dev_msg, identifier, str_len, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    // if constexpr(std::is_same_v<T, cudaSurfaceObject_t>) // did not work (using needs c++17 +)
    if(is_tex)
    {
        fprintf(stderr, "NOT SURE ABOUT THIS TEXTURE VERSION\n");
        printTexture<<<1, 1>>>(obj, dev_msg, start_x, end_x, start_y, end_y, level);
    }
    else
    {
        printSurface<<<1, 1>>>(obj, dev_msg, start_x, end_x, start_y, end_y, level);
    }
    cudaDeviceSynchronize();

    cudaFree(dev_msg);
}
}

}
