/*
 * Copyright 2016-2017, Simula Research Laboratory
 *           2018-2024, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/plane_2d.h"
#include "gauss_filter.h"
#include "sift_constants.h"
#include "sift_pyramid.h"

using std::cout;
using std::endl;

namespace popsift {
namespace normalizedSource {

__global__ static void horiz(
  cudaTextureObject_t src_linear_tex, cudaSurfaceObject_t dst_data, int dst_w, int dst_h, float shift)
{
    // Create octave-0 - level-0 from the input image.
    const int write_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int write_y = blockIdx.y;

    if(write_x >= dst_w)
        return;

    const int span = d_gauss.dd.span[0];
    const float* filter = &d_gauss.dd.filter[0];
    const float read_x = (blockIdx.x * blockDim.x + threadIdx.x + shift) / dst_w;
    const float read_y = (blockIdx.y + shift) / dst_h;

    float out = 0.0f;

    // if(read_x == 0.5)
    //     printf("\n\n\t\t\tis zero!!!");

    if(write_x == 5 && write_y == 5)
    {
        printf("\n\t\t\tread (%f, %f)\n", read_x, read_y);

        float u = (write_x + shift) / dst_w;
        float v = (write_y + shift) / dst_h;
        printf("\n\t\t\tread (%f, %f)\n", u, v);
    }

    if(write_y == 0 && write_x == 1)
    {
        printf("dst_w: %d -- dst_h: %d", dst_w, dst_h);
        printf("span: %d\n", span);
        printf("filter: %f %f %f %f %f %f\n", filter[6], filter[5], filter[4], filter[3], filter[2], filter[1]);
        printf("filter:");

        for(int offset = span; offset > 0; offset--)
        {
            printf(" %f ", filter[offset]);
        }
        printf("\nread_x: %f\n", read_x);
        printf("read_y: %f\n", read_y);

        // Prinout OG image -- middle of texels
        printf("\n\nPRE SCALE UP -- SMALL OG IMGAGE:\n");
        // for(int y = 0; y < 10; ++y)
        // {
        //     for(int x = 0; x < 10; ++x)
        //     {
        for(int y = 426 - 13; y < 426; ++y)
        {
            for(int x = 640 - 13; x < 640; ++x)
            {
                // Normalize coordinates to [0, 1) range
                // add 0.5 to get middle of texel
                float u = (x + 0.5) / 640.0f;
                float v = (y + 0.5) / 426.0f;

                // float u = (x) / 640.0f;
                // float v = (y) / 640.0f;
                // Sample the texture
                float pixel = tex2D<float>(src_linear_tex, u, v);

                // Print the pixel value
                printf("%06.2f ", pixel * 255.0f);
            }
            printf("\n");
        }

        printf("\n");
        printf("\n");
        printf("\n");
        // printout scaled image as done in popsift
        // for(int y = 0; y < 12; ++y)
        // {
        //     for(int x = 0; x < 12; ++x)
        //     {

        printf("SCALED UP: dst_h = %d --- dst_w=%d\n", dst_h, dst_w);
        for(int y = dst_h - 13; y < dst_h; ++y)
        {
            for(int x = dst_w - 13; x < dst_w; ++x)
            {
                // Normalize coordinates to [0, 1) range
                float u = (x + shift) / dst_w;
                float v = (y + shift) / dst_h;

                // float u = (x) / 640.0f;
                // float v = (y) / 640.0f;
                // Sample the texture
                float pixel = tex2D<float>(src_linear_tex, u, v);

                // Print the pixel value
                printf("%06.2f ", pixel * 255.0f);
            }
            printf("\n");
        }

        // for(int i = 0; i >
        // for(int y = 0; y < 5; ++y)
        // {
        //     for(int x = 0; x < 5; ++x)
        //     {
        //         printf("\t\tValue at %d %d: %f\n", x, y, tex2D<float>(src_linear_tex, x, y));
        //     }
        // }
        printf("\n");
        printf("\n");
    }

#pragma unroll
    for(int offset = span; offset > 0; offset--)
    {
        const float& g = filter[offset];
        const float offrel = float(offset) / dst_w;
        const float v1 = tex2D<float>(src_linear_tex, read_x - offrel, read_y);
        const float v2 = tex2D<float>(src_linear_tex, read_x + offrel, read_y);
        out += ((v1 + v2) * g);

        if(write_x == 0 && write_y == 0)
        {
            printf("offset: %d v1=%f v2=%f\n", offset, v1, v2);
        }
    }
    const float& g = filter[0];
    const float v3 = tex2D<float>(src_linear_tex, read_x, read_y);
    out += (v3 * g);

    // x is multiplied by 4 as it is byte coordinates I believe and float is 4 bytes obv
    // This one writes to level 0 so z -level 0 -- other kernels wiill write to different
    // parts of the texture in terms of z level
    surf2DLayeredwrite(out * 255.0f, dst_data, write_x * 4, write_y, 0, cudaBoundaryModeZero);
}

} // namespace normalizedSource

__global__ static void printIntermediate(cudaSurfaceObject_t intermediate, int width, int height)
{
    printf("\nPrint intermediate\n");
    float val = {};
    for(int y = height - 8; y < height; ++y)
    {
        for(int x = width - 8; x < width; ++x)
        {
            // printf("\t\tValue at %d %d: %f\n", x, y, tex2D<float>(src_linear_tex, x, y));
            surf2DLayeredread(&val, intermediate, x * 4, y, 0, cudaBoundaryModeZero);
            printf("%10.6f ", val);
        }
        printf("\n");
    }
    printf("\n\n");
}

__host__ void Pyramid::horiz_from_input_image(const Config& conf, ImageBase* base, cudaStream_t stream)
{
    Octave& oct_obj = _octaves[0];

    const int width = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    dim3 block(128, 1);
    dim3 grid;
    grid.x = grid_divide(width, 128);
    grid.y = height;

    float shift = 0.5f * powf(2.0f, conf.getUpscaleFactor());

    cout << "\n\nHoriz_from_input_image" << endl;
    cout << "Grid x: " << grid.x << " y: " << grid.y << endl;
    cout << "Shift: " << shift << endl;
    cout << "Widht: " << width << " Height: " << height << endl;
    cout << endl << endl;

    normalizedSource::horiz<<<grid, block, 0, stream>>>(
      base->getInputTexture(), oct_obj.getIntermediateSurface(), width, height, shift);

    cudaDeviceSynchronize();
    printIntermediate<<<1, 1>>>(oct_obj.getIntermediateSurface(), width, height);

    POP_SYNC_CHK;
}

} // namespace popsift
