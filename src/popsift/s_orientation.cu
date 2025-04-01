/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/debug_macros.h"
#include "common/excl_blk_prefix_sum.h"
#include "common/warp_bitonic_sort.h"
#include "s_gradiant.h"
#include "sift_config.h"
#include "sift_constants.h"
#include "sift_pyramid.h"

// #include <__clang_cuda_builtin_vars.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>

using namespace popsift;
using namespace std;

/* Smoothing like VLFeat is the default mode.
 * If you choose to undefine it, you get the smoothing approach taken by OpenCV
 */
#define WITH_VLFEAT_SMOOTHING

namespace popsift {

__device__ inline float compute_angle(int bin, float hc, float hn, float hp)
{
    /* interpolate */
    float di = bin + 0.5f * (hn - hp) / (hc + hc - hn - hp);

    /* clamp */
    di = (di < 0) ? (di + ORI_NBINS) : ((di >= ORI_NBINS) ? (di - ORI_NBINS) : (di));

    float th = __fdividef(M_PI2 * di, ORI_NBINS) - M_PI;
    // float th = ((M_PI2 * di) / ORI_NBINS);
    return th;
}

/*
 * Histogram smoothing helper
 */
template<int D>
__device__ inline static float smoothe(const float* const src, const int bin)
{
    const int prev = (bin == 0) ? ORI_NBINS - 1 : bin - 1;
    const int next = (bin == ORI_NBINS - 1) ? 0 : bin + 1;

    const float f = (src[prev] + src[bin] + src[next]) / 3.0f;

    return f;
}

/*
 * Compute the keypoint orientations for each extremum
 * using 16 threads for each of them.
 * direct curve fitting approach
 */
// I think using 2 warps per ori_par could potentially be beneficial with modern GPU's with high SM count
// The 980ti had 22 so could have 1408 warps but 4070 ti has 60SM and hence can have 3840 so I think it might be
// beneficial to use two warps per keypoint to have better chance of fully saturating the GPU for this step (depends on
// number of extrems in image/frame) could be interesting to test probs minimal differences however
__global__ void ori_par(
  const int octave, const int ext_ct_prefix_sum, cudaTextureObject_t layer, const int w, const int h)
{
    const int extremum_index = blockIdx.x * blockDim.y;

    // Warp voting function...
    // Returns true only iff all non-exited trheads in warp have the contidion as true (non-zero)
    if(popsift::all(extremum_index >= dct.ext_ct[octave]))
        return; // a few trailing warps

    const int iext_off = dobuf.i_ext_off[octave][extremum_index];
    const InitialExtremum* iext = &dobuf.i_ext_dat[octave][iext_off];

    __shared__ float hist[64];
    __shared__ float sm_hist[64];
    __shared__ float refined_angle[64];
    __shared__ float yval[64];

    // Each thread in warp sets 2 parts of hist to zero
    // Hence when all warps have run this code all of hist will be zero
    hist[threadIdx.x + 0] = 0.0f;
    hist[threadIdx.x + 32] = 0.0f;

    /* keypoint fractional geometry */
    const float x = iext->xpos;
    const float y = iext->ypos;
    const int level = iext->lpos; // old_level;
    const float sig = iext->sigma;

    /* orientation histogram radius */
    const float sigw = ORI_WINFACTOR * sig; // how large raduis we need to search (sig is scale/sigma of the keypoint)
    const int32_t rad = (int)roundf((3.0f * sigw)); // why is it 3 times sigw IDK(but thats the maths)
    // rad is the radius used that is rounded to a whole number

    // Makes a negative number based on the scale of the point
    // The larger the scale the smaller the fatcor It's how
    // Simulates a gaussian that is applied to the magnitude of the histogram making the values futher
    // away from the point (x, y) that we are making descriptor for have less impact on the histogram
    const float factor = __fdividef(-0.5f, (sigw * sigw));
    const int sq_thres = rad * rad;

    // int xmin = max(1,     (int)floor(x - rad));
    // int xmax = min(w - 2, (int)floor(x + rad));
    // int ymin = max(1,     (int)floor(y - rad));
    // int ymax = min(h - 2, (int)floor(y + rad));
    int xmin = max(1, (int)roundf(x) - rad);
    int xmax = min(w - 2, (int)roundf(x) + rad);
    int ymin = max(1, (int)roundf(y) - rad);
    int ymax = min(h - 2, (int)roundf(y) + rad);
    // Compute search area from the radius around the keypoint
    // Should be bounded already. Min cannot be smaller than 1 and max cannot be larger than w-2 or h-2 hence no part
    // of the area can be out of bounds so no need to clamp

    // wx == hy as long as the clamping does not take place so most of the time the search area is a square
    int wx = xmax - xmin + 1;
    int hy = ymax - ymin + 1;
    int loops = wx * hy; // Total pixels in the rectangle (most likely square)
                         // that will be part of the histogram and descriptor
    if(blockIdx.x == 5 && blockIdx.y == 0 && octave == 0 && threadIdx.x == 0)
    {
        printf(
          "\n\nx = %f - y = %f - level = %d - sig = %f ---- sigw = %f rad = %d ---- factor = %f sq_thres = %d\n xmin "
          "= %d xmax = %d ymin = %d ymax = %d ---- wx = %d hy = %d loops = %d\n",
          x,
          y,
          level,
          sig,
          sigw,
          rad,
          factor,
          sq_thres,
          xmin,
          xmax,
          ymin,
          ymax,
          wx,
          hy,
          loops);
    }

    // All threads up until this point have computed the same values in the warp (block in this case)
    // the only difference they have done is setting different parts of hist to 0.0
    // Besides that they have executed the same code with same values

    __syncthreads(); // could be __syncwarp but would be the same result
    // conntinues if any of the non-exited threads in warp has condition i < loops as true (non-zero)
    for(int i = threadIdx.x; popsift::any(i < loops); i += blockDim.x) // plussing on blockDim.x == 32 (warp wide)
                                                                       // So every pixel is just ran once
    // Why do we keep running for threads that could just be masked out (diverged) is it because it is better to not
    // have diverged threads but the if will maek them diverge... Seems to be it would be just as good to just do the
    // condition each trhead by them selves and exit based on that... doing any to vote in the warp seems like it adds
    // alot of unnecessary overhead unless this has a crucial function that I don't understand
    {
        if(i < loops) // some threads (the ones with higer index could finish earlier than the smaller but only by 1
        {             // iteration depends on loops % 32 if 0 every one will be done at same time

            // Compute the pixel location for this current iteration in thread
            int yy = i / wx + ymin; // Current threads position in image
            int xx = i % wx + xmin;

            float grad;
            float theta;
            get_gradiant(grad, theta, xx, yy, layer, level);

            float dx = xx - x;
            float dy = yy - y;

            int sq_dist = dx * dx + dy * dy;
            if(sq_dist <= sq_thres)
            {
                // The larger sq_dist is the larger the negative number will be and hence the smaller the e^(sq_dist *
                // factor) will be so less emphasis is put on value futher away from the point. The factor pluss the
                // distance in this way resutls most likely in the gaussian kernel for this sigma(scale/size) of the
                // keypoint (grad is the magnitude of the gradient at this point)
                float weight = grad * expf(sq_dist * factor);

                // int bidx = (int)rintf( __fdividef( ORI_NBINS * (theta + M_PI), M_PI2 ) );

                // maps theta (angle of gradient vector) to it's bin. negative dy/dx will result in negative theta and
                // that will be mapped to the top half of the unit circle the bottom part will be posetive dy/dx values
                // So we end up having all orientations so range of values theta + M_PI can be is [0, M_PI2] and hence
                // the division is 0-1 and we multiply by number of bins to get the one it fits into and round to get
                // accurate index
                int bidx = (int)roundf(__fdividef(float(ORI_NBINS) * (theta + M_PI), M_PI2));

                // Neither of these if's should be possible to happen so not sure why they are here (debugging ?)
                if(bidx > ORI_NBINS)
                {
                    printf("Crashing: bin %d theta %f :-)\n", bidx, theta);
                }
                if(bidx < 0)
                {
                    printf("Crashing: bin %d theta %f :-)\n", bidx, theta);
                }

                // Because we are using round it rounds to closest and away on .5 so 0 will only happen for [0, 0.5) so
                // we need and ORI_NBINS is not a viable location so we need to add another half to 0 bin making each
                // bin evenely spaced out and as that is [.5 .0] it includes 0.5 for one and not the other like every
                // other bin so 0 bin is absolute theta values close to PI
                bidx = (bidx == ORI_NBINS) ? 0 : bidx;

                // bidx can only be [0, 35] so what are the rest of the 64 used for??
                atomicAdd(&hist[bidx], weight);
            }
        }
    }
    __syncthreads();

#ifdef WITH_VLFEAT_SMOOTHING
    for(int i = 0; i < 3; i++)
    {
        // using two arrays as it cannot be done inplace due to taking (prev + next + bin)/3 in smoothing
        sm_hist[threadIdx.x + 0] = smoothe<0>(hist, threadIdx.x + 0);
        sm_hist[threadIdx.x + 32] = smoothe<1>(hist, threadIdx.x + 32);
        __syncthreads();
        hist[threadIdx.x + 0] = smoothe<2>(sm_hist, threadIdx.x + 0);
        hist[threadIdx.x + 32] = smoothe<3>(sm_hist, threadIdx.x + 32);
        __syncthreads();
    }

    // Why are we doing this copy and not just use hist? which has the final smoothed value?
    sm_hist[threadIdx.x + 0] = hist[threadIdx.x + 0];
    sm_hist[threadIdx.x + 32] = hist[threadIdx.x + 32];
    __syncthreads();
#else  // not WITH_VLFEAT_SMOOTHING
    for(int bin = threadIdx.x; bin < ORI_NBINS; bin += blockDim.x)
    {
        int prev2 = bin - 2;
        int prev1 = bin - 1;
        int next1 = bin + 1;
        int next2 = bin + 2;
        if(prev2 < 0)
            prev2 += ORI_NBINS;
        if(prev1 < 0)
            prev1 += ORI_NBINS;
        if(next1 >= ORI_NBINS)
            next1 -= ORI_NBINS;
        if(next2 >= ORI_NBINS)
            next2 -= ORI_NBINS;
        sm_hist[bin] = (hist[prev2] + hist[next2] + (hist[prev1] + hist[next1]) * 4.0f + hist[bin] * 6.0f) / 16.0f;
    }
    __syncthreads();
#endif // not WITH_VLFEAT_SMOOTHING

    // sub-cell refinement of the histogram cell index, yielding the angle
    // not necessary to initialize, every cell is computed

    for(int bin = threadIdx.x; popsift::any(bin < ORI_NBINS); bin += blockDim.x)
    // Still don't see the point of needing to continue past ORI_NBINS
    // I they will access part that is zeroed and does the same math don't see how that is better than just masking
    // those threads out and then you don't have to do the any warp voting function each can just realy on them selves
    {
        const int prev = bin == 0 ? ORI_NBINS - 1 : bin - 1;
        const int next = bin == ORI_NBINS - 1 ? 0 : bin + 1;

        bool predicate = (bin < ORI_NBINS) && (sm_hist[bin] > max(sm_hist[prev], sm_hist[next]));

        // If bin is not larger than both prev and next it is set to 0
        // if predicate is true ^ then num is negative as bin is larger than both prev and next
        const float num = predicate ? 3.0f * sm_hist[prev] - 4.0f * sm_hist[bin] + 1.0f * sm_hist[next] : 0.0f;
        // const float num  = predicate ?   2.0f * sm_hist[prev]
        //                                - 4.0f * sm_hist[bin]
        //                                + 2.0f * sm_hist[next]
        //                              : 0.0f;

        // If predicate is true denB will be negative as bin is larger than both prev and next
        // if false predicate then it's 1
        const float denB = predicate ? 2.0f * (sm_hist[prev] - 2.0f * sm_hist[bin] + sm_hist[next]) : 1.0f;

        // in true case it's negative divided by negative hence newbin is posetive
        // in false for predicate newbin will be 0
        const float newbin = __fdividef(num, denB); // verified: accuracy OK

        // Don't see the purpose of this second part newbin >= 0.0f as that will always be true when the predicate is
        // true... hence we could omit I think(quite sure) only thing that can change the predicate is newbin <= 2.0f
        // In theory that would be enought but due to shor circuting and the fact that predicate will most of the time
        // be false it is pobably better to have it in the expression
        predicate = (predicate && newbin >= 0.0f && newbin <= 2.0f);

        refined_angle[bin] = predicate ? prev + newbin : -1;
        yval[bin] = predicate ? -(num * num) / (4.0f * denB) + sm_hist[prev] : -INFINITY;
        // 64 - ORI_NBINS will have -INFINITY by default...
    }
    __syncthreads();

    int2 best_index = make_int2(threadIdx.x, threadIdx.x + 32);

#define BLOCK 390

    // if(blockIdx.x == BLOCK && threadIdx.x == 0 && octave == 0)
    // {
    //     printf("\nBEFORE: best_index (%d, %d)", best_index.x, best_index.y);
    //     for(int i = 1; i < 64; ++i)
    //     {
    //         if(yval[i - 1] != -INFINITY && yval[i] != -INFINITY)
    //         {
    //             printf("Found for blockIdx.x = %d", blockIdx.x);
    //         }
    //     }
    // }

    if(blockIdx.x == BLOCK && threadIdx.x == 0 && octave == 0)
    {
        printf("\nBEFORE: best_index (%d, %d)\n", best_index.x, best_index.y);
        for(int i = 0; i < 64; i += 2)
        {
            printf("\tidx=%d --> %.6f  -- ", i, yval[i]);
            printf("idx=%d --> %.6f\n ", i + 1, yval[i + 1]);
        }
        printf("\n");
    }
    // Only goal here is to find the index that holds the higest value in yval right??
    // and we want that value to be at threadidx.x == 0
    BitonicSort::Warp32<float> sorter(yval);
    // if(blockIdx.x == BLOCK && threadIdx.x == 0 && octave == 0)
    //     printf("\n\nStarting sort\n");
    sorter.sort64(best_index);
    __syncthreads();
    if(blockIdx.x == BLOCK && threadIdx.x == 0 && octave == 0)
    {
        printf("\nAFTER: best_index (%d, %d)", best_index.x, best_index.y);
        //     printf("\n\nAFTER\n");
        //     for(int i = 0; i < 64; ++i)
        //     {
        //         printf(" %.6f ", yval[i]);
        //     }
    }

    // All threads retrieve the yval of thread 0, the largest
    // of all yvals.
    const float best_val = yval[best_index.x];
    const float yval_ref =
      0.8f * popsift::shuffle(best_val, 0); // broadcast best_val from thread 0 and multiply by 0.8f

    // Valid only if your thread's value is greater than or equal to 80% of the best val
    const bool valid = (best_val >= yval_ref);
    bool written = false;

    Extremum* ext = &dobuf.extrema[ext_ct_prefix_sum + extremum_index];

    if(threadIdx.x < ORIENTATION_MAX_COUNT)
    {
        if(valid)
        {
            float chosen_bin = refined_angle[best_index.x];
            if(chosen_bin >= ORI_NBINS)
                chosen_bin -= ORI_NBINS;
            // float th = __fdividef(M_PI2 * chosen_bin , ORI_NBINS) - M_PI;
            float th = ::fmaf(M_PI2 * chosen_bin, 1.0f / ORI_NBINS, -M_PI);
            ext->orientation[threadIdx.x] = th;
            written = true;
        }
    }

    int angles = __popc(popsift::ballot(written));
    if(threadIdx.x == 0)
    {
        ext->xpos = iext->xpos;
        ext->ypos = iext->ypos;
        ext->lpos = iext->lpos;
        ext->sigma = iext->sigma;
        ext->octave = octave;
        ext->num_ori = angles;
    }
}

}; // namespace popsift

class ExtremaRead
{
    const Extremum* const _oris;

  public:
    inline __device__ explicit ExtremaRead(const Extremum* const d_oris)
      : _oris(d_oris)
    {}

    inline __device__ int get(int n) const { return _oris[n].num_ori; }
};

class ExtremaWrt
{
    Extremum* _oris;

  public:
    inline __device__ explicit ExtremaWrt(Extremum* d_oris)
      : _oris(d_oris)
    {}

    inline __device__ void set(int n, int value) { _oris[n].idx_ori = value; }
};

class ExtremaTot
{
    int& _extrema_counter;

  public:
    inline __device__ explicit ExtremaTot(int& extrema_counter)
      : _extrema_counter(extrema_counter)
    {}

    inline __device__ void set(int value) { _extrema_counter = value; }
};

class ExtremaWrtMap
{
    int* _featvec_to_extrema_mapper;
    int _max_feat;

  public:
    inline __device__ ExtremaWrtMap(int* featvec_to_extrema_mapper, int max_feat)
      : _featvec_to_extrema_mapper(featvec_to_extrema_mapper)
      , _max_feat(max_feat)
    {}

    inline __device__ void set(int base, int num, int value)
    {
        int* baseptr = &_featvec_to_extrema_mapper[base];
        do
        {
            num--;
            if(base + num < _max_feat)
            {
                baseptr[num] = value;
            }
        } while(num > 0);
    }
};

__global__ void ori_prefix_sum(const int total_ext_ct, const int num_octaves)
{
    int total_ori = 0;
    Extremum* extremum = dobuf.extrema;
    int* feat_to_ext_map = dobuf.feat_to_ext_map;

    ExtremaRead r(extremum);
    ExtremaWrt w(extremum);
    ExtremaTot t(total_ori);
    ExtremaWrtMap wrtm(feat_to_ext_map, max(d_consts.max_orientations, dbuf.ori_allocated));
    ExclusivePrefixSum::Block<ExtremaRead, ExtremaWrt, ExtremaTot, ExtremaWrtMap>(total_ext_ct, r, w, t, wrtm);

    __syncthreads();

    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        dct.ext_ps[0] = 0;
        for(int o = 1; o < MAX_OCTAVES; o++)
        {
            dct.ext_ps[o] = dct.ext_ps[o - 1] + dct.ext_ct[o - 1];
        }

        for(int o = 0; o < MAX_OCTAVES; o++)
        {
            if(dct.ext_ct[o] == 0)
            {
                dct.ori_ct[o] = 0;
            }
            else
            {
                int fe = dct.ext_ps[o];         /* first extremum for this octave */
                int le = dct.ext_ps[o + 1] - 1; /* last  extremum for this octave */
                int lo_ori_index = dobuf.extrema[fe].idx_ori;
                int num_ori = dobuf.extrema[le].num_ori;
                int hi_ori_index = dobuf.extrema[le].idx_ori + num_ori;
                dct.ori_ct[o] = hi_ori_index - lo_ori_index;
            }
        }

        dct.ori_ps[0] = 0;
        for(int o = 1; o < MAX_OCTAVES; o++)
        {
            dct.ori_ps[o] = dct.ori_ps[o - 1] + dct.ori_ct[o - 1];
        }

        dct.ori_total = dct.ori_ps[MAX_OCTAVES - 1] + dct.ori_ct[MAX_OCTAVES - 1];
        dct.ext_total = dct.ext_ps[MAX_OCTAVES - 1] + dct.ext_ct[MAX_OCTAVES - 1];
    }
}

__host__ void Pyramid::orientation(const Config& conf)
{
    // How is synchronization ensured here??
    // I would assume that this would have to wait for the extrema to be finished
    // but there is no sycnthreads device or usage of the event. The event is recorded but
    // it is never waited on???
    readDescCountersFromDevice();

    int ext_total = 0;
    for(int o : hct.ext_ct)
    {
        if(o > 0)
        {
            ext_total += o;
        }
    }

    // Seems like this one does nothing the way it's set up for me only reutns ext_total that is passed int so doing
    // nothing...  but different if the ifdef goes the other way then the function does a lot
    // Filter functions are only called if necessary. They are very expensive, therefore add 10% slack.
    if(conf.getFilterMaxExtrema() > 0 && int(conf.getFilterMaxExtrema() * 1.1) < ext_total)
    {
        ext_total = extrema_filter_grid(conf, ext_total);
    }

    reallocExtrema(ext_total);

    int ext_ct_prefix_sum = 0;
    for(int octave = 0; octave < _num_octaves; octave++)
    {
        hct.ext_ps[octave] = ext_ct_prefix_sum;
        ext_ct_prefix_sum += hct.ext_ct[octave];
    }
    hct.ext_total = ext_ct_prefix_sum;

    cudaStream_t oct_0_str = _octaves[0].getStream();

    // for( int octave=0; octave<_num_octaves; octave++ )
    for(int octave = _num_octaves - 1; octave >= 0; octave--)
    {
        Octave& oct_obj = _octaves[octave];

        cudaStream_t oct_str = oct_obj.getStream();

        int num = hct.ext_ct[octave];

        if(num > 0)
        {
            dim3 block;
            dim3 grid;

            block.x = 32;
            block.y = 1;
            grid.x = num;

            ori_par<<<grid, block, 4 * 64 * sizeof(float), oct_str>>>(
              octave, hct.ext_ps[octave], oct_obj.getDataTexPoint(), oct_obj.getWidth(), oct_obj.getHeight());
            POP_SYNC_CHK;

            if(octave != 0)
            {
                cuda::event_record(oct_obj.getEventOriDone(), oct_str, __FILE__, __LINE__);
                cuda::event_wait(oct_obj.getEventOriDone(), oct_0_str, __FILE__, __LINE__);
            }
        }
    }

    /* Compute and set the orientation prefixes on the device */
    dim3 block;
    dim3 grid;
    block.x = 32;
    block.y = 32;
    grid.x = 1;
    ori_prefix_sum<<<grid, block, 0, oct_0_str>>>(ext_ct_prefix_sum, _num_octaves);
    POP_SYNC_CHK;

    cudaDeviceSynchronize();
}
