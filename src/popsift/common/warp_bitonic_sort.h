/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "assist.h"

#include <cuda_runtime.h>

namespace popsift {
namespace BitonicSort {

template<class T>
class Warp32
{
    T* _array;

  public:
    __device__ inline Warp32(T* array)
      : _array(array)
    {}

    __device__ inline int sort32(int my_index)
    {
        for(int outer = 0; outer < 5; outer++)
        {
            for(int inner = outer; inner >= 0; inner--)
            {
                my_index = shiftit(my_index, inner, outer + 1, false);
            }
        }
        return my_index;
    }

    __device__ inline void sort64(int2& my_indeces)
    {
        for(int outer = 0; outer < 5; outer++)
        {
            for(int inner = outer; inner >= 0; inner--)
            {
                my_indeces.x = shiftit(my_indeces.x, inner, outer + 1, false);
                my_indeces.y = shiftit(my_indeces.y, inner, outer + 1, true);
            }
        }

        if(_array[my_indeces.x] < _array[my_indeces.y])
            swap(my_indeces.x, my_indeces.y);

        for(int outer = 0; outer < 5; outer++)
        {
            for(int inner = outer; inner >= 0; inner--)
            {
                my_indeces.x = shiftit(my_indeces.x, inner, outer + 1, false);
                my_indeces.y = shiftit(my_indeces.y, inner, outer + 1, false);
            }
        }
    }

  private:
    __device__ inline int shiftit(const int my_index, const int shift, const int direction, const bool increasing)
    {
        const T my_val = _array[my_index];
        const T other_val = popsift::shuffle_xor(my_val, 1 << shift);

        // This computes the same mask but for one higher than the one below so different &
        // it's direction number of zero then direction number of direction(aka true) and that repeats until end
        const bool reverse = (threadIdx.x & (1 << direction));
        // So the (1 << direction) is values {2, 4, 8, 16, 32} so for 2 every even threadIdx.x is true and rest false
        // the rule is: first direction number of threadIdx.x values is false then direction number of values is true
        // and then direction value of number is false and so on so for 4 the warp reverse values would be
        // 00001111000011110000111100001111
        // and for 8
        // 00000000111111110000000011111111
        // and for 16
        // 00000000000000001111111111111111
        // When true it means that other_val is received from a thread with lower threadIdx.x and when true it means
        // its' received from a threadIdx.x with higher value

        // True for threadIdx.x that get's values in the shuffle_xor that is from a higher threadIdx.x
        // true for shift number of threads then false for shift number of threads and repeats until end(31)
        const bool id_less = ((threadIdx.x & (1 << shift)) == 0);

        // If it thread get other_val from a thread with higher id it will be true if it's value is higher than other
        // otherwise if it gets other_val from lower thread id it will be true if my_val is smaler than other_val
        // if equal it's always false
        const bool my_more = id_less ? (my_val > other_val) : (my_val < other_val);

        // xor my_more with reverse and then xor that with increasing ^ is bitwise xor but onely one bit for bool
        const bool must_swap = !(my_more ^ reverse ^ increasing);

        int lane = must_swap ? (1 << shift) : 0;

        // Returns wether or not the index need to swap if lane == 0 it will keep it's value
        return popsift::shuffle_xor(my_index, lane);
    }

    __device__ inline void swap(int& l, int& r)
    {
        int m = r;
        r = l;
        l = m;
    }
};
} // namespace popsift
} // namespace BitonicSort
