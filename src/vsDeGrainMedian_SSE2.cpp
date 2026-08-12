#include <emmintrin.h>
#include <cstring>

#include "vsDeGrainMedian.h"

#define LoadPixelsSSE2 \
    __m128i p1, p2, p3, \
            p4, p5, p6, \
            p7, p8, p9; \
    \
    __m128i s1, s2, s3, \
            s4, s5, s6, \
            s7, s8, s9; \
    \
    __m128i n1, n2, n3, \
            n4, n5, n6, \
            n7, n8, n9; \
    \
    p1 = _mm_loadu_si128((const __m128i *)&prevp[x - p_dist - 1]); \
    p2 = _mm_loadu_si128((const __m128i *)&prevp[x - p_dist]); \
    p3 = _mm_loadu_si128((const __m128i *)&prevp[x - p_dist + 1]); \
    p4 = _mm_loadu_si128((const __m128i *)&prevp[x - 1]); \
    p5 = _mm_loadu_si128((const __m128i *)&prevp[x]); \
    p6 = _mm_loadu_si128((const __m128i *)&prevp[x + 1]); \
    p7 = _mm_loadu_si128((const __m128i *)&prevp[x + p_dist - 1]); \
    p8 = _mm_loadu_si128((const __m128i *)&prevp[x + p_dist]); \
    p9 = _mm_loadu_si128((const __m128i *)&prevp[x + p_dist + 1]); \
    \
    s1 = _mm_loadu_si128((const __m128i *)&srcp[x - s_dist - 1]); \
    s2 = _mm_loadu_si128((const __m128i *)&srcp[x - s_dist]); \
    s3 = _mm_loadu_si128((const __m128i *)&srcp[x - s_dist + 1]); \
    s4 = _mm_loadu_si128((const __m128i *)&srcp[x - 1]); \
    s5 = _mm_loadu_si128((const __m128i *)&srcp[x]); \
    s6 = _mm_loadu_si128((const __m128i *)&srcp[x + 1]); \
    s7 = _mm_loadu_si128((const __m128i *)&srcp[x + s_dist - 1]); \
    s8 = _mm_loadu_si128((const __m128i *)&srcp[x + s_dist]); \
    s9 = _mm_loadu_si128((const __m128i *)&srcp[x + s_dist + 1]); \
    \
    n1 = _mm_loadu_si128((const __m128i *)&nextp[x - n_dist - 1]); \
    n2 = _mm_loadu_si128((const __m128i *)&nextp[x - n_dist]); \
    n3 = _mm_loadu_si128((const __m128i *)&nextp[x - n_dist + 1]); \
    n4 = _mm_loadu_si128((const __m128i *)&nextp[x - 1]); \
    n5 = _mm_loadu_si128((const __m128i *)&nextp[x]); \
    n6 = _mm_loadu_si128((const __m128i *)&nextp[x + 1]); \
    n7 = _mm_loadu_si128((const __m128i *)&nextp[x + n_dist - 1]); \
    n8 = _mm_loadu_si128((const __m128i *)&nextp[x + n_dist]); \
    n9 = _mm_loadu_si128((const __m128i *)&nextp[x + n_dist + 1]);

template <typename PixelType>
static AVS_FORCEINLINE __m128i mm_min_epu(const __m128i& a, const __m128i& b)
{
    if (sizeof(PixelType) == 1)
        return _mm_min_epu8(a, b);
    else
    {
        const __m128i sign = _mm_set1_epi16(32768);
        return _mm_add_epi16(_mm_min_epi16(_mm_sub_epi16(a, sign), _mm_sub_epi16(b, sign)), sign);
    }
}

template <typename PixelType>
static AVS_FORCEINLINE __m128i mm_max_epu(const __m128i& a, const __m128i& b)
{
    if (sizeof(PixelType) == 1)
        return _mm_max_epu8(a, b);
    else
    {
        const __m128i sign = _mm_set1_epi16(32768);
        return _mm_add_epi16(_mm_max_epi16(_mm_sub_epi16(a, sign), _mm_sub_epi16(b, sign)), sign);
    }
}

template <typename PixelType>
static AVS_FORCEINLINE __m128i mm_adds_epu(const __m128i& a, const __m128i& b, const __m128i& pixel_max)
{
    if (sizeof(PixelType) == 1)
        return _mm_adds_epu8(a, b);
    else
    {
        __m128i sum = _mm_adds_epu16(a, b);
        return mm_min_epu<PixelType>(sum, pixel_max);
    }
}

template <typename PixelType>
static AVS_FORCEINLINE __m128i mm_subs_epu(const __m128i& a, const __m128i& b)
{
    if (sizeof(PixelType) == 1)
        return _mm_subs_epu8(a, b);
    else
    {
        return _mm_subs_epu16(a, b);
    }
}

template <typename PixelType>
static AVS_FORCEINLINE __m128i mm_cmpeq_epi(const __m128i& a, const __m128i& b)
{
    if (sizeof(PixelType) == 1)
        return _mm_cmpeq_epi8(a, b);
    else
    {
        return _mm_cmpeq_epi16(a, b);
    }
}

template <typename PixelType>
static AVS_FORCEINLINE __m128i mm_set1_epi(int value)
{
    if (sizeof(PixelType) == 1)
        return _mm_set1_epi8(value);
    else
    {
        return _mm_set1_epi16(value);
    }
}

template <typename PixelType>
static AVS_FORCEINLINE void checkBetterNeighboursSSE2(const __m128i& a, const __m128i& b, __m128i& diff, __m128i& min, __m128i& max)
{
    __m128i new_min = mm_min_epu<PixelType>(a, b);
    __m128i new_max = mm_max_epu<PixelType>(a, b);
    __m128i new_diff = mm_subs_epu<PixelType>(new_max, new_min);

    __m128i mask = mm_subs_epu<PixelType>(new_diff, diff);
    mask = mm_cmpeq_epi<PixelType>(mask, _mm_setzero_si128());

    new_min = _mm_and_si128(new_min, mask);
    new_max = _mm_and_si128(new_max, mask);
    new_diff = _mm_and_si128(new_diff, mask);

    mask = mm_cmpeq_epi<PixelType>(mask, _mm_setzero_si128());

    diff = _mm_and_si128(diff, mask);
    min = _mm_and_si128(min, mask);
    max = _mm_and_si128(max, mask);

    diff = _mm_or_si128(diff, new_diff);
    min = _mm_or_si128(min, new_min);
    max = _mm_or_si128(max, new_max);
}

template <int mode, typename PixelType>
struct Asdf
{
    static AVS_FORCEINLINE void diagWeightSSE2(const __m128i& oldp, const __m128i& bound1, const __m128i& bound2, __m128i& old_result, __m128i& old_weight, const __m128i& pixel_max)
    {
        __m128i max = mm_max_epu<PixelType>(bound1, bound2);
        __m128i min = mm_min_epu<PixelType>(bound1, bound2);
        __m128i diff = mm_subs_epu<PixelType>(max, min);

        __m128i reg2 = mm_subs_epu<PixelType>(oldp, max);

        __m128i newp = mm_min_epu<PixelType>(max, oldp);
        newp = mm_max_epu<PixelType>(newp, min);

        __m128i weight = mm_subs_epu<PixelType>(min, oldp);
        weight = mm_max_epu<PixelType>(weight, reg2);

        if (mode == 4)
            weight = mm_adds_epu<PixelType>(weight, weight, pixel_max);
        else if (mode == 2)
            diff = mm_adds_epu<PixelType>(diff, diff, pixel_max);
        else if (mode == 1)
        {
            diff = mm_adds_epu<PixelType>(diff, diff, pixel_max);
            diff = mm_adds_epu<PixelType>(diff, diff, pixel_max);
        }

        weight = mm_adds_epu<PixelType>(weight, diff, pixel_max);

        old_weight = mm_min_epu<PixelType>(old_weight, weight);
        weight = mm_cmpeq_epi<PixelType>(weight, old_weight);
        old_result = mm_subs_epu<PixelType>(old_result, weight);
        weight = _mm_and_si128(weight, newp);
        old_result = _mm_or_si128(old_result, weight);
    }
};

template <typename PixelType>
struct Asdf<5, PixelType>
{
    static AVS_FORCEINLINE void diagWeightSSE2(const __m128i& oldp, const __m128i& bound1, const __m128i& bound2, __m128i& old_result, __m128i& old_weight, const __m128i& pixel_max)
    {
        (void)pixel_max;

        __m128i max = mm_max_epu<PixelType>(bound1, bound2);
        __m128i min = mm_min_epu<PixelType>(bound1, bound2);

        __m128i newp = mm_min_epu<PixelType>(max, oldp);
        newp = mm_max_epu<PixelType>(newp, min);

        __m128i reg2 = mm_subs_epu<PixelType>(oldp, max);
        __m128i weight = mm_subs_epu<PixelType>(min, oldp);
        weight = mm_max_epu<PixelType>(weight, reg2);

        old_weight = mm_min_epu<PixelType>(old_weight, weight);
        weight = mm_cmpeq_epi<PixelType>(weight, old_weight);
        old_result = mm_subs_epu<PixelType>(old_result, weight);
        weight = _mm_and_si128(weight, newp);
        old_result = _mm_or_si128(old_result, weight);
    }
};

template <typename PixelType>
static AVS_FORCEINLINE __m128i limitPixelCorrectionSSE2(const __m128i& old_pixel, const __m128i& new_pixel, const __m128i& limit,
    const __m128i& pixel_max)
{
    const __m128i upper = mm_adds_epu<PixelType>(old_pixel, limit, pixel_max);
    const __m128i lower = mm_subs_epu<PixelType>(old_pixel, limit);

    const __m128i clamped = mm_min_epu<PixelType>(new_pixel, upper);

    return mm_max_epu<PixelType>(clamped, lower);
}

// Wrapped in struct because function templates can't be partially specialised.
template <int mode, bool norow, typename PixelType>
struct DegrainSSE2
{
    static AVS_FORCEINLINE __m128i degrainPixels(const PixelType* AVS_RESTRICT prevp, const PixelType* AVS_RESTRICT srcp,
        const PixelType* AVS_RESTRICT nextp, int x, int p_dist, int s_dist, int n_dist, const __m128i& limit, const __m128i& pixel_max)
    {
        LoadPixelsSSE2;

        __m128i result;
        __m128i weight = _mm_set1_epi8(255);

        Asdf<mode, PixelType>::diagWeightSSE2(s5, s1, s9, result, weight, pixel_max);
        Asdf<mode, PixelType>::diagWeightSSE2(s5, s7, s3, result, weight, pixel_max);
        Asdf<mode, PixelType>::diagWeightSSE2(s5, s8, s2, result, weight, pixel_max);
        if (!norow)
            Asdf<mode, PixelType>::diagWeightSSE2(s5, s6, s4, result, weight, pixel_max);

        Asdf<mode, PixelType>::diagWeightSSE2(s5, n1, p9, result, weight, pixel_max);
        Asdf<mode, PixelType>::diagWeightSSE2(s5, n3, p7, result, weight, pixel_max);
        Asdf<mode, PixelType>::diagWeightSSE2(s5, n7, p3, result, weight, pixel_max);
        Asdf<mode, PixelType>::diagWeightSSE2(s5, n9, p1, result, weight, pixel_max);
        Asdf<mode, PixelType>::diagWeightSSE2(s5, n8, p2, result, weight, pixel_max);
        Asdf<mode, PixelType>::diagWeightSSE2(s5, n2, p8, result, weight, pixel_max);
        Asdf<mode, PixelType>::diagWeightSSE2(s5, n4, p6, result, weight, pixel_max);
        Asdf<mode, PixelType>::diagWeightSSE2(s5, n6, p4, result, weight, pixel_max);
        Asdf<mode, PixelType>::diagWeightSSE2(s5, n5, p5, result, weight, pixel_max);

        return limitPixelCorrectionSSE2<PixelType>(s5, result, limit, pixel_max);
    }
};

template <bool norow, typename PixelType>
struct DegrainSSE2<0, norow, PixelType>
{
    static AVS_FORCEINLINE __m128i degrainPixels(const PixelType* AVS_RESTRICT prevp, const PixelType* AVS_RESTRICT srcp,
        const PixelType* AVS_RESTRICT nextp, int x, int p_dist, int s_dist, int n_dist, const __m128i& limit, const __m128i& pixel_max)
    {
        LoadPixelsSSE2;

        __m128i diff = _mm_set1_epi8(255);
        __m128i min = _mm_setzero_si128();
        __m128i max = _mm_set1_epi8(255);

        checkBetterNeighboursSSE2<PixelType>(n1, p9, diff, min, max);
        checkBetterNeighboursSSE2<PixelType>(n3, p7, diff, min, max);
        checkBetterNeighboursSSE2<PixelType>(n7, p3, diff, min, max);
        checkBetterNeighboursSSE2<PixelType>(n9, p1, diff, min, max);
        checkBetterNeighboursSSE2<PixelType>(n8, p2, diff, min, max);
        checkBetterNeighboursSSE2<PixelType>(n2, p8, diff, min, max);
        checkBetterNeighboursSSE2<PixelType>(n4, p6, diff, min, max);
        checkBetterNeighboursSSE2<PixelType>(n6, p4, diff, min, max);
        checkBetterNeighboursSSE2<PixelType>(n5, p5, diff, min, max);

        checkBetterNeighboursSSE2<PixelType>(s1, s9, diff, min, max);
        checkBetterNeighboursSSE2<PixelType>(s3, s7, diff, min, max);
        checkBetterNeighboursSSE2<PixelType>(s2, s8, diff, min, max);
        if (!norow)
            checkBetterNeighboursSSE2<PixelType>(s4, s6, diff, min, max);

        __m128i result = mm_max_epu<PixelType>(min, mm_min_epu<PixelType>(s5, max));

        return limitPixelCorrectionSSE2<PixelType>(s5, result, limit, pixel_max);
    }
};

template <int mode, bool norow, typename PixelType>
void degrainPlaneSSE2(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8, const uint8_t* AVS_RESTRICT nextp8,
    uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride, int width, int height, int limit,
    int interlaced, int pixel_max)
{
    const PixelType* prevp = (const PixelType*)prevp8;
    const PixelType* srcp = (const PixelType*)srcp8;
    const PixelType* nextp = (const PixelType*)nextp8;
    PixelType* dstp = (PixelType*)dstp8;

    prev_stride /= sizeof(PixelType);
    src_stride /= sizeof(PixelType);
    next_stride /= sizeof(PixelType);
    dst_stride /= sizeof(PixelType);
    width /= sizeof(PixelType);

    // Each frame steps by its own pitch. prev, src and next need not share
    // one: a field subframe views its woven parent at double pitch, while a
    // freshly allocated frame is compact, and a filter upstream can hand back
    // a mixture of the two.
    const int p_dist = prev_stride << interlaced;
    const int s_dist = src_stride << interlaced;
    const int n_dist = next_stride << interlaced;
    const int skip_rows = 1 << interlaced;

    // Copy first line(s).
    for (int y = 0; y < skip_rows; ++y)
    {
        memcpy(dstp, srcp, width * sizeof(PixelType));

        prevp += prev_stride;
        srcp += src_stride;
        nextp += next_stride;
        dstp += dst_stride;
    }

    __m128i packed_limit = mm_set1_epi<PixelType>(limit);

    // Only used in the uint16_t case.
    __m128i packed_pixel_max = _mm_set1_epi16(pixel_max);

    const int pixels_in_xmm = 16 / sizeof(PixelType);

    // Highest vector position whose 3x3x3 neighbourhood stays inside the row:
    // degrainPixels reads columns [x-1, x+pixels_in_xmm] and the store covers
    // [x, x+pixels_in_xmm-1], while columns 0 and width-1 are copied verbatim.
    //
    // The previous bound, (width & ~(pixels_in_xmm-1)) + 2 clamped against the
    // stride, let the loop run one vector too far whenever width was a multiple
    // of pixels_in_xmm: it read two pixels and wrote one pixel beyond the row,
    // into the padding. Harmless on every row but the last, where the frame
    // buffer ends and the overrun leaves the allocation - an intermittent
    // access violation, since it only faults when the buffer happens to end
    // flush against an unmapped page.
    const int x_last = width - pixels_in_xmm - 1;

    for (int y = skip_rows; y < height - skip_rows; ++y)
    {
        if (x_last < 1)
        {
            // Plane narrower than pixels_in_xmm + 2 - a 32x32 clip in YV12 has
            // 16-column chroma. No vector position has a complete neighbourhood,
            // so pass the row through rather than indexing before the row start.
            memcpy(dstp, srcp, width * sizeof(PixelType));
        }
        else
        {
            dstp[0] = srcp[0];

            int x = 1;
            for (; x <= x_last; x += pixels_in_xmm)
                _mm_storeu_si128((__m128i*) & dstp[x], DegrainSSE2<mode, norow, PixelType>::degrainPixels(prevp, srcp, nextp, x, p_dist, s_dist, n_dist, packed_limit, packed_pixel_max));

            // Widths that are not 1 + k*pixels_in_xmm leave a remainder; one more
            // vector at x_last covers it, overlapping what the loop already wrote.
            if (x - pixels_in_xmm != x_last)
                _mm_storeu_si128((__m128i*) & dstp[x_last], DegrainSSE2<mode, norow, PixelType>::degrainPixels(prevp, srcp, nextp, x_last, p_dist, s_dist, n_dist, packed_limit, packed_pixel_max));

            dstp[width - 1] = srcp[width - 1];
        }

        prevp += prev_stride;
        srcp += src_stride;
        nextp += next_stride;
        dstp += dst_stride;
    }

    // Copy last line(s).
    for (int y = 0; y < skip_rows; ++y)
    {
        memcpy(dstp, srcp, width * sizeof(PixelType));

        srcp += src_stride;
        dstp += dst_stride;
    }
}

template void degrainPlaneSSE2<0, true, uint8_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<1, true, uint8_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<2, true, uint8_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<3, true, uint8_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<4, true, uint8_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<5, true, uint8_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);

template void degrainPlaneSSE2<0, false, uint8_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<1, false, uint8_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<2, false, uint8_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<3, false, uint8_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<4, false, uint8_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<5, false, uint8_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);

template void degrainPlaneSSE2<0, true, uint16_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<1, true, uint16_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<2, true, uint16_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<3, true, uint16_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<4, true, uint16_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<5, true, uint16_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);

template void degrainPlaneSSE2<0, false, uint16_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<1, false, uint16_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<2, false, uint16_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<3, false, uint16_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<4, false, uint16_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
template void degrainPlaneSSE2<5, false, uint16_t>(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8,
    const uint8_t* AVS_RESTRICT nextp8, uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride,
    int width, int height, int limit, int interlaced, int pixel_max);
