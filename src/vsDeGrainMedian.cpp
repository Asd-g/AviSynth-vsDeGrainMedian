#include <algorithm>

#include "vsDeGrainMedian.h"

#define LoadPixelsScalar \
    int p1, p2, p3, \
        p4, p5, p6, \
        p7, p8, p9; \
    \
    int s1, s2, s3, \
        s4, s5, s6, \
        s7, s8, s9; \
    \
    int n1, n2, n3, \
        n4, n5, n6, \
        n7, n8, n9; \
    \
    p1 = prevp[x - distance - 1]; \
    p2 = prevp[x - distance]; \
    p3 = prevp[x - distance + 1]; \
    p4 = prevp[x - 1]; \
    p5 = prevp[x]; \
    p6 = prevp[x + 1]; \
    p7 = prevp[x + distance - 1]; \
    p8 = prevp[x + distance]; \
    p9 = prevp[x + distance + 1]; \
    \
    s1 = srcp[x - distance - 1]; \
    s2 = srcp[x - distance]; \
    s3 = srcp[x - distance + 1]; \
    s4 = srcp[x - 1]; \
    s5 = srcp[x]; \
    s6 = srcp[x + 1]; \
    s7 = srcp[x + distance - 1]; \
    s8 = srcp[x + distance]; \
    s9 = srcp[x + distance + 1]; \
    \
    n1 = nextp[x - distance - 1]; \
    n2 = nextp[x - distance]; \
    n3 = nextp[x - distance + 1]; \
    n4 = nextp[x - 1]; \
    n5 = nextp[x]; \
    n6 = nextp[x + 1]; \
    n7 = nextp[x + distance - 1]; \
    n8 = nextp[x + distance]; \
    n9 = nextp[x + distance + 1];

static FORCE_INLINE void checkBetterNeighboursScalar(int a, int b, int& diff, int& min, int& max)
{
    int new_diff = std::abs(a - b);

    if (new_diff <= diff)
    {
        diff = new_diff;
        min = std::min(a, b);
        max = std::max(a, b);
    }
}

template <int mode>
static FORCE_INLINE void diagWeightScalar(int oldp, int bound1, int bound2, int& old_result, int& old_weight, int pixel_max)
{
    // Sucks but I can't figure it out any further.

    int newp = std::max(bound1, bound2);
    int weight = std::min(bound1, bound2);

    int reg2 = std::max(0, oldp - std::max(bound1, bound2));

    newp = std::max(weight, std::min(newp, oldp));
    weight = std::max(0, weight - oldp);
    weight = std::max(weight, reg2);

    int diff = std::abs(bound1 - bound2);

    if (mode == 4)
        weight += weight;
    else if (mode == 2)
        diff += diff;
    else if (mode == 1)
    {
        diff += diff;
        diff += diff;
    }

    weight = std::min(weight + diff, pixel_max);

    if (weight <= old_weight)
    {
        old_weight = weight;
        old_result = newp;
    }
}

template <>
FORCE_INLINE void diagWeightScalar<5>(int oldp, int bound1, int bound2, int& old_result, int& old_weight, int pixel_max)
{
    (void)pixel_max;

    int newp = std::max(bound1, bound2);
    int weight = std::min(bound1, bound2);
    int reg2 = std::max(0, oldp - newp);
    newp = std::min(newp, oldp);
    newp = std::max(newp, weight);
    weight = std::max(0, weight - oldp);
    weight = std::max(weight, reg2);

    if (weight <= old_weight)
    {
        old_weight = weight;
        old_result = newp;
    }
}

static FORCE_INLINE int limitPixelCorrectionScalar(int old_pixel, int new_pixel, int limit, int pixel_max)
{
    int lower = std::max(0, old_pixel - limit);
    int upper = std::min(old_pixel + limit, pixel_max);
    return std::max(lower, std::min(new_pixel, upper));
}

template <int mode, bool norow, typename PixelType>
struct DegrainScalar
{

    static FORCE_INLINE int degrainPixel(const PixelType* prevp, const PixelType* srcp, const PixelType* nextp, int x, int distance, int limit, int pixel_max)
    {
        LoadPixelsScalar;

        // 65535 works for any bit depth between 8 and 16.
        int result = 0;
        int weight = 65535;

        diagWeightScalar<mode>(s5, s1, s9, result, weight, pixel_max);
        diagWeightScalar<mode>(s5, s7, s3, result, weight, pixel_max);
        diagWeightScalar<mode>(s5, s8, s2, result, weight, pixel_max);
        if (!norow)
            diagWeightScalar<mode>(s5, s6, s4, result, weight, pixel_max);

        diagWeightScalar<mode>(s5, n1, p9, result, weight, pixel_max);
        diagWeightScalar<mode>(s5, n3, p7, result, weight, pixel_max);
        diagWeightScalar<mode>(s5, n7, p3, result, weight, pixel_max);
        diagWeightScalar<mode>(s5, n9, p1, result, weight, pixel_max);
        diagWeightScalar<mode>(s5, n8, p2, result, weight, pixel_max);
        diagWeightScalar<mode>(s5, n2, p8, result, weight, pixel_max);
        diagWeightScalar<mode>(s5, n4, p6, result, weight, pixel_max);
        diagWeightScalar<mode>(s5, n6, p4, result, weight, pixel_max);
        diagWeightScalar<mode>(s5, n5, p5, result, weight, pixel_max);

        return limitPixelCorrectionScalar(s5, result, limit, pixel_max);
    }
};

template <bool norow, typename PixelType>
struct DegrainScalar<0, norow, PixelType>
{

    static FORCE_INLINE int degrainPixel(const PixelType* prevp, const PixelType* srcp, const PixelType* nextp, int x, int distance, int limit, int pixel_max)
    {
        LoadPixelsScalar;

        // 65535 works for any bit depth between 8 and 16.
        int diff = 65535;
        int min = 0;
        int max = 65535;

        checkBetterNeighboursScalar(n1, p9, diff, min, max);
        checkBetterNeighboursScalar(n3, p7, diff, min, max);
        checkBetterNeighboursScalar(n7, p3, diff, min, max);
        checkBetterNeighboursScalar(n9, p1, diff, min, max);
        checkBetterNeighboursScalar(n8, p2, diff, min, max);
        checkBetterNeighboursScalar(n2, p8, diff, min, max);
        checkBetterNeighboursScalar(n4, p6, diff, min, max);
        checkBetterNeighboursScalar(n6, p4, diff, min, max);
        checkBetterNeighboursScalar(n5, p5, diff, min, max);

        checkBetterNeighboursScalar(s1, s9, diff, min, max);
        checkBetterNeighboursScalar(s3, s7, diff, min, max);
        checkBetterNeighboursScalar(s2, s8, diff, min, max);
        if (!norow)
            checkBetterNeighboursScalar(s4, s6, diff, min, max);

        int result = std::max(min, std::min(s5, max));

        return limitPixelCorrectionScalar(s5, result, limit, pixel_max);
    }
};

template <int mode, bool norow, typename PixelType>
static void degrainPlaneScalar(const uint8_t* prevp8, const uint8_t* srcp8, const uint8_t* nextp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int limit, int interlaced, int pixel_max)
{
    const PixelType* prevp = (const PixelType*)prevp8;
    const PixelType* srcp = (const PixelType*)srcp8;
    const PixelType* nextp = (const PixelType*)nextp8;
    PixelType* dstp = (PixelType*)dstp8;

    stride /= sizeof(PixelType);
    dst_stride /= sizeof(PixelType);
    width /= sizeof(PixelType);

    const int distance = stride << interlaced;
    const int skip_rows = 1 << interlaced;

    // Copy first line(s).
    for (int y = 0; y < skip_rows; ++y)
    {
        memcpy(dstp, srcp, width * sizeof(PixelType));

        prevp += stride;
        srcp += stride;
        nextp += stride;
        dstp += dst_stride;
    }

    for (int y = skip_rows; y < height - skip_rows; ++y)
    {
        dstp[0] = srcp[0];

        for (int x = 1; x < width - 1; ++x)
            dstp[x] = DegrainScalar<mode, norow, PixelType>::degrainPixel(prevp, srcp, nextp, x, distance, limit, pixel_max);

        dstp[width - 1] = srcp[width - 1];

        prevp += stride;
        srcp += stride;
        nextp += stride;
        dstp += dst_stride;
    }

    // Copy last line(s).
    for (int y = 0; y < skip_rows; ++y)
    {
        memcpy(dstp, srcp, width * sizeof(PixelType));

        srcp += stride;
        dstp += dst_stride;
    }
}

vsDeGrainMedian::vsDeGrainMedian(PClip _child, int limitY, int limitU, int limitV, int modeY, int modeU, int modeV, bool interlaced, bool norow, int opt, IScriptEnvironment* env)
    : GenericVideoFilter(_child), _interlaced(interlaced)
{
    limit[0] = limitY;
    limit[1] = limitU;
    limit[2] = limitV;

    if (!vi.IsPlanar() || vi.BitsPerComponent() == 32)
        env->ThrowError("vsDeGrainMedian: only 8..16-bit planar clips are supported.");
    if (limit[0] < 0 || limit[0] > 255)
        env->ThrowError("vsDeGrainMedian: limitY must be between 0..255.");
    if (limit[1] < 0 || limit[1] > 255)
        env->ThrowError("vsDeGrainMedian: limitU must be between 0..255.");
    if (limit[2] < 0 || limit[2] > 255)
        env->ThrowError("vsDeGrainMedian: limitV must be between 0..255.");
    if (modeY < 0 || modeY > 5)
        env->ThrowError("vsDeGrainMedian: modeY must be between 0..5.");
    if (modeU < 0 || modeU > 5)
        env->ThrowError("vsDeGrainMedian: modeU must be between 0..5.");
    if (modeV < 0 || modeV > 5)
        env->ThrowError("vsDeGrainMedian: modeV must be between 0..5.");
    if (opt < -1 || opt > 1)
        env->ThrowError("vsDeGrainMedian: opt must be between -1..1.");
    if (opt == 1 && !(env->GetCPUFlags() & CPUF_SSE2))
        env->ThrowError("vsTEdgeMask: opt=1 requires SSE2.");

    const int mode[3] = { modeY, modeU, modeV };

    for (int i = 0; i < vi.NumComponents(); ++i)
    {
        if ((!!(env->GetCPUFlags() & CPUF_SSE2) && opt == -1) || opt == 1)
        {
            if (vi.BitsPerComponent() == 8)
            {
                if (norow)
                {
                    switch (mode[i])
                    {
                        case 0: degrainp[i] = degrainPlaneSSE2<0, true, uint8_t>; break;
                        case 1: degrainp[i] = degrainPlaneSSE2<1, true, uint8_t>; break;
                        case 2: degrainp[i] = degrainPlaneSSE2<2, true, uint8_t>; break;
                        case 3: degrainp[i] = degrainPlaneSSE2<3, true, uint8_t>; break;
                        case 4: degrainp[i] = degrainPlaneSSE2<4, true, uint8_t>; break;
                        case 5: degrainp[i] = degrainPlaneSSE2<5, true, uint8_t>; break;
                    }
                }
                else
                {
                    switch (mode[i])
                    {
                        case 0: degrainp[i] = degrainPlaneSSE2<0, false, uint8_t>; break;
                        case 1: degrainp[i] = degrainPlaneSSE2<1, false, uint8_t>; break;
                        case 2: degrainp[i] = degrainPlaneSSE2<2, false, uint8_t>; break;
                        case 3: degrainp[i] = degrainPlaneSSE2<3, false, uint8_t>; break;
                        case 4: degrainp[i] = degrainPlaneSSE2<4, false, uint8_t>; break;
                        case 5: degrainp[i] = degrainPlaneSSE2<5, false, uint8_t>; break;
                    }
                }
            }
            else
            {
                if (norow)
                {
                    switch (mode[i])
                    {
                        case 0: degrainp[i] = degrainPlaneSSE2<0, true, uint16_t>; break;
                        case 1: degrainp[i] = degrainPlaneSSE2<1, true, uint16_t>; break;
                        case 2: degrainp[i] = degrainPlaneSSE2<2, true, uint16_t>; break;
                        case 3: degrainp[i] = degrainPlaneSSE2<3, true, uint16_t>; break;
                        case 4: degrainp[i] = degrainPlaneSSE2<4, true, uint16_t>; break;
                        case 5: degrainp[i] = degrainPlaneSSE2<5, true, uint16_t>; break;
                    }
                }
                else
                {
                    switch (mode[i])
                    {
                        case 0: degrainp[i] = degrainPlaneSSE2<0, false, uint16_t>; break;
                        case 1: degrainp[i] = degrainPlaneSSE2<1, false, uint16_t>; break;
                        case 2: degrainp[i] = degrainPlaneSSE2<2, false, uint16_t>; break;
                        case 3: degrainp[i] = degrainPlaneSSE2<3, false, uint16_t>; break;
                        case 4: degrainp[i] = degrainPlaneSSE2<4, false, uint16_t>; break;
                        case 5: degrainp[i] = degrainPlaneSSE2<5, false, uint16_t>; break;
                    }
                }
            }
        }
        else
        {
            if (vi.BitsPerComponent() == 8)
            {
                if (norow)
                {
                    switch (mode[i])
                    {
                        case 0: degrainp[i] = degrainPlaneScalar<0, true, uint8_t>; break;
                        case 1: degrainp[i] = degrainPlaneScalar<1, true, uint8_t>; break;
                        case 2: degrainp[i] = degrainPlaneScalar<2, true, uint8_t>; break;
                        case 3: degrainp[i] = degrainPlaneScalar<3, true, uint8_t>; break;
                        case 4: degrainp[i] = degrainPlaneScalar<4, true, uint8_t>; break;
                        case 5: degrainp[i] = degrainPlaneScalar<5, true, uint8_t>; break;
                    }
                }
                else
                {
                    switch (mode[i])
                    {
                        case 0: degrainp[i] = degrainPlaneScalar<0, false, uint8_t>; break;
                        case 1: degrainp[i] = degrainPlaneScalar<1, false, uint8_t>; break;
                        case 2: degrainp[i] = degrainPlaneScalar<2, false, uint8_t>; break;
                        case 3: degrainp[i] = degrainPlaneScalar<3, false, uint8_t>; break;
                        case 4: degrainp[i] = degrainPlaneScalar<4, false, uint8_t>; break;
                        case 5: degrainp[i] = degrainPlaneScalar<5, false, uint8_t>; break;
                    }
                }
            }
            else
            {
                if (norow)
                {
                    switch (mode[i])
                    {
                        case 0: degrainp[i] = degrainPlaneScalar<0, true, uint16_t>; break;
                        case 1: degrainp[i] = degrainPlaneScalar<1, true, uint16_t>; break;
                        case 2: degrainp[i] = degrainPlaneScalar<2, true, uint16_t>; break;
                        case 3: degrainp[i] = degrainPlaneScalar<3, true, uint16_t>; break;
                        case 4: degrainp[i] = degrainPlaneScalar<4, true, uint16_t>; break;
                        case 5: degrainp[i] = degrainPlaneScalar<5, true, uint16_t>; break;
                    }
                }
                else
                {
                    switch (mode[i])
                    {
                        case 0: degrainp[i] = degrainPlaneScalar<0, false, uint16_t>; break;
                        case 1: degrainp[i] = degrainPlaneScalar<1, false, uint16_t>; break;
                        case 2: degrainp[i] = degrainPlaneScalar<2, false, uint16_t>; break;
                        case 3: degrainp[i] = degrainPlaneScalar<3, false, uint16_t>; break;
                        case 4: degrainp[i] = degrainPlaneScalar<4, false, uint16_t>; break;
                        case 5: degrainp[i] = degrainPlaneScalar<5, false, uint16_t>; break;
                    }
                }
            }
        }
    }

    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; };
}

PVideoFrame __stdcall vsDeGrainMedian::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame prev = child->GetFrame(n - 1, env);
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame next = child->GetFrame(n + 1, env);
    PVideoFrame dst = (has_at_least_v8) ? env->NewVideoFrameP(vi, &src) : env->NewVideoFrame(vi);

    const int pixel_max = (1 << vi.BitsPerComponent()) - 1;
    const int planes_y[3] = { PLANAR_Y, PLANAR_U, PLANAR_V };
    const int planes_r[3] = { PLANAR_R, PLANAR_G, PLANAR_B };
    const int* planes = (vi.IsRGB()) ? planes_r : planes_y;
    const int planecount = std::min(vi.NumComponents(), 3);

    for (int i = 0; i < planecount; ++i)
    {
        const int stride = src->GetPitch(planes[i]);
        const int dst_stride = dst->GetPitch(planes[i]);
        const int width = src->GetRowSize(planes[i]);
        const int height = src->GetHeight(planes[i]);
        const uint8_t* srcp = src->GetReadPtr(planes[i]);
        uint8_t* dstp = dst->GetWritePtr(planes[i]);

        if (limit[i] == 0)
        {
            env->BitBlt(dstp, dst_stride, srcp, stride, width, height);

            continue;
        }

        const uint8_t* prevp = prev->GetReadPtr(planes[i]);
        const uint8_t* nextp = next->GetReadPtr(planes[i]);

        int _limit = pixel_max * limit[i] / 255;

        degrainp[i](prevp, srcp, nextp, dstp, stride, dst_stride, width, height, _limit, _interlaced, pixel_max);
    }

    return dst;
}

AVSValue __cdecl Create_vsDeGrainMedian(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    const int limitY = args[1].AsInt(4);
    const int limitU = args[2].AsInt(limitY);
    const int modeY = args[4].AsInt(1);
    const int modeU = args[5].AsInt(modeY);

    return new vsDeGrainMedian(
        args[0].AsClip(),
        limitY,
        limitU,
        args[3].AsInt(limitU),
        modeY,
        modeU,
        args[6].AsInt(modeU),
        args[7].AsBool(false),
        args[8].AsBool(false),
        args[9].AsInt(-1),
        env);
}

const AVS_Linkage* AVS_linkage;

extern "C" __declspec(dllexport)
const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

    env->AddFunction("vsDeGrainMedian", "c[limitY]i[limitU]i[limitV]i[modeY]i[modeU]i[modeV]i[interlaced]b[norow]b[opt]i", Create_vsDeGrainMedian, 0);

    return "vsDeGrainMedian";
}
