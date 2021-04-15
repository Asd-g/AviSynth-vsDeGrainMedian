#pragma once

#include "avisynth.h"

#ifdef _WIN32
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

template <int mode, bool norow, typename PixelType>
void degrainPlaneSSE2(const uint8_t* prevp8, const uint8_t* srcp8, const uint8_t* nextp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int limit, int interlaced, int pixel_max);

class vsDeGrainMedian : public GenericVideoFilter
{
    int limit[3];
    bool _interlaced;
    bool has_at_least_v8;

    void (*degrainp[3])(const uint8_t* prevp8, const uint8_t* srcp8, const uint8_t* nextp8, uint8_t* dstp8, int stride, int dst_stride, int width, int height, int limit, int interlaced, int pixel_max);

public:
    vsDeGrainMedian(PClip _child, int limitY, int limitU, int limitV, int modeY, int modeU, int modeV, bool interlaced, bool norow, int opt, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
    int __stdcall SetCacheHints(int cachehints, int frame_range)
    {
        return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
};
