#pragma once

#include "avisynth.h"

template <int mode, bool norow, typename PixelType>
void degrainPlaneSSE2(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8, const uint8_t* AVS_RESTRICT nextp8,
    uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride, int width, int height, int limit,
    int interlaced, int pixel_max);

class vsDeGrainMedian : public GenericVideoFilter
{
    int limit[3];
    bool _interlaced;
    bool has_at_least_v8;

    void (*degrainp[3])(const uint8_t* AVS_RESTRICT prevp8, const uint8_t* AVS_RESTRICT srcp8, const uint8_t* AVS_RESTRICT nextp8,
        uint8_t* AVS_RESTRICT dstp8, int prev_stride, int src_stride, int next_stride, int dst_stride, int width, int height, int limit,
        int interlaced, int pixel_max);

public:
    vsDeGrainMedian(PClip _child, int limitY, int limitU, int limitV, int modeY, int modeU, int modeV, bool interlaced, bool norow, int opt, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
    int __stdcall SetCacheHints(int cachehints, int frame_range)
    {
        return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
};
