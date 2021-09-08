## Description

DeGrainMedian is a spatio-temporal limited median denoiser. It uses various methods to replace every pixel with one selected from its 3x3 neighbourhood, from either the current, previous, or next frame.

The first column and the last column are simply copied from the source frame. The first row and the last row are also copied from the source frame. If interlaced=True, then the second row and the second-to-last row are also copied from the source frame.

This is [a port of the VapourSynth plugin DegrainMedian](https://github.com/dubhater/vapoursynth-degrainmedian).

### Requirements:

- AviSynth 2.60 / AviSynth+ 3.4 or later

- Microsoft VisualC++ Redistributable Package 2022 (can be downloaded from [here](https://github.com/abbodi1406/vcredist/releases))

### Usage:

```
vsDeGrainMedian (clip input, int "limitY", int "limitU", int "limitV", int "modeY", int "modeU", int "modeV", bool "interlaced", bool "norow", int "opt")
```

### Parameters:

- input\
    A clip to process.\
    It must be in 8..16-bit planar format.
    
- limitY, limitU, limitV\
    Limits how much a pixel is changed. Each new pixel will be in the range \[old pixel - limit, old pixel + limit].\
    limitX = 0: plane will be copied from the input frame.\
    Must be between 0..255.\
    Default: limitY = 4; limitU = limitY; limitV = limitU.
    
- modeY, modeU, modeV\
    Processing mode.\
    Mode 0 is the strongest.\
    Mode 5 is the weakest.\
    Default: modeY = 1; modeU = modeY; modeV = modeU.

- interlaced\
    If True, the top line and the bottom line of the 3x3 neighbourhood will come from the same field as the middle line. In other words, one line will be skipped between the top line and the middle line, and between the middle line and the bottom line.\
    This parameter should only be used when the input clip contains interlaced video.\
    Default: false.
    
- norow\
    If True, the two pixels to the left and right of the original pixel will not be used in the calculations. The corresponding pixels from the previous and next frames are still used.\
    Default: false.
    
- opt\
    Sets which cpu optimizations to use.\
    -1: Auto-detect.\
    0: Use C++ code.\
    1: Use SSE2 code.\
    Default: -1.
    
### Building:

- Windows\
    Use solution files.

- Linux
    ```
    Requirements:
        - Git
        - C++11 compiler
        - CMake >= 3.16
    ```
    ```
    git clone https://github.com/Asd-g/AviSynth-vsDeGrainMedian && \
    cd AviSynth-vsDeGrainMedian && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    sudo make install
    ```
