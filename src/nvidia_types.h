#ifndef NV_TYPES_H
#define NV_TYPES_H

#include <nvrtc.h>
#include <cuda.h>

const char *get_nvrtc_error_string(nvrtcResult result);
const char *get_cuda_error_string(CUresult result);

#endif