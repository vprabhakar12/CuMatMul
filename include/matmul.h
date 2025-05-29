#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void launch_naive(float* A, float* B, float* C, int N);
void launch_tiled(float* A, float* B, float* C, int N);
#ifdef __cplusplus
}
#endif
