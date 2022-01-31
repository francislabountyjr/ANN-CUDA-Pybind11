#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helper.cuh"

__device__ float clip(float prediction, float epsilon = 1e-12);

__global__ void softmax_loss_kernel(float* reduced_loss, float* predict, float* target, float* workspace, int batch_size, int num_outputs);
void run_softmax_loss_kernel(int num_blocks, float* reduced_loss, float* predict, float* target, float* workspace, int batch_size, int num_outputs);

__global__ void init_one_vec(float* d_one_vec, size_t length);
void run_init_one_vec(float* d_one_vec, size_t length);

__global__ void concat2tensors(const float* in1, size_t numElems1, const float* in2, size_t numElems2, float* out, size_t totalElems, size_t numElemsPerBatch);
void run_concat2tensors(const float* in1, size_t numElems1, const float* in2, size_t numElems2, float* out, size_t totalElems, size_t numElemsPerBatch);

__global__ void concat2tensors_backward(float* in1, size_t numElems1, float* in2, size_t numElems2, const float* out, size_t totalElems, size_t numElemsPerBatch);
void run_concat2tensors_backward(float* in1, size_t numElems1, float* in2, size_t numElems2, const float* out, size_t totalElems, size_t numElemsPerBatch);