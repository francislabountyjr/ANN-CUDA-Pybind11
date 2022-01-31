#include "cuda_functions.cuh"

__device__ float clip(float prediction, float epsilon)
{
	return fmin(fmax(prediction, epsilon), 1.f - epsilon);
}

__global__ void softmax_loss_kernel(float* reduced_loss, float* predict, float* target, float* workspace, int batch_size, int num_outputs)
{
	int batch_idx = blockDim.x * blockIdx.x + threadIdx.x;

	extern __shared__ float s_data[];
	float loss = 0.f;

	// each thread calculates entropy and accumulates to shared memory
	for (int c = 0; c < num_outputs; c++)
	{
		loss += target[batch_idx * num_outputs + c] * logf(predict[batch_idx * num_outputs + c]);
	}
	workspace[batch_idx] = -loss;

	// then, do reduction on the result to calculate the loss using 1 thread block
	if (blockIdx.x > 0)
	{
		return;
	}

	// cumulate workspace data
	s_data[threadIdx.x] = 0.f;
	for (int i = 0; i < batch_size; i += blockDim.x)
	{
		s_data[threadIdx.x] += workspace[threadIdx.x + i];
	}

	__syncthreads();

	// reduction
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (threadIdx.x + stride < batch_size)
		{
			s_data[threadIdx.x] += s_data[threadIdx.x + stride];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		reduced_loss[blockIdx.x] = s_data[0];
	}
}

__global__ void init_one_vec(float* d_one_vec, size_t length)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < length)
	{
		d_one_vec[i] = 1.f;
	}
}

__global__ void concat2tensors(const float* in1, size_t numElems1, const float* in2, size_t numElems2, float* out, size_t totalElems, size_t numElemsPerBatch) // numElemsPerBatch of out
{
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < totalElems; i += blockDim.x * gridDim.x)
	{
		int batchIdx = i / numElemsPerBatch;
		int batchOffset = i - batchIdx * numElemsPerBatch;
		out[i] = (batchOffset < numElems1) ? in1[batchOffset + batchIdx * numElems1] : in2[(batchOffset - numElems1) + batchIdx * numElems2];
	}
}

__global__ void concat2tensors_backward(float* in1, size_t numElems1, float* in2, size_t numElems2, const float* out, size_t totalElems, size_t numElemsPerBatch)
{
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < totalElems; i += blockDim.x * gridDim.x)
	{
		int batchIdx = i / numElemsPerBatch;
		int batchOffset = i - batchIdx * numElemsPerBatch;
		if (batchOffset < numElems1)
			in1[batchOffset + batchIdx * numElems1] = out[i];
		else
			in2[(batchOffset - numElems1) + batchIdx * numElems2] = out[i];
	}
}

void run_softmax_loss_kernel(int num_blocks, float* reduced_loss, float* predict, float* target, float* workspace, int batch_size, int num_outputs)
{
	softmax_loss_kernel<<<num_blocks, BLOCK_DIM_1D, BLOCK_DIM_1D * sizeof(float), 0>>>(reduced_loss, predict, target, workspace, batch_size, num_outputs);
}

void run_init_one_vec(float* d_one_vec, size_t length)
{
	init_one_vec<<<(length + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D, BLOCK_DIM_1D>>>(d_one_vec, length);
}

void run_concat2tensors(const float* in1, size_t numElems1, const float* in2, size_t numElems2, float* out, size_t totalElems, size_t numElemsPerBatch)
{
	int num_sms;
	int num_blocks_per_sm;

	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, softmax_loss_kernel, BLOCK_DIM_1D, 0);

	int num_blocks = std::min(num_blocks_per_sm * num_sms, (int)(totalElems + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D);

	concat2tensors<<<num_blocks, BLOCK_DIM_1D>>>(in1, numElems1, in2, numElems2, out, totalElems, numElemsPerBatch);
	cudaDeviceSynchronize();
}

void run_concat2tensors_backward(float* in1, size_t numElems1, float* in2, size_t numElems2, const float* out, size_t totalElems, size_t numElemsPerBatch)
{
	int num_sms;
	int num_blocks_per_sm;

	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, softmax_loss_kernel, BLOCK_DIM_1D, 0);

	int num_blocks = std::min(num_blocks_per_sm * num_sms, (int)(totalElems + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D);

	concat2tensors_backward<<<num_blocks, BLOCK_DIM_1D>>>(in1, numElems1, in2, numElems2, out, totalElems, numElemsPerBatch);
	cudaDeviceSynchronize();
}