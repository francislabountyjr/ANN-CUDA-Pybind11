#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_functions.cuh"
#include "tensor.cuh"
#include "helper.cuh"

namespace cudl
{
	class CrossEntropyLoss
	{
	public:
		CrossEntropyLoss()
		{
			cudaMalloc((void**)&d_loss_, sizeof(float));
		};

		~CrossEntropyLoss()
		{
			if (d_loss_ != nullptr)
			{
				cudaFree(d_loss_);
				d_loss_ = nullptr;
			}

			if (d_workspace_ != nullptr)
			{
				cudaFree(d_workspace_);
			}
		};

		float loss(Tensor<float>* predict, Tensor<float>* target)
		{
			int num_sms;
			int num_blocks_per_sm;

			cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
			cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, softmax_loss_kernel, BLOCK_DIM_1D, BLOCK_DIM_1D * sizeof(float));

			int batch_size = target->n();
			int num_outputs = target->c();

			init_workspace(batch_size);

#if (DEBUG_LOSS)
			std::cout << "[LOSS]\n";
			predict->print("predict", true);
			target->print("target", true);
#endif // DEBUG_LOSS

			int num_blocks = std::min(num_blocks_per_sm * num_sms, (target->size() + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D);

			run_softmax_loss_kernel(num_blocks, d_loss_, predict->cuda(), target->cuda(), d_workspace_, batch_size, num_outputs);

			cudaMemcpy(&h_loss_, d_loss_, sizeof(float), cudaMemcpyDeviceToHost);

			// batch mean loss
			return h_loss_ / float(batch_size);
		};
		//float accuracy(Tensor<float>* predict, Tensor<float>* target);

	private:
		// reduced loss
		float h_loss_ = 0.f;
		float* d_loss_ = nullptr;

		float* d_workspace_ = nullptr;
		void init_workspace(int batch_size)
		{
			if (d_workspace_ == nullptr)
			{
				cudaMalloc((void**)&d_workspace_, sizeof(float) * batch_size);
			}
		};
	};
}