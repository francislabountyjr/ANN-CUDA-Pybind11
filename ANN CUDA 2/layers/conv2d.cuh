#pragma once

#include <vector>

#include "layer.cuh"

namespace cudl
{
	class Conv2D : public Layer
	{
	public:
		Conv2D(std::string name, int out_channels, int kernel_size, int stride = 1, int padding = 0, int dilation = 1) :
			out_channels_(out_channels),
			kernel_size_(kernel_size),
			stride_(stride),
			padding_(padding),
			dilation_(dilation)
		{
			name_ = name;

			// create cudnn container handles
			cudnnCreateFilterDescriptor(&filter_desc_);

			cudnnCreateConvolutionDescriptor(&conv_desc_);
			checkCudnnErrors(cudnnSetConvolution2dDescriptor(conv_desc_, padding_, padding_,
				stride_, stride_, dilation_, dilation_, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

			// set cudnn convolution math type
			checkCudnnErrors(cudnnSetConvolutionMathType(conv_desc_, CUDNN_DEFAULT_MATH));

			d_workspace_ = nullptr;
		};

		virtual ~Conv2D()
		{
			// destroy cudnn container resources
			cudnnDestroyFilterDescriptor(filter_desc_);
			cudnnDestroyConvolutionDescriptor(conv_desc_);

			// terminate internal tensors
			if (d_workspace_ != nullptr)
			{
				cudaFree(d_workspace_);
				d_workspace_ = nullptr;
			}
		};

		Conv2D* call(std::vector<Layer*> layers) {
			if (layers.size() != 1)
				throw std::invalid_argument("Conv2D layer must accept only 1 input.");
			inbound_layers_.push_back(layers[0]);
			layers[0]->add_outbound(this);
			return this;
		}

		Conv2D* call(Layer* layer) {
			inbound_layers_.push_back(layer);
			layer->add_outbound(this);
			return this;
		}

		Conv2D* call() {
			return this;
		}

		virtual void forward()
		{
			checkCudnnErrors(cudnnConvolutionForward(cuda_->cudnn(),
				&cuda_->one, input_desc_, input_->cuda(),
				filter_desc_, weights_->cuda(), conv_desc_, conv_fwd_algo_, d_workspace_, workspace_size_,
				&cuda_->zero, output_desc_, output_->cuda()));

			checkCudnnErrors(cudnnAddTensor(cuda_->cudnn(),
				&cuda_->one, bias_desc_, biases_->cuda(),
				&cuda_->one, output_desc_, output_->cuda()));

#if (DEBUG_CONV & 0x01)
			input_->print(name_ + "::input", true, input_->n(), 28);
			weights_->print(name_ + "::weight", true);
			biases_->print(name_ + "::bias", true);
			output_->print(name_ + "::output", true);
#endif // DEBUG_CONV
		};

		virtual void backward()
		{
			// gradients of biases
			checkCudnnErrors(cudnnConvolutionBackwardBias(cuda_->cudnn(),
				&cuda_->one,
				output_desc_, grad_output_->cuda(),
				&cuda_->zero,
				bias_desc_, grad_biases_->cuda()));

			// gradients of weights
			checkCudnnErrors(cudnnConvolutionBackwardFilter(cuda_->cudnn(),
				&cuda_->one,
				input_desc_, input_->cuda(),
				output_desc_, grad_output_->cuda(),
				conv_desc_, conv_bwd_filter_algo_, d_workspace_, workspace_size_,
				&cuda_->zero,
				filter_desc_, grad_weights_->cuda()));

			// gradients of input data
			if (!gradient_stop_)
			{
				checkCudnnErrors(cudnnConvolutionBackwardData(cuda_->cudnn(),
					&cuda_->one,
					filter_desc_, weights_->cuda(),
					output_desc_, grad_output_->cuda(),
					conv_desc_, conv_bwd_data_algo_, d_workspace_, workspace_size_,
					&cuda_->zero,
					input_desc_, grad_input_->cuda()));
			}

#if (DEBUG_CONV & 0x02)
			std::cout << name_ << "[BACKWARD]\n";
			grad_output->print(name_ + "::gradients", true;
			grad_weights_->print(name_ + "::gfilter", true);
			grad_biases_->print(name_ + "::gbias", true);
			if (!gradient_stop_)
			{
				grad_input_->print(name_ + "::gdata", true);
			}
#endif // DEBUG_CONV

#if (DEBUG_CONV & 0x04)
			std::cout << name_ << "[BACKWARD]\n";
			grad_output->print(name_ + "::gradients", true;
			grad_weights_->print(name_ + "::gfilter", true);
			grad_biases_->print(name_ + "::gbias", true);
			if (!gradient_stop_)
			{
				grad_input_->print(name_ + "::gdata", true);
			}
#endif // DEBUG_CONV
		};

		// operators
		virtual std::string repr() const
		{
			return name_ + "\tOutput Channels: " + std::to_string(out_channels_);
		}

	private:
		void fwd_initialize()
		{
			input_ = inbound_layers_[0]->get_output();
			// initialize weights and bias
			if (weights_ == nullptr)
			{
				// initialize containers handles

				checkCudnnErrors(cudnnSetFilter4dDescriptor(filter_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
					out_channels_, input_->c(), kernel_size_, kernel_size_));

				weights_ = new Tensor<float>(out_channels_, input_->c(), kernel_size_, kernel_size_);
				biases_ = new Tensor<float>(1, out_channels_);
				bias_desc_ = biases_->tensor();
			}

			// initialize input and output
			if (output_ == nullptr || batch_size_ != input_->n())
			{
				input_desc_ = input_->tensor();
				batch_size_ = input_->n();

				// initialize output
				checkCudnnErrors(cudnnGetConvolution2dForwardOutputDim(conv_desc_, input_desc_, filter_desc_,
					&output_size_[0], &output_size_[1], &output_size_[2], &output_size_[3]));

				if (output_ == nullptr)
				{
					output_ = new Tensor<float>(output_size_);
				}
				else
				{
					output_->reset(output_size_);
				}

				output_desc_ = output_->tensor();

				// initialize workspace for cudnn
				set_workspace();

				// initialize weights
				if (load_pretrain_ && !freeze_)
				{
					if (load_parameter())
					{
						py::print("error occured loading weights for ", name_, '\n');
						exit(-1);
					}
				}
				else if (!freeze_)
				{
					init_weight_bias();
				}
				else
				{
					// do nothing
				}
			}
		};

		void bwd_initialize()
		{
			if (grad_weights_ == nullptr)
			{
				grad_weights_ = new Tensor<float>(weights_->shape());
				grad_biases_ = new Tensor<float>(1, biases_->c());
			}

			// initialize grad_output back-propagation space
			if (grad_input_ == nullptr || batch_size_ != outbound_layers_[0]->get_grad_input(index_)->n())
			{
				if (grad_input_ == nullptr)
				{
					grad_input_ = new Tensor<float>(input_->shape());
					if (outbound_layers_.size() > 1)
						grad_output_ = new Tensor<float>(output_->shape());
				}
				else
				{
					grad_input_->reset(input_->shape());
					if (outbound_layers_.size() > 1)
						grad_output_->reset(output_->shape());
				}
			}

			if (outbound_layers_.size() == 1)
				grad_output_ = outbound_layers_[0]->get_grad_input(index_);
			else { // only edit grad_output_ if copied to avoid affecting other layers' gradient calculations
				grad_output_->copy_cuda(outbound_layers_[0]->get_grad_input(index_));
				for (int i = 1; i < outbound_layers_.size(); i++) {
					checkCudnnErrors(cudnnAddTensor(cuda_->cudnn(),
						&cuda_->one,
						output_desc_,
						outbound_layers_[i]->get_grad_input(index_)->cuda(),
						&cuda_->one,
						output_desc_,
						grad_output_->cuda()
					));
				}
			}
		};

		virtual void set_workspace()
		{
			size_t temp_size = 0;

			// forward
#if CUDNN_MAJOR >= 7
			std::vector<cudnnConvolutionFwdAlgoPerf_t> fwd_algo_perf_results(CUDNN_CONVOLUTION_FWD_ALGO_COUNT);
			std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filter_algo_perf_results(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT);
			std::vector<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_algo_perf_results(CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT);

			int algo_max_count;
			int returnedAlgoCount = 0;
			checkCudnnErrors(cudnnGetConvolutionForwardAlgorithmMaxCount(cuda_->cudnn(), &algo_max_count));
#if (DEBUG_FIND_ALGO & 1)
			std::cout << this->name_ << ": Available Algorithm Count [FWD]: " << algo_max_count << '\n';

			checkCudnnErrors(cudnnFindConvolutionForwardAlgorithm(cuda_->cudnn(),
				input_desc_, filter_desc_, conv_desc_, output_desc_,
				algo_max_count, &returnedAlgoCount, &fwd_algo_perf_results[0]));

			std::cout << "returned algo_count: " << returnedAlgoCount << '\n';

			for (int i = 0; i < returnedAlgoCount; i++)
			{
				std::cout << "fwd algo[" << i << "] time: " << fwd_algo_perf_results[i].time << ", memory: " << fwd_algo_perf_results[i].memory << '\n';
#else
			checkCudnnErrors(cudnnGetConvolutionForwardAlgorithm_v7(cuda_->cudnn(),
				input_desc_, filter_desc_, conv_desc_, output_desc_,
				algo_max_count, &returnedAlgoCount, &fwd_algo_perf_results[0]));
#endif
			// choose the fastest algorithm
			conv_fwd_algo_ = fwd_algo_perf_results[0].algo;
#else
			checkCudnnErrors(cudnnGetConvolutionForwardAlgorithm(cuda_->cudnn(),
				input_desc_, filter_desc_, conv_desc_, output_desc_,
				CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &conv_fwd_algo_));
#endif
			checkCudnnErrors(cudnnGetConvolutionForwardWorkspaceSize(cuda_->cudnn(),
				input_desc_, filter_desc_, conv_desc_, output_desc_,
				conv_fwd_algo_, &temp_size));

			workspace_size_ = std::max(workspace_size_, temp_size);

			// bwd - filter
#if CUDNN_MAJOR >= 7
			checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cuda_->cudnn(), &algo_max_count));
#if (DEBUG_FIND_ALGO & 1)
			std::cout << this->name_ << ": Available Algorithm Count [BWD-filter]: " << algo_max_count << '\n';

			checkCudnnErrors(cudnnFindConvolutionBackwardFilterAlgorithm(cuda_->cudnn(),
				input_desc_, output_desc_, conv_desc_, filter_desc_,
				algo_max_count, &returnedAlgoCount, &bwd_filter_algo_perf_results[0]));

			std::cout << "returned algo_count: " << returnedAlgoCount << '\n';

			for (int i = 0; i < returnedAlgoCount; i++)
			{
				std::cout << "bwd filter algo[" << i << "] time: " << bwd_filter_algo_perf_results[i].time << ", memory: " << bwd_filter_algo_perf_results[i].memory << '\n';
#else
			checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithm_v7(cuda_->cudnn(),
				input_desc_, output_desc_, conv_desc_, filter_desc_,
				algo_max_count, &returnedAlgoCount, &bwd_filter_algo_perf_results[0]));
#endif
			// choose the fastest algorithm
			conv_bwd_filter_algo_ = bwd_filter_algo_perf_results[0].algo;
#else
			checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithm(cuda_->cudnn(),
				input_desc_, output_desc_, conv_desc_, filter_desc_,
				CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &conv_bwd_filter_algo_));
#endif
			checkCudnnErrors(cudnnGetConvolutionBackwardFilterWorkspaceSize(cuda_->cudnn(),
				input_desc_, output_desc_, conv_desc_, filter_desc_,
				conv_bwd_filter_algo_, &temp_size));

			workspace_size_ = std::max(workspace_size_, temp_size);

			// bwd - data
#if CUDNN_MAJOR >= 7
			checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cuda_->cudnn(), &algo_max_count));
#if (DEBUG_FIND_ALGO & 1)
			std::cout << this->name_ << ": Available Algorithm Count [BWD-data]: " << algo_max_count << '\n';

			checkCudnnErrors(cudnnFindConvolutionBackwardDataAlgorithm(cuda_->cudnn(),
				filter_desc_, output_desc_, conv_desc_, input_desc_,
				algo_max_count, &returnedAlgoCount, &bwd_data_algo_perf_results[0]));

			std::cout << "returned algo_count: " << returnedAlgoCount << '\n';

			for (int i = 0; i < returnedAlgoCount; i++)
			{
				std::cout << "bwd data algo[" << i << "] time: " << bwd_data_algo_perf_results[i].time << ", memory: " << bwd_data_algo_perf_results[i].memory << '\n';
#else
			checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithm_v7(cuda_->cudnn(),
				filter_desc_, output_desc_, conv_desc_, input_desc_,
				algo_max_count, &returnedAlgoCount, &bwd_data_algo_perf_results[0]));
#endif
			// choose the fastest algorithm
			conv_bwd_data_algo_ = bwd_data_algo_perf_results[0].algo;
#else
			checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithm(cuda_->cudnn(),
				filter_desc_, output_desc_, conv_desc_, input_desc_,
				CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &conv_bwd_data_algo_));
#endif
			checkCudnnErrors(cudnnGetConvolutionBackwardDataWorkspaceSize(cuda_->cudnn(),
				filter_desc_, output_desc_, conv_desc_, input_desc_,
				conv_bwd_data_algo_, &temp_size));

			workspace_size_ = std::max(workspace_size_, temp_size);

			if (workspace_size_ > 0)
			{
				if (d_workspace_ != nullptr)
				{
					checkCudaErrors(cudaFree(d_workspace_));
				}

				checkCudaErrors(cudaMalloc((void**)&d_workspace_, workspace_size_));
			}
		};

		int out_channels_;
		int kernel_size_;
		int stride_;
		int padding_;
		int dilation_;

		std::array<int, 4> output_size_;

		// convolution
		cudnnConvolutionDescriptor_t conv_desc_;

		cudnnConvolutionFwdAlgo_t conv_fwd_algo_;
		cudnnConvolutionBwdDataAlgo_t conv_bwd_data_algo_;
		cudnnConvolutionBwdFilterAlgo_t conv_bwd_filter_algo_;

		size_t workspace_size_ = 0;
		float* d_workspace_ = nullptr;
	};
}