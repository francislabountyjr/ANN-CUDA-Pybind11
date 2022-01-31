#pragma once

#include "layer.cuh"

namespace cudl
{
	class Pooling : public Layer
	{
	public:
		Pooling(std::string name, int kernel_size, int padding, int stride, cudnnPoolingMode_t mode) :
			kernel_size_(kernel_size),
			padding_(padding),
			stride_(stride),
			mode_(mode)
		{
			name_ = name;

			cudnnCreatePoolingDescriptor(&pool_desc_);
			cudnnSetPooling2dDescriptor(pool_desc_, mode_, CUDNN_PROPAGATE_NAN,
				kernel_size_, kernel_size_, padding_, padding_, stride_, stride_);
		};

		virtual ~Pooling()
		{
			cudnnDestroyPoolingDescriptor(pool_desc_);
		};

		Pooling* call(std::vector<Layer*> layers) {
			if (layers.size() != 1)
				throw std::invalid_argument("Pooling layer must accept only 1 input.");
			inbound_layers_.push_back(layers[0]);
			layers[0]->add_outbound(this);
			return this;
		}

		Pooling* call(Layer* layer) {
			inbound_layers_.push_back(layer);
			layer->add_outbound(this);
			return this;
		}

		Pooling* call() {
			return this;
		}

		virtual void forward()
		{
			cudnnPoolingForward(cuda_->cudnn(), pool_desc_,
				&cuda_->one,
				input_desc_, input_->cuda(),
				&cuda_->zero,
				output_desc_, output_->cuda());
		};

		virtual void backward()
		{
			checkCudnnErrors(cudnnPoolingBackward(cuda_->cudnn(), pool_desc_,
				&cuda_->one,
				output_desc_, output_->cuda(),
				output_desc_, grad_output_->cuda(),
				input_desc_, input_->cuda(),
				&cuda_->zero,
				input_desc_, grad_input_->cuda()));
		};

		// operators
		virtual std::string repr() const
		{
			return name_ + "\tPooling Mode: " + std::to_string(mode_);
		}

	private:
		void fwd_initialize()
		{
			input_ = inbound_layers_[0]->get_output();
			if (output_ == nullptr || batch_size_ != input_->n())
			{
				// resource initialization
				input_desc_ = input_->tensor();
				batch_size_ = input_->n();

				// setting output
				cudnnGetPooling2dForwardOutputDim(pool_desc_, input_desc_,
					&output_size_[0], &output_size_[1], &output_size_[2], &output_size_[3]);

				if (output_ == nullptr)
				{
					output_ = new Tensor<float>(output_size_);
				}
				else
				{
					output_->reset(output_size_);
				}

				output_desc_ = output_->tensor();
			}
		};

		void bwd_initialize()
		{
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

		int kernel_size_;
		int padding_;
		int stride_;

		cudnnPoolingMode_t mode_;

		std::array<int, 4> output_size_;

		cudnnPoolingDescriptor_t pool_desc_;
	};
}