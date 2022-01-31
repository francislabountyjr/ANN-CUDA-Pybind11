#pragma once

#include "layer.cuh"

namespace cudl
{
	class Activation : public Layer
	{
	public:
		Activation(std::string name, cudnnActivationMode_t mode, float coef = 0.f)
		{
			name_ = name;
			act_mode_ = mode;
			act_coef_ = coef;

			cudnnCreateActivationDescriptor(&act_desc_);
			cudnnSetActivationDescriptor(act_desc_, act_mode_, CUDNN_PROPAGATE_NAN, act_coef_);
		};

		virtual ~Activation()
		{
			cudnnDestroyActivationDescriptor(act_desc_);
		};

		Activation* call(std::vector<Layer*> layers) {
			if (layers.size() != 1)
				throw std::invalid_argument("Activation layer must accept only 1 input.");
			inbound_layers_.push_back(layers[0]);
			layers[0]->add_outbound(this);
			return this;
		}

		Activation* call(Layer* layer) {
			inbound_layers_.push_back(layer);
			layer->add_outbound(this);
			return this;
		}

		Activation* call() {
			return this;
		}

		virtual void forward()
		{
			cudnnActivationForward(
				cuda_->cudnn(),
				act_desc_,
				&cuda_->one,
				input_desc_, input_->cuda(),
				&cuda_->zero,
				output_desc_, output_->cuda()
			);
		};

		virtual void backward()
		{
			cudnnActivationBackward(
				cuda_->cudnn(),
				act_desc_,
				&cuda_->one,
				output_desc_, output_->cuda(),
				output_desc_, grad_output_->cuda(),
				input_desc_, input_->cuda(),
				&cuda_->zero,
				input_desc_, grad_input_->cuda()
			);
		};

		// operators
		virtual std::string repr() const
		{
			return name_ + "\tActivation Mode: " + std::to_string(act_mode_);
		}

	private:
		void fwd_initialize()
		{
			input_ = inbound_layers_[0]->get_output();
			if (output_ == nullptr || batch_size_ != input_->n())
			{
				input_desc_ = input_->tensor();
				batch_size_ = input_->n();

				if (output_ == nullptr)
				{
					output_ = new Tensor<float>(input_->shape());
				}
				else
				{
					output_->reset(input_->shape());
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

		cudnnActivationDescriptor_t act_desc_;
		cudnnActivationMode_t act_mode_;
		float act_coef_;
	};
}