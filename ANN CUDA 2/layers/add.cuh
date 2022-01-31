#pragma once

#include "layer.cuh"

namespace cudl
{
	class Add : public Layer
	{
	public:
		Add(std::string name, float alpha = 1.f, float beta = 1.f)
		{
			name_ = name;
			alpha_ = alpha;
			beta_ = beta;
		};

		virtual ~Add()
		{
		};

		Add* call(std::vector<Layer*> layers) {
			if (layers.size() != 2)
				throw std::invalid_argument("Add layer must accept 2 inputs.");
			inbound_layers_.push_back(layers[0]);
			inbound_layers_.push_back(layers[1]);
			layers[0]->add_outbound(this);
			layers[1]->add_outbound(this);
			return this;
		}

		Add* call(Layer* layer) {
			throw std::invalid_argument("Add layer must accept 2 inputs.");
		}

		Add* call() {
			return this;
		}

		virtual void forward()
		{
			checkCudnnErrors(cudnnAddTensor(cuda_->cudnn(),
				&alpha_,
				input_desc_,
				input_->cuda(),
				&beta_,
				output_desc_,
				output_->cuda()
			));
		};

		virtual void backward()
		{
		};

		// operators
		virtual std::string repr() const
		{
			return name_ + "\t Function: (" + std::to_string(alpha_) + "*" + inbound_layers_[0]->get_name() + " + " + std::to_string(beta_) + "*" + inbound_layers_[1]->get_name();
		}

	private:
		void fwd_initialize()
		{
			input_ = inbound_layers_[0]->get_output();
			if (output_ == nullptr || batch_size_ != input_->n())
			{
				input_desc_ = input_->tensor();
				batch_size_ = input_->n();

				if (input_->size() != inbound_layers_[1]->get_output()->size())
				{
					py::print(inbound_layers_[0]->get_name(), " shape: ", input_->shape());
					py::print(inbound_layers_[1]->get_name(), " shape: ", inbound_layers_[1]->get_output()->shape());
					throw std::invalid_argument("Add layer input tensor dimensions must be equal.");
				}

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

			output_->copy_cuda(inbound_layers_[1]->get_output());
		};

		void bwd_initialize()
		{
			if (grad_input_ == nullptr || batch_size_ != outbound_layers_[0]->get_grad_input(index_)->n())
			{
				if (grad_input_ == nullptr)
				{
					grad_input_ = new Tensor<float>(input_->shape());
				}
				else
				{
					grad_input_->reset(input_->shape());
				}
			}

			if (outbound_layers_.size() == 1)
				grad_input_ = outbound_layers_[0]->get_grad_input(index_); // gradient is passed down so set grad_input_ directly
			else { // only edit grad_input_ if copied to avoid affecting other layers' gradient calculations
				grad_input_->copy_cuda(outbound_layers_[0]->get_grad_input(index_));
				for (int i = 1; i < outbound_layers_.size(); i++) {
					checkCudnnErrors(cudnnAddTensor(cuda_->cudnn(),
						&cuda_->one,
						output_desc_,
						outbound_layers_[i]->get_grad_input(index_)->cuda(),
						&cuda_->one,
						output_desc_,
						grad_input_->cuda()
					));
				}
			}

			grad_output_ = grad_input_;
		};

		float alpha_;
		float beta_;
	};
}