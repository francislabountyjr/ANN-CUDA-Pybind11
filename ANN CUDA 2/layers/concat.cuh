#pragma once

#include "layer.cuh"
#include "../utility/cuda_functions.cuh"

namespace cudl
{
	class Concat : public Layer
	{
	public:
		Concat(std::string name)
		{
			// Concatenates two layers by the channel dimension
			name_ = name;
		};

		virtual ~Concat()
		{
		};

		Concat* call(std::vector<Layer*> layers) {
			if (layers.size() != 2)
				throw std::invalid_argument("Add layer must accept 2 inputs.");
			inbound_layers_.push_back(layers[0]);
			inbound_layers_.push_back(layers[1]);
			layers[0]->add_outbound(this);
			layers[1]->add_outbound(this);
			return this;
		}

		Concat* call(Layer* layer) {
			throw std::invalid_argument("Add layer must accept 2 inputs.");
		}

		Concat* call() {
			return this;
		}

		virtual void forward()
		{
			run_concat2tensors(input_->cuda(), input_->len(), input2_->cuda(), input2_->len(), output_->cuda(), output_->len(), output_->size());
		};

		virtual void backward()
		{
			run_concat2tensors_backward(grad_input_map_[inbound_layers_[0]->index_]->cuda(), grad_input_map_[inbound_layers_[0]->index_]->len(), grad_input_map_[inbound_layers_[1]->index_]->cuda(), grad_input_map_[inbound_layers_[1]->index_]->len(), grad_output_->cuda(), grad_output_->len(), grad_output_->size());
		};

		// operators
		virtual std::string repr() const
		{
			return name_ + "\t Concat: (" + inbound_layers_[0]->get_name() + " + " + inbound_layers_[1]->get_name();
		}

		Tensor<float>* get_grad_input(int index = 0) {
			return grad_input_map_[index];
		}

		std::vector<Tensor<float>*> get_grad_inputs() {
			std::vector<Tensor<float>*> out;
			out.reserve(grad_input_map_.size());

			for (auto kv : grad_input_map_)
				out.push_back(kv.second);

			return out;
		}

	private:
		void fwd_initialize()
		{
			input_ = inbound_layers_[0]->get_output();
			input2_ = inbound_layers_[1]->get_output();
			if (output_ == nullptr || batch_size_ != input_->n())
			{
				//input_desc_ = input_->tensor();
				//input_desc2_ = input2_->tensor();
				batch_size_ = input_->n();

				if (input_->h() != input2_->h() || input_->w() != input2_->w())
					throw std::invalid_argument("Cannot concat layers with unequal h and w dimensions.");

				if (output_ == nullptr)
				{
					output_ = new Tensor<float>(input_->n(), input_->c() + input2_->c(), input_->h(), input_->w());
				}
				else
				{
					output_->reset(input_->n(), input_->c() + input2_->c(), input_->h(), input_->w());
				}

				output_desc_ = output_->tensor();
			}
		};

		void bwd_initialize()
		{
			if (grad_input_map_.size() == 0 || batch_size_ != outbound_layers_[0]->get_grad_input(index_)->n())
			{
				if (grad_input_map_.size() == 0)
				{
					grad_input_map_[inbound_layers_[0]->index_] = new Tensor<float>(input_->shape());
					grad_input_map_[inbound_layers_[1]->index_] = new Tensor<float>(input2_->shape());
					if (outbound_layers_.size() > 1)
						grad_output_ = new Tensor<float>(output_->shape());
				}
				else
				{
					grad_input_map_[inbound_layers_[0]->index_]->reset(input_->shape());
					grad_input_map_[inbound_layers_[1]->index_]->reset(input2_->shape());
					if (outbound_layers_.size() > 1)
						grad_output_->reset(output_->shape());
				}
			}
			
			if (outbound_layers_.size() == 1) {
				grad_output_ = outbound_layers_[0]->get_grad_input(index_); // gradient is passed down so set grad_input_ directly
			}
			else { // only edit grad_input_ if copied to avoid affecting other layers' gradient calculations
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

		Tensor<float>* input2_;
		std::unordered_map<int, Tensor<float>*> grad_input_map_;
	};
}