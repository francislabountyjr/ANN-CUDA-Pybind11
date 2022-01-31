#pragma once

#include <assert.h>

#include "layer.cuh"

namespace cudl
{
	class Softmax : public Layer
	{
	public:
		Softmax(std::string name)
		{
			name_ = name;
		};

		virtual ~Softmax() {};

		Softmax* call(std::vector<Layer*> layers) {
			if (layers.size() != 1)
				throw std::invalid_argument("Softmax layer must accept only 1 input.");
			inbound_layers_.push_back(layers[0]);
			layers[0]->add_outbound(this);
			return this;
		}

		Softmax* call(Layer* layer) {
			inbound_layers_.push_back(layer);
			layer->add_outbound(this);
			return this;
		}

		Softmax* call() {
			return this;
		}

		virtual void forward()
		{
#if (DEBUG_SOFTMAX & 0x01)
			std::cout << name_ << "[FORWARD]\n";
			input_->print(name_ + "::input", true, input->n());
#endif // DEBUG_SOFTMAX

			checkCudnnErrors(cudnnSoftmaxForward(cuda_->cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
				&cuda_->one, input_desc_, input_->cuda(),
				&cuda_->zero, output_desc_, output_->cuda()));

#if (DEBUG_SOFTMAX & 0x01)
			output_->print(name_ + "::output", true, input->n());
#endif // DEBUG_SOFTMAX
		};

		virtual void backward()
		{
			/*checkCudnnErrors(cudnnSoftmaxBackward(cuda_->cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
				&cuda_->one,
				output_desc_, output_->cuda(),
				output_desc_, grad_output_->cuda(),
				&cuda_->zero,
				input_desc_, grad_input_->cuda()));*/
			// set grad_input_ as predict
			checkCudaErrors(cudaMemcpyAsync(grad_input_->cuda(), output_->cuda(), output_->buf_size(), cudaMemcpyDeviceToDevice));

			// set grad_input_ = predict - target (grad_output_)
			checkCublasErrors(cublasSaxpy(cuda_->cublas(), grad_output_->len(),
				&cuda_->minus_one, grad_output_->cuda(), 1,
				grad_input_->cuda(), 1));

			// normalize grad_input_ by the batch size
			int grad_output_size = grad_output_->n() * grad_output_->c() * grad_output_->h() * grad_output_->w();
			float scale = 1.f / static_cast<float>(grad_output_->n());

			checkCublasErrors(cublasSscal(cuda_->cublas(), grad_output_size, &scale, grad_input_->cuda(), 1));

#if (DEBUG_SOFTMAX & 0x02)
			std::cout << name_ << "[BACKWARD]\n";
			input_->print(name_ + "::input", true);
			output_->print(name_ + "::predict", true);
			target->print(name_ + "::y", true, target->n());
			grad_input_->print(name_ + "::dx", true, target->n());
#endif // DEBUG_SOFTMAX
		};

		float get_loss(Tensor<float>* target)
		{
			return loss_.loss(output_, target);
		};

		int get_accuracy(Tensor<float>* target)
		{
			int batch_size = output_->n();
			int output_size = output_->size();

			assert(batch_size == target->n());
			assert(output_size == target->size());

			float* h_output, * h_target;
			int idx_output, idx_target;
			int hit_count = 0;

			// get predictions and targets
			h_output = output_->to(host);
			h_target = target->to(host);

			for (int b = 0; b < batch_size; b++)
			{
				idx_output = 0;
				idx_target = 0;

				for (int i = 1; i < 10; i++)
				{
					if (h_output[b * output_size + i] > h_output[b * output_size + idx_output])
					{
						idx_output = i;
					}

					if (h_target[b * output_size + i] > h_target[b * output_size + idx_target])
					{
						idx_target = i;
					}
				}

				if (idx_output == idx_target)
				{
					hit_count++;
				}
			}

			return hit_count;
		};

		// operators
		virtual std::string repr() const
		{
			return name_;
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
			if (grad_input_ == nullptr || batch_size_ != grad_output_->n())
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
		};

		CrossEntropyLoss loss_;
	};
}