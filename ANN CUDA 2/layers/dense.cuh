#pragma once

#include "layer.cuh"
#include "../utility/cuda_functions.cuh"

namespace cudl
{
	class Dense : public Layer
	{
	public:
		Dense(std::string name, int output_size)
		{
			name_ = name;
			output_size_ = output_size;
		};

		virtual ~Dense()
		{
			if (d_one_vec != nullptr)
			{
				cudaFree(d_one_vec);
				d_one_vec = nullptr;
			}
		};

		Dense* call(std::vector<Layer*> layers) {
			if (layers.size() != 1)
				throw std::invalid_argument("Dense layer must accept only 1 input.");
			inbound_layers_.push_back(layers[0]);
			layers[0]->add_outbound(this);
			return this;
		}

		Dense* call(Layer* layer) {
			inbound_layers_.push_back(layer);
			layer->add_outbound(this);
			return this;
		}

		Dense* call() {
			return this;
		}

		//Layer* call(Network* nn) {
		//	nn->add_layer(this);
		//	return this;
		//}

		virtual void forward()
		{
			// output = weights_T * input (without bias)
			checkCublasErrors(cublasSgemm(
				cuda_->cublas(),
				CUBLAS_OP_T, CUBLAS_OP_N,
				output_size_, batch_size_, input_size_,
				&cuda_->one,
				weights_->cuda(), input_size_,
				input_->cuda(), input_size_,
				&cuda_->zero,
				output_->cuda(), output_size_
			));

			// output += biases * d_one_vec^T
			checkCublasErrors(cublasSgemm(
				cuda_->cublas(),
				CUBLAS_OP_N, CUBLAS_OP_N,
				output_size_, batch_size_, 1,
				&cuda_->one,
				biases_->cuda(), output_size_,
				d_one_vec, 1,
				&cuda_->one,
				output_->cuda(), output_size_
			));

#if (DEBUG_DENSE & 0x01)
			input_->print(name_ + "::input", true);
			weights_->print(name_ + "::weight", true);
			biases_->print(name_ + "::bias", true);
			output_->print(name_ + "::output", true);
#endif // DEBUG_DENSE
		};

		virtual void backward()
		{
			// db = dy * d_one_vec
			checkCublasErrors(cublasSgemv(
				cuda_->cublas(),
				CUBLAS_OP_N,
				output_size_, batch_size_,
				&cuda_->one,
				grad_output_->cuda(), output_size_,
				d_one_vec, 1,
				&cuda_->zero,
				grad_biases_->cuda(), 1
			));

			// dw = x * dy^T
			checkCublasErrors(cublasSgemm(
				cuda_->cublas(),
				CUBLAS_OP_N, CUBLAS_OP_T,
				input_size_, output_size_, batch_size_,
				&cuda_->one,
				input_->cuda(), input_size_,
				grad_output_->cuda(), output_size_,
				&cuda_->zero,
				grad_weights_->cuda(), input_size_
			));

			// dx = W * dy
			if (!gradient_stop_)
			{
				checkCublasErrors(cublasSgemm(
					cuda_->cublas(),
					CUBLAS_OP_N, CUBLAS_OP_N,
					input_size_, batch_size_, output_size_,
					&cuda_->one,
					weights_->cuda(), input_size_,
					grad_output_->cuda(), output_size_,
					&cuda_->zero,
					grad_input_->cuda(), input_size_
				));
			}

#if (DEBUG_DENSE & 0x02)
			std::cout << name_ << "[BACKWARD]\n";
			grad_output_->print(name_ + "::gradients", true, grad_output->n());
			grad_weights_->print(name_ + "::gfilter", true);
			grad_biases_->print(name_ + "::gbias", true);
			if (!gradient_stop_)
			{
				grad_input_->print(name_ + "::gdata", true);
			}
#endif // DEBUG_DENSE
		};

		// operators
		virtual std::string repr() const
		{
			return name_ + "\tOutput Size: { " + std::to_string(output_size_) + " }";
		}

	private:
		void fwd_initialize()
		{
			input_ = inbound_layers_[0]->get_output();
			// initialize weights and biases
			if (weights_ == nullptr)
			{
				// setup parameter size information
				input_size_ = input_->c() * input_->h() * input_->w();

				// initialize weight/bias
				weights_ = new Tensor<float>(1, 1, input_size_, output_size_);
				biases_ = new Tensor<float>(1, 1, output_size_);
			}

			// initialize input and output
			if (output_ == nullptr || batch_size_ != input_->n())
			{
				batch_size_ = input_->n();

				if (output_ == nullptr)
				{
					output_ = new Tensor<float>(batch_size_, output_size_);
				}
				else
				{
					output_->reset(batch_size_, output_size_);
				}

				output_->tensor();

				if (d_one_vec != nullptr)
				{
					cudaFree(d_one_vec);
				}

				checkCudaErrors(cudaMalloc((void**)&d_one_vec, sizeof(float) * batch_size_));

				run_init_one_vec(d_one_vec, batch_size_);

				// initialize weights and biases
				if (load_pretrain_ && !freeze_)
				{
					if (load_parameter())
					{
						py::print("error occured..\n");
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
				grad_biases_ = new Tensor<float>(biases_->shape());
			}

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

		int input_size_ = 0;
		int output_size_ = 0;

		float* d_one_vec = nullptr;
	};
}