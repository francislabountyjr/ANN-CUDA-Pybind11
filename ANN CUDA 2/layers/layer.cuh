#pragma once

#include <string>
#include <random>
#include <cassert>
#include <math.h>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>

#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>

#include "../utility/tensor.cuh"
#include "../utility/loss.cuh"
#include "../utility/helper.cuh"

namespace cudl
{
	class Layer
	{
	public:
		Layer() {};

		virtual ~Layer()
		{
#if (DEBUG_FORWARD > 0 || DEBUG_BACKWARD > 0)
			std::cout << "Destroy Layer: " << name_ << '\n';
#endif
			if (output_ != nullptr) { delete output_; output_ = nullptr; }
			if (grad_input_ != nullptr) { delete grad_input_; grad_input_ = nullptr; }

			if (weights_ != nullptr) { delete weights_; weights_ = nullptr; }
			if (biases_ != nullptr) { delete biases_; biases_ = nullptr; }

			if (grad_weights_ != nullptr) { delete grad_weights_; grad_weights_ = nullptr; }
			if (grad_biases_ != nullptr) { delete grad_biases_; grad_biases_ = nullptr; }
		};

		virtual void forward() = 0;
		virtual void backward() = 0;

		std::string get_name() { return name_; }

		void set_parameter_directory(std::string& parameter_location) { parameter_location_ = parameter_location; }

		virtual float get_loss(Tensor<float>* target)
		{
			assert("No Loss layer - no loss" && false);
			return EXIT_FAILURE;
		};

		virtual int get_accuracy(Tensor<float>* target)
		{
			assert("No Loss layer - cannot estimate accuracy" && false);
			return EXIT_FAILURE;
		};

		Tensor<float>* get_input() { return input_; }

		Tensor<float>* get_output() { return output_; }

		Tensor<float>* get_weights() { return weights_; }

		Tensor<float>* get_biases() { return biases_; }

		Tensor<float>* get_grad_weights() { return grad_weights_; }

		Tensor<float>* get_grad_biases() { return grad_biases_; }

		virtual Tensor<float>* get_grad_input(int index = 0) { return grad_input_; }

		virtual std::vector<Tensor<float>*> get_grad_inputs() {
			std::vector<Tensor<float>*> out = { grad_input_ };
			return out;
		}

		Tensor<float>* get_grad_output() { return grad_output_; }

		std::vector<Layer*> get_inbound_layers() { return inbound_layers_; }

		Layer* get_inbound_layers(unsigned int index) {
			if (index > inbound_layers_.size())
				throw std::invalid_argument("Index must be a valid index.");
			return inbound_layers_[index];
		}

		std::vector<Layer*> get_outbound_layers() {
			return outbound_layers_;
		}

		Layer* get_outbound_layers(unsigned int index) {
			if (index > outbound_layers_.size())
				throw std::invalid_argument("Index must be a valid index.");
			return outbound_layers_[index];
		}

		void set_cuda_context(CudaContext* context) { cuda_ = context; }

		void set_load_pretrain() { load_pretrain_ = true; }
		void set_gradient_stop() { gradient_stop_ = true; }

		// weight freeze or unfreeze
		void freeze() { freeze_ = true; }
		void unfreeze() { freeze_ = false; }

		void add_outbound(Layer* layer) {
			outbound_layers_.push_back(layer);
		}

		std::string inbound_print() {
			if (inbound_layers_.size() < 1)
				return "No inbound layers.";
			std::stringstream ss;
			ss << "Inbound Layers: { ";
			for (auto layer : inbound_layers_)
				ss << layer->get_name() << ", ";
			ss.seekp(-2, ss.cur);
			ss << " }";
			return ss.str();
		}

		std::string outbound_print() {
			if (outbound_layers_.size() < 1)
				return "No outbound layers.";
			std::stringstream ss;
			ss << "Outbound Layers: { ";
			for (auto layer : outbound_layers_)
				ss << layer->get_name() << ", ";
			ss.seekp(-2, ss.cur);
			ss << " }";
			return ss.str();
		}

		// operators
		friend std::ostream& operator<<(std::ostream& os, const Layer& layer);

		virtual std::string repr() const = 0;

		// index for layer ordering
		int index_;
		
	protected:
		virtual void fwd_initialize() = 0;
		virtual void bwd_initialize() = 0;

		// layer name
		std::string name_;
		
		// graph info
		std::vector<Layer*> inbound_layers_;
		std::vector<Layer*> outbound_layers_;

		// tensor descriptor for input and output tensors
		cudnnTensorDescriptor_t input_desc_;
		cudnnTensorDescriptor_t output_desc_;

		// weight and bias descriptor
		cudnnFilterDescriptor_t filter_desc_;
		cudnnTensorDescriptor_t bias_desc_;

		// output memory
		Tensor<float>* input_ = nullptr; // x
		Tensor<float>* output_ = nullptr; // y
		Tensor<float>* grad_input_ = nullptr; // dx
		Tensor<float>* grad_output_ = nullptr; // dy

		// master weights and bias
		bool freeze_ = false; // control parameter updates
		Tensor<float>* weights_ = nullptr; // w
		Tensor<float>* biases_ = nullptr; // b
		Tensor<float>* grad_weights_ = nullptr; // dw
		Tensor<float>* grad_biases_ = nullptr; //db

		int batch_size_ = 0; // mini-batch size

		// initialize weights along with the input size
		void init_weight_bias(unsigned int seed = 0)
		{
			checkCudaErrors(cudaDeviceSynchronize());

			if (weights_ == nullptr || biases_ == nullptr)
			{
				return;
			}

			// create random network
			std::random_device rd;
			std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

			// he uniform distribution
			float range = sqrt(6.f / input_->size()); // he initializaiton
			std::uniform_real_distribution<> dis(-range, range);

			for (int i = 0; i < weights_->len(); i++)
			{
				weights_->ptr()[i] = static_cast<float>(dis(gen));
			}

			for (int i = 0; i < biases_->len(); i++)
			{
				biases_->ptr()[i] = 0.f;
			}

			// copy initialized values to the device
			weights_->to(DeviceType::cuda);
			biases_->to(DeviceType::cuda);

			py::print("..initialized ", name_, " layer..\n");
		};

		void update_weights_biases(float learning_rate)
		{
			float eps = -1.f * learning_rate;

			if (weights_ != nullptr && grad_weights_ != nullptr)
			{
#if (DEBUG_UPDATE)
				weights_->print(name_ + "::weights (before update)", true);
				grad_weights_->print(name_ + "::gweights", true);
#endif // DEBUG_UPDATE

				// w = w + eps * dw
				checkCublasErrors(cublasSaxpy(cuda_->cublas(),
					weights_->len(),
					&eps,
					grad_weights_->cuda(), 1,
					weights_->cuda(), 1));

#if (DEBUG_UPDATE)
				weights_->print(name_ + "::weights (after update)", true);
#endif  // DEBUG_UPDATE
			}

			if (biases_ != nullptr && grad_biases_ != nullptr)
			{
#if (DEBUG_UPDATE)
				biases_->print(name_ + "::biases (before update)", true);
				grad_biases_->print(name_ + "::gbiases", true);
#endif  // DEBUG_UPDATE

				// b = b + eps * db
				checkCublasErrors(cublasSaxpy(cuda_->cublas(),
					biases_->len(),
					&eps,
					grad_biases_->cuda(), 1,
					biases_->cuda(), 1));

#if (DEBUG_UPDATE)
				biases_->print(name_ + "::biases (after update)", true);
#endif  // DEBUG_UPDATE
			}
		};

		// cuda handle container
		CudaContext* cuda_ = nullptr;

		// folder to save parameters in
		std::string parameter_location_;

		// pretrain parameters
		bool load_pretrain_ = false;

		int load_parameter()
		{
			std::stringstream filename_weights, filename_biases;

			// load pretrained weights and biases
			filename_weights << parameter_location_ << '/' << name_ << ".bin";
			if (weights_->file_read(filename_weights.str()))
			{
				return -1;
			}

			filename_biases << parameter_location_ << '/' << name_ << ".bias.bin";
			if (biases_->file_read(filename_biases.str()))
			{
				return -2;
			}

			py::print("..loaded ", name_, " pretrained weights and biases..\n");

			return 0;
		};

		int save_parameter()
		{
			std::stringstream filename_weights, filename_biases;

			py::print("..saving", name_, "weights and biases..\n");

			// write weights file
			if (weights_)
			{
				filename_weights << parameter_location_ << '/' << name_ << ".bin";
				if (weights_->file_write(filename_weights.str()))
				{
					return -1;
				}
			}

			// write bias file
			if (biases_)
			{
				filename_biases << parameter_location_ << '/' << name_ << ".bias.bin";
				if (biases_->file_write(filename_biases.str()))
				{
					return -2;
				}
			}

			py::print("..done..\n");

			return 0;
		};

		// gradient stop tagging
		bool gradient_stop_ = false;

		friend class Network;
	};

	std::ostream& operator<<(std::ostream& os, const Layer& layer)
	{
		os << layer.repr();
		return os;
	}
}