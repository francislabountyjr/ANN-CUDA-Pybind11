#pragma once

#include <vector>
#include <iomanip>
#include <stack>
#include <nvtx3/nvToolsExt.h>
/*
#include "helper.cuh"
#include "loss.cuh"
#include "layer.cuh"
#include "dense.cuh"
#include "conv2d.cuh"
#include "activation.cuh"
#include "pooling.cuh"
#include "softmax.cuh"*/

namespace cudl
{
	typedef enum
	{
		training,
		inference
	} WorkloadType;

	class Network
	{
	public:
		Network() {};
		~Network()
		{
			for (auto layer : layers_)
			{
				delete layer;
			}

			if (cuda_ != nullptr)
			{
				delete cuda_;
			}
		};

		void add_layer(Layer* layer)
		{
			layers_.push_back(layer);

			// tag layer to stop gradient if it is the first layer
			//if (layers_.size() == 1)
			//{
			//	layers_.at(0)->set_gradient_stop();
			//}
			int index = layers_.size() - 1;
			layers_[index]->index_ = index;
		};

		void add_layer(Input* layer)
		{
			layers_.push_back(layer);
			input_layers_.push_back(layer);

			int index = layers_.size() - 1;
			layers_[index]->index_ = index;
		};

		Layer* add(Layer* layer) {
			add_layer(layer);
			return layer;
		}

		Input* add(Input* layer) {
			add_layer(layer);
			return layer;
		}

		//std::vector<Tensor<float>*> forward(std::vector<Tensor<float>*> input)
		void forward(std::vector<Tensor<float>*> input)
		{
			//output_ = input;
			if (input.size() != input_layers_.size())
				throw std::invalid_argument("Number of input tensors does not match number of input layers.");

			for (int i = 0; i < input_layers_.size(); i++)
				input_layers_[i]->output_ = input[i];

			nvtxRangePushA("Forward");
			for (auto layer : layers_topological_)
			{
#if (DEBUG_FORWARD)
				std::cout << "[Forward][" << std::setw(7) << layer->get_name() << "]\t(" << output_->n() << ", " << output_->c() << ", " << output_->h() << ", " << output_->w() << ")\t";
#endif // DEBUG_FORWARD

				layer->fwd_initialize();
				layer->forward();

#if (DEBUG_FORWARD)
				std::cout << "--> (" << output_->n() << ", " << output_->c() << ", " << output_->h() << ", " << output_->w() << ")\n";
				checkCudaErrors(cudaDeviceSynchronize());

#if (DEBUG_FORWARD > 1)
				output_->print("output", true);

				if (phase_ == inference)
				{
					getchar();
				}
#endif
#endif // DEBUG_FORWARD
			}
			nvtxRangePop();

			//output_.clear();
			//for (auto layer : output_layers_)
			//	output_.push_back(layer->output_);

			//return output_;
		};

		void backward(std::vector<Tensor<float>*> target)
		{
			if (phase_ == inference)
			{
				return;
			}

			if (target.size() != output_layers_.size())
				throw std::invalid_argument("Number of target tensors does not match number of output layers.");

			for (int i = 0; i < output_layers_.size(); i++)
				output_layers_[i]->grad_output_ = target[i];

			// back propagation (update weights internally)
			nvtxRangePushA("Backward");
			for (auto layer = layers_topological_.rbegin(); layer != layers_topological_.rend(); layer++)
			{
				// getting back propagation status with gradient size
#if (DEBUG_BACKWARD)
				std::cout << "[Backward][" << std::setw(7) << (*layer)->get_name() << "]\t(" << gradient->n() << ", " << gradient->c() << ", " << gradient->h() << ", " << gradient->w() << ")\t";
#endif // DEBUG_BACKWARD

				(*layer)->bwd_initialize();
				(*layer)->backward();

#if (DEBUG_BACKWARD)
				std::cout << "--> (" << gradient->n() << ", " << gradient->c() << ", " << gradient->h() << ", " << gradient->w() << ")\n";
				checkCudaErrors(cudaDeviceSynchronize());

#if (DEBUG_BACKWARD > 1)
				gradient->print((*layer)->get_name() + "::dx", true);
				getchar();
#endif
#endif // DEBUG_BACKWARD
			}
			nvtxRangePop();
		};

		void update(float learning_rate = 0.02f)
		{
			if (phase_ == inference)
			{
				return;
			}

#if (DEBUG_UPDATE)
			std::cout << "Start update...lr = " << learning_rate << '\n';
#endif // DEBUG_UPDATE

			nvtxRangePushA("Update");
			for (auto layer : layers_)
			{
				// pass if no parameters
				if (layer->weights_ == nullptr || layer->grad_weights_ == nullptr || layer->biases_ == nullptr || layer->grad_biases_ == nullptr)
				{
					continue;
				}

				layer->update_weights_biases(learning_rate);
			}
			nvtxRangePop();
		};

		void load(std::string& parameter_location)
		{
			py::print("...Loading Weights...\n");

			for (auto layer : layers_)
			{
				layer->set_parameter_directory(parameter_location);
				layer->set_load_pretrain();
			}
		};

		void save(std::string& parameter_location)
		{
			py::print("...Storing Weights...\n");

			for (auto layer : layers_)
			{
				layer->set_parameter_directory(parameter_location);
				int err = layer->save_parameter();

				if (err != 0)
				{
					py::print("-> error code:", err,'\n');
					exit(err);
				}
			}
		};

		std::vector<float> loss(std::vector<Tensor<float>*> target)
		{
			if (target.size() != output_layers_.size())
				throw std::invalid_argument("Number of target tensors does not match number of output layers.");

			std::vector<float> out;
			for (int i = 0; i < output_layers_.size(); i++)
				out.push_back(output_layers_[i]->get_loss(target[i]));
			return out;
		};

		std::vector<int> get_accuracy(std::vector<Tensor<float>*> target)
		{
			if (target.size() != output_layers_.size())
				throw std::invalid_argument("Number of target tensors does not match number of output layers.");

			std::vector<int> out;
			for (int i = 0; i < output_layers_.size(); i++)
				out.push_back(output_layers_[i]->get_accuracy(target[i]));
			return out;
		};

		// 1. initialize cuda resource container
		// 2. register the resource container to all layers
		// 3. topologically sort layer graph
		// 4. calculate network's output layers
		void cuda()
		{
			cuda_ = new CudaContext();

			py::print("...Model Configuration...\n");

			for (auto layer : layers_)
			{
				py::print("CUDA:", layer->get_name(), '\n');
				layer->set_cuda_context(cuda_);
			}

			topological_sort();
			calculate_output_layers();
		};

		std::string repr()
		{
			std::stringstream ss;
			ss << "...Model Configuration...\n";
			for (auto layer : layers_)
			{
				ss << "Layer: " << *layer << "\n";//" In Shape:("
					//<< layer->input_->c() << ", " << layer->input_->h()
					//<< ", " << layer->input_->w() << ") \t" << "Out Shape:("
					//<< layer->output_->c() << ", " << layer->output_->h()
					//<< ", " << layer->output_->w() << ")\n";
			}

			return ss.str();
		}

		void train()
		{
			phase_ = training;

			// unfreeze all layers
			for (auto layer : layers_)
			{
				layer->unfreeze();
			}
		};

		void test()
		{
			phase_ = inference;

			// freeze all layers
			for (auto layer : layers_)
			{
				layer->freeze();
			}
		};

		std::vector<Layer*> layers()
		{
			return layers_;
		};

		Layer* layers(unsigned int index) {
			if (index > layers_.size() - 1)
				throw std::invalid_argument("Index must be a valid index.");
			return layers_[index];
		};

		std::vector<Input*> input_layers()
		{
			return input_layers_;
		};

		Input* input_layers(unsigned int index)
		{
			if (index > input_layers_.size() - 1)
				throw std::invalid_argument("Index must be a valid index.");
			return input_layers_[index];
		};

		std::vector<Layer*> output_layers()
		{
			return output_layers_;
		};

		Layer* output_layers(unsigned int index)
		{
			if (index > output_layers_.size() - 1 || output_layers_.size() < 1)
				throw std::invalid_argument("Index must be a valid index.");
			return output_layers_[index];
		};

		std::vector<Layer*> layers_topological()
		{
			return layers_topological_;
		};

		Layer* layers_topological(unsigned int index) {
			if (index > layers_topological_.size() - 1)
				throw std::invalid_argument("Index must be a valid index.");
			return layers_topological_[index];
		}

		void topological_sort_recursive_call(int i, bool* visited, std::stack<int>& Stack) {
			// mark layer as visited
			visited[i] = true;

			// run recursive function over all outbound layers
			for (auto layer : layers_[i]->outbound_layers_)
				if (!visited[layer->index_]) // index_ is the position in layer_ vector
					topological_sort_recursive_call(layer->index_, visited, Stack);

			// push current layer's index to stack
			Stack.push(i);
		}

		void topological_sort() {
			// initialize stack to store sorted indexs of layers in layers_ vector
			std::stack<int> Stack;

			// mark all layers as unvisited
			bool* visited = new bool[layers_.size()];
			memset(visited, false, sizeof(bool) * layers_.size());

			// call recursive topological sort function for each layer
			for (int i = 0; i < layers_.size(); i++)
				if (visited[i] == false)
					topological_sort_recursive_call(i, visited, Stack);

			if (Stack.size() != layers_.size())
				throw std::invalid_argument("Cycle exists in graph!");

			// copy stack data to layers_topological_ vector
			layers_topological_.clear();
			while (Stack.empty() == false) {
				layers_topological_.push_back(layers_[Stack.top()]);
				Stack.pop();
			}
		}

		void calculate_output_layers() {
			output_layers_.clear();
			for (int i = 0; i < layers_.size(); i++)
				if (layers_[i]->outbound_layers_.size() < 1)
					output_layers_.push_back(layers_[i]);
		}

		std::vector<Tensor<float>*> output_;

	private:
		std::vector<Layer*> layers_;
		std::vector<Layer*> layers_topological_;
		std::vector<Input*> input_layers_;
		std::vector<Layer*> output_layers_;

		std::vector<Layer*> layers_forward_;
		std::vector<Layer*> layers_backward_;

		CudaContext* cuda_ = nullptr;

		WorkloadType phase_ = inference;
	};
}