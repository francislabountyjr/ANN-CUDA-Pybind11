#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>b
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "utility/tensor.cuh"
#include "layers/layer.cuh"
#include "layers/input.cuh"
#include "layers/softmax.cuh"
#include "layers/dense.cuh"
#include "layers/add.cuh"
#include "layers/concat.cuh"
#include "layers/activation.cuh"
#include "layers/conv2d.cuh"
#include "layers/pooling.cuh"
#include "network/network.cuh"

namespace py = pybind11;
using namespace cudl;

PYBIND11_MODULE(ann_cuda, m)
{
	// docstring
	m.doc() = "A python module for utilizing cudnn for graph based neural networks.";
	
	// cudnn activation enum
	py::enum_<cudnnActivationMode_t>(m, "cudnnActivationMode_t")
		.value("CUDNN_ACTIVATION_SIGMOID", cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID)
		.value("CUDNN_ACTIVATION_RELU", cudnnActivationMode_t::CUDNN_ACTIVATION_RELU)
		.value("CUDNN_ACTIVATION_TANH", cudnnActivationMode_t::CUDNN_ACTIVATION_TANH)
		.value("CUDNN_ACTIVATION_CLIPPED_RELU", cudnnActivationMode_t::CUDNN_ACTIVATION_CLIPPED_RELU)
		.value("CUDNN_ACTIVATION_ELU", cudnnActivationMode_t::CUDNN_ACTIVATION_ELU)
		.value("CUDNN_ACTIVATION_IDENTITY", cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY)
		.value("CUDNN_ACTIVATION_SWISH", cudnnActivationMode_t::CUDNN_ACTIVATION_SWISH)
		.export_values();

	// cudnn pooling enum
	py::enum_<cudnnPoolingMode_t>(m, "cudnnPoolingMode_t")
		.value("CUDNN_POOLING_MAX", cudnnPoolingMode_t::CUDNN_POOLING_MAX)
		.value("CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING", cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
		.value("CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING", cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
		.value("CUDNN_POOLING_MAX_DETERMINISTIC", cudnnPoolingMode_t::CUDNN_POOLING_MAX_DETERMINISTIC)
		.export_values();

	// devicetype enum
	py::enum_<DeviceType>(m, "DeviceType")
		.value("cuda", DeviceType::cuda)
		.value("host", DeviceType::host)
		.export_values();

	// bind Tensor class
	py::class_<Tensor<float>, std::unique_ptr<Tensor<float>, py::nodelete>>(m, "Tensor")
		.def(py::init<int, int, int, int>(), py::arg("n"), py::arg("c"), py::arg("h"), py::arg("w"))
		.def(py::init<std::array<int, 4>>(), py::arg("size"))
		.def(py::init<py::array_t<float>>(), py::arg("data"))
		.def("reset", static_cast<void (Tensor<float>::*)(int, int, int, int)>(&Tensor<float>::reset), py::arg("n"), py::arg("c"), py::arg("h"), py::arg("w"))
		.def("reset", static_cast<void (Tensor<float>::*)(std::array<int, 4>)>(&Tensor<float>::reset), py::arg("size"))
		.def("shape", &Tensor<float>::shape)
		.def("size", &Tensor<float>::size)
		.def("len", &Tensor<float>::len)
		.def("buf_size", &Tensor<float>::buf_size)
		.def("n", &Tensor<float>::n)
		.def("c", &Tensor<float>::c)
		.def("h", &Tensor<float>::h)
		.def("w", &Tensor<float>::w)
		.def("data", static_cast<py::array_t<float>(Tensor<float>::*)()>(&Tensor<float>::data))
		.def("data", static_cast<void (Tensor<float>::*)(py::array_t<float>)>(&Tensor<float>::data), py::arg("input"))
		.def("to", &Tensor<float>::to_, py::arg("target"))
		.def("print", &Tensor<float>::print, py::arg("name"), py::arg("view_param"), py::arg("num_batch"), py::arg("width"))
		.def("__str__", &Tensor<float>::repr)
		.def("__repr__", &Tensor<float>::repr);

	// bind Network class
	py::class_<Network>(m, "Network")
		.def(py::init<>())
		.def("forward", &Network::forward, py::arg("input"))
		.def("backward", &Network::backward, py::arg("target"))
		.def("update", &Network::update, py::arg("learning_rate"))
		.def("load", &Network::load, py::arg("parameter_location"))
		.def("save", &Network::save, py::arg("parameter_location"))
		.def("loss", &Network::loss, py::arg("target"))
		.def("get_accuracy", &Network::get_accuracy, py::arg("target"))
		.def("cuda", &Network::cuda)
		.def("train", &Network::train)
		.def("test", &Network::test)
		.def("layers", static_cast<std::vector<Layer*> (Network::*)()>(&Network::layers))
		.def("layers", static_cast<Layer* (Network::*)(unsigned int)>(&Network::layers), py::arg("index"))
		.def("input_layers", static_cast<std::vector<Input*>(Network::*)()>(&Network::input_layers))
		.def("input_layers", static_cast<Input* (Network::*)(unsigned int)>(&Network::input_layers), py::arg("index"))
		.def("output_layers", static_cast<std::vector<Layer*>(Network::*)()>(&Network::output_layers))
		.def("output_layers", static_cast<Layer* (Network::*)(unsigned int)>(&Network::output_layers), py::arg("index"))
		.def("layers_topological", static_cast<std::vector<Layer*>(Network::*)()>(&Network::layers_topological))
		.def("layers_topological", static_cast<Layer * (Network::*)(unsigned int)>(&Network::layers_topological), py::arg("index"))
		.def("topological_sort", &Network::topological_sort)
		.def("calculate_output_layers", &Network::calculate_output_layers)
		.def("__add__", static_cast<Input* (Network::*)(Input*)>(&Network::add), py::arg("layer"))
		.def("__add__", static_cast<Layer* (Network::*)(Layer*)>(&Network::add), py::arg("layer"))
		.def("__str__", &Network::repr)
		.def("__repr__", &Network::repr);

	// bind Layer class
	py::class_<Layer>(m, "Layer");

	// bind Input class
	py::class_<Input, Layer, std::unique_ptr<Input, py::nodelete>>(m, "Input")
		.def(py::init<std::string>(), py::arg("name"))
		.def("output", &Input::get_output)
		.def("input", &Input::get_input)
		.def("weights", &Input::get_weights)
		.def("biases", &Input::get_biases)
		.def("grad_weights", &Input::get_grad_weights)
		.def("grad_biases", &Input::get_grad_biases)
		.def("grad_input", &Input::get_grad_inputs)
		.def("grad_output", &Input::get_grad_output)
		.def("inbound_print", &Input::inbound_print)
		.def("outbound_print", &Input::outbound_print)
		.def("inbound", static_cast<std::vector<Layer*>(Input::*)()>(&Input::get_inbound_layers))
		.def("inbound", static_cast<Layer * (Input::*)(unsigned int)>(&Input::get_inbound_layers), py::arg("index"))
		.def("outbound", static_cast<std::vector<Layer*>(Input::*)()>(&Input::get_outbound_layers))
		.def("outbound", static_cast<Layer * (Input::*)(unsigned int)>(&Input::get_outbound_layers), py::arg("index"))
		.def_readonly("index_", &Input::index_)
		.def("__call__", static_cast<Input * (Input::*)(int)>(&Input::call), py::arg("shape"))
		.def("__call__", static_cast<Input * (Input::*)(std::array<int, 3>)>(&Input::call), py::arg("shape"))
		.def("__str__", &Input::repr)
		.def("__repr__", &Input::repr);

	// bind Dense class
	py::class_<Dense, Layer, std::unique_ptr<Dense, py::nodelete>>(m, "Dense")
		.def(py::init<std::string, int>(), py::arg("name"), py::arg("output_size"))
		.def("output", &Dense::get_output)
		.def("input", &Dense::get_input)
		.def("weights", &Dense::get_weights)
		.def("biases", &Dense::get_biases)
		.def("grad_weights", &Dense::get_grad_weights)
		.def("grad_biases", &Dense::get_grad_biases)
		.def("grad_input", &Dense::get_grad_inputs)
		.def("grad_output", &Dense::get_grad_output)
		.def("inbound_print", &Dense::inbound_print)
		.def("outbound_print", &Dense::outbound_print)
		.def("inbound", static_cast<std::vector<Layer*>(Dense::*)()>(&Dense::get_inbound_layers))
		.def("inbound", static_cast<Layer* (Dense::*)(unsigned int)>(&Dense::get_inbound_layers), py::arg("index"))
		.def("outbound", static_cast<std::vector<Layer*>(Dense::*)()>(&Dense::get_outbound_layers))
		.def("outbound", static_cast<Layer* (Dense::*)(unsigned int)>(&Dense::get_outbound_layers), py::arg("index"))
		.def_readonly("index_", &Dense::index_)
		.def("__call__", static_cast<Dense* (Dense::*)()>(&Dense::call))
		.def("__call__", static_cast<Dense* (Dense::*)(std::vector<Layer*>)>(&Dense::call), py::arg("layers"))
		.def("__call__", static_cast<Dense* (Dense::*)(Layer*)>(&Dense::call), py::arg("layer"))
		.def("__str__", &Dense::repr)
		.def("__repr__", &Dense::repr);
	
	// bind Conv2D class
	py::class_<Conv2D, Layer, std::unique_ptr<Conv2D, py::nodelete>>(m, "Conv2D")
		.def(py::init<std::string, int, int, int, int, int>(), py::arg("name"), py::arg("out_channels"), py::arg("kernel_size"), py::arg("stride"), py::arg("padding"), py::arg("dilation"))
		.def("output", &Conv2D::get_output)
		.def("input", &Conv2D::get_input)
		.def("weights", &Conv2D::get_weights)
		.def("biases", &Conv2D::get_biases)
		.def("grad_weights", &Conv2D::get_grad_weights)
		.def("grad_biases", &Conv2D::get_grad_biases)
		.def("grad_input", &Conv2D::get_grad_inputs)
		.def("grad_output", &Conv2D::get_grad_output)
		.def("inbound_print", &Conv2D::inbound_print)
		.def("outbound_print", &Conv2D::outbound_print)
		.def("inbound", static_cast<std::vector<Layer*>(Conv2D::*)()>(&Conv2D::get_inbound_layers))
		.def("inbound", static_cast<Layer* (Conv2D::*)(unsigned int)>(&Conv2D::get_inbound_layers), py::arg("index"))
		.def("outbound", static_cast<std::vector<Layer*>(Conv2D::*)()>(&Conv2D::get_outbound_layers))
		.def("outbound", static_cast<Layer* (Conv2D::*)(unsigned int)>(&Conv2D::get_outbound_layers), py::arg("index"))
		.def_readonly("index_", &Conv2D::index_)
		.def("__call__", static_cast<Conv2D* (Conv2D::*)()>(&Conv2D::call))
		.def("__call__", static_cast<Conv2D* (Conv2D::*)(std::vector<Layer*>)>(&Conv2D::call), py::arg("layers"))
		.def("__call__", static_cast<Conv2D* (Conv2D::*)(Layer*)>(&Conv2D::call), py::arg("layer"))
		.def("__str__", &Conv2D::repr)
		.def("__repr__", &Conv2D::repr);

	// bind Add class
	py::class_<Add, Layer, std::unique_ptr<Add, py::nodelete>>(m, "Add")
		.def(py::init<std::string, float, float>(), py::arg("name"), py::arg("alpha"), py::arg("beta"))
		.def("output", &Add::get_output)
		.def("input", &Add::get_input)
		.def("weights", &Add::get_weights)
		.def("biases", &Add::get_biases)
		.def("grad_weights", &Add::get_grad_weights)
		.def("grad_biases", &Add::get_grad_biases)
		.def("grad_input", &Add::get_grad_inputs)
		.def("grad_output", &Add::get_grad_output)
		.def("inbound_print", &Add::inbound_print)
		.def("outbound_print", &Add::outbound_print)
		.def("inbound", static_cast<std::vector<Layer*>(Add::*)()>(&Add::get_inbound_layers))
		.def("inbound", static_cast<Layer* (Add::*)(unsigned int)>(&Add::get_inbound_layers), py::arg("index"))
		.def("outbound", static_cast<std::vector<Layer*>(Add::*)()>(&Add::get_outbound_layers))
		.def("outbound", static_cast<Layer* (Add::*)(unsigned int)>(&Add::get_outbound_layers), py::arg("index"))
		.def_readonly("index_", &Add::index_)
		.def("__call__", static_cast<Add* (Add::*)()>(&Add::call))
		.def("__call__", static_cast<Add* (Add::*)(std::vector<Layer*>)>(&Add::call), py::arg("layers"))
		.def("__call__", static_cast<Add* (Add::*)(Layer*)>(&Add::call), py::arg("layer"))
		.def("__str__", &Add::repr)
		.def("__repr__", &Add::repr);

	// bind Concat class
	py::class_<Concat, Layer, std::unique_ptr<Concat, py::nodelete>>(m, "Concat")
		.def(py::init<std::string>(), py::arg("name"))
		.def("output", &Concat::get_output)
		.def("input", &Concat::get_input)
		.def("weights", &Concat::get_weights)
		.def("biases", &Concat::get_biases)
		.def("grad_weights", &Concat::get_grad_weights)
		.def("grad_biases", &Concat::get_grad_biases)
		.def("grad_input", &Concat::get_grad_inputs)
		.def("grad_output", &Concat::get_grad_output)
		.def("inbound_print", &Concat::inbound_print)
		.def("outbound_print", &Concat::outbound_print)
		.def("inbound", static_cast<std::vector<Layer*>(Concat::*)()>(&Concat::get_inbound_layers))
		.def("inbound", static_cast<Layer* (Concat::*)(unsigned int)>(&Concat::get_inbound_layers), py::arg("index"))
		.def("outbound", static_cast<std::vector<Layer*>(Concat::*)()>(&Concat::get_outbound_layers))
		.def("outbound", static_cast<Layer* (Concat::*)(unsigned int)>(&Concat::get_outbound_layers), py::arg("index"))
		.def_readonly("index_", &Concat::index_)
		.def("__call__", static_cast<Concat* (Concat::*)()>(&Concat::call))
		.def("__call__", static_cast<Concat* (Concat::*)(std::vector<Layer*>)>(&Concat::call), py::arg("layers"))
		.def("__call__", static_cast<Concat* (Concat::*)(Layer*)>(&Concat::call), py::arg("layer"))
		.def("__str__", &Concat::repr)
		.def("__repr__", &Concat::repr);

	 //bind Activation class
	 py::class_<Activation, Layer, std::unique_ptr<Activation, py::nodelete>>(m, "Activation")
		.def(py::init<std::string, cudnnActivationMode_t, float>(), py::arg("name"), py::arg("mode"), py::arg("coef"))
		 .def("output", &Activation::get_output)
		 .def("input", &Activation::get_input)
		 .def("weights", &Activation::get_weights)
		 .def("biases", &Activation::get_biases)
		 .def("grad_weights", &Activation::get_grad_weights)
		 .def("grad_biases", &Activation::get_grad_biases)
		 .def("grad_input", &Activation::get_grad_inputs)
		 .def("grad_output", &Activation::get_grad_output)
		 .def("inbound_print", &Activation::inbound_print)
		 .def("outbound_print", &Activation::outbound_print)
		 .def("inbound", static_cast<std::vector<Layer*>(Activation::*)()>(&Activation::get_inbound_layers))
		 .def("inbound", static_cast<Layer* (Activation::*)(unsigned int)>(&Activation::get_inbound_layers), py::arg("index"))
		 .def("outbound", static_cast<std::vector<Layer*>(Activation::*)()>(&Activation::get_outbound_layers))
		 .def("outbound", static_cast<Layer* (Activation::*)(unsigned int)>(&Activation::get_outbound_layers), py::arg("index"))
		 .def_readonly("index_", &Activation::index_)
		.def("__call__", static_cast<Activation* (Activation::*)()>(&Activation::call))
		.def("__call__", static_cast<Activation* (Activation::*)(std::vector<Layer*>)>(&Activation::call), py::arg("layers"))
		.def("__call__", static_cast<Activation* (Activation::*)(Layer*)>(&Activation::call), py::arg("layer"))
		.def("__str__", &Activation::repr)
		.def("__repr__", &Activation::repr);

	// bind Pooling class
	py::class_<Pooling, Layer, std::unique_ptr<Pooling, py::nodelete>>(m, "Pooling")
		.def(py::init<std::string, int, int, int, cudnnPoolingMode_t>(), py::arg("name"), py::arg("kernel_size"), py::arg("padding"), py::arg("stride"), py::arg("mode"))
		.def("output", &Pooling::get_output)
		.def("input", &Pooling::get_input)
		.def("weights", &Pooling::get_weights)
		.def("biases", &Pooling::get_biases)
		.def("grad_weights", &Pooling::get_grad_weights)
		.def("grad_biases", &Pooling::get_grad_biases)
		.def("grad_input", &Pooling::get_grad_inputs)
		.def("grad_output", &Pooling::get_grad_output)
		.def("inbound_print", &Pooling::inbound_print)
		.def("outbound_print", &Pooling::outbound_print)
		.def("inbound", static_cast<std::vector<Layer*>(Pooling::*)()>(&Pooling::get_inbound_layers))
		.def("inbound", static_cast<Layer* (Pooling::*)(unsigned int)>(&Pooling::get_inbound_layers), py::arg("index"))
		.def("outbound", static_cast<std::vector<Layer*>(Pooling::*)()>(&Pooling::get_outbound_layers))
		.def("outbound", static_cast<Layer* (Pooling::*)(unsigned int)>(&Pooling::get_outbound_layers), py::arg("index"))
		.def_readonly("index_", &Pooling::index_)
		.def("__call__", static_cast<Pooling* (Pooling::*)()>(&Pooling::call))
		.def("__call__", static_cast<Pooling* (Pooling::*)(std::vector<Layer*>)>(&Pooling::call), py::arg("layers"))
		.def("__call__", static_cast<Pooling* (Pooling::*)(Layer*)>(&Pooling::call), py::arg("layer"))
		.def("__str__", &Pooling::repr)
		.def("__repr__", &Pooling::repr);

	// bind Softmax class
	py::class_<Softmax, Layer, std::unique_ptr<Softmax, py::nodelete>>(m, "Softmax")
		.def(py::init<std::string>(), py::arg("name"))
		.def("output", &Softmax::get_output)
		.def("input", &Softmax::get_input)
		.def("weights", &Softmax::get_weights)
		.def("biases", &Softmax::get_biases)
		.def("grad_weights", &Softmax::get_grad_weights)
		.def("grad_biases", &Softmax::get_grad_biases)
		.def("grad_input", &Softmax::get_grad_inputs)
		.def("grad_output", &Softmax::get_grad_output)
		.def("inbound_print", &Softmax::inbound_print)
		.def("outbound_print", &Softmax::outbound_print)
		.def("inbound", static_cast<std::vector<Layer*>(Softmax::*)()>(&Softmax::get_inbound_layers))
		.def("inbound", static_cast<Layer* (Softmax::*)(unsigned int)>(&Softmax::get_inbound_layers), py::arg("index"))
		.def("outbound", static_cast<std::vector<Layer*>(Softmax::*)()>(&Softmax::get_outbound_layers))
		.def("outbound", static_cast<Layer* (Softmax::*)(unsigned int)>(&Softmax::get_outbound_layers), py::arg("index"))
		.def_readonly("index_", &Softmax::index_)
		.def("__call__", static_cast<Softmax* (Softmax::*)()>(&Softmax::call))
		.def("__call__", static_cast<Softmax* (Softmax::*)(std::vector<Layer*>)>(&Softmax::call), py::arg("layers"))
		.def("__call__", static_cast<Softmax* (Softmax::*)(Layer*)>(&Softmax::call), py::arg("layer"))
		.def("__str__", &Softmax::repr)
		.def("__repr__", &Softmax::repr);
}