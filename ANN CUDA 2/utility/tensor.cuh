#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cudnn.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <array>
#include <string>
#include <iostream>
#include <fstream>

#include "helper.cuh"

namespace py = pybind11;

namespace cudl
{
	typedef enum
	{
		host,
		cuda
	} DeviceType;

	template <typename ftype> class Tensor
	{
	public:
		Tensor(int n = 1, int c = 1, int h = 1, int w = 1) :
			n_(n),
			c_(c),
			h_(h),
			w_(w)
		{
			h_ptr_ = new float[n_ * c_ * h_ * w_];
		}

		Tensor(std::array<int, 4> size) :
			n_(size[0]),
			c_(size[1]),
			h_(size[2]),
			w_(size[3])
		{
			h_ptr_ = new float[n_ * c_ * h_ * w_];
		}

		Tensor(py::array_t<ftype> data)
		{
			py::buffer_info buf_info = data.request();
			// size check
			if (buf_info.size < 1)
				throw std::invalid_argument("Array can't have less than 1 element!");
			// dim check
			if (buf_info.ndim != 4)
				throw std::invalid_argument("Must be 4D array in [n, c, h, w] format!");

			n_ = buf_info.shape[0];
			c_ = buf_info.shape[1];
			h_ = buf_info.shape[2];
			w_ = buf_info.shape[3];
			h_ptr_ = (ftype*)buf_info.ptr;
		}

		~Tensor()
		{
			if (h_ptr_ != nullptr)
			{
				delete[] h_ptr_;
			}

			if (d_ptr_ != nullptr)
			{
				cudaFree(d_ptr_);
			}

			if (is_tensor_)
			{
				cudnnDestroyTensorDescriptor(tensor_desc_);
			}
		}

		// reset current tensor with new size information
		void reset(int  n = 1, int c = 1, int h = 1, int w = 1)
		{
			// update size information
			n_ = n;
			c_ = c;
			h_ = h;
			w_ = w;

			// terminate current buffers
			if (h_ptr_ != nullptr)
			{
				delete[] h_ptr_;
				h_ptr_ = nullptr;
			}

			if (d_ptr_ != nullptr)
			{
				cudaFree(d_ptr_);
				d_ptr_ = nullptr;
			}

			// create new buffer
			h_ptr_ = new float[n_ * c_ * h_ * w_];
			cuda();

			// reset tensor descriptor if previous state was a tensor
			if (is_tensor_)
			{
				cudnnDestroyTensorDescriptor(tensor_desc_);
				is_tensor_ = false;
			}
		}

		void reset(std::array<int, 4> size) { reset(size[0], size[1], size[2], size[3]); }

		// backpropagation helper functions for copying device data

		void copy_cuda(Tensor* in) {
			checkCudaErrors(cudaMemcpyAsync(this->cuda(), in->cuda(), this->buf_size(), cudaMemcpyDeviceToDevice));
		}

		// returns array of tensor shape
		std::array<int, 4> shape() { return std::array<int, 4>({ n_, c_, h_, w_ }); }

		// returns array of tensor shape without batch dimension
		std::array<int, 3> shape_no_batch() { return std::array<int, 3>({ c_, h_, w_ }); }

		// returns number of elements for 1 batch
		int size() { return c_ * h_ * w_; }

		// returns number of total elements in tensor including batch
		int len() { return n_ * c_ * h_ * w_; }

		// returns size of allocated memory
		int buf_size() { return sizeof(ftype) * len(); }

		int n() const { return n_; }
		int c() const { return c_; }
		int h() const { return h_; }
		int w() const { return w_; }

		// Tensor Control
		bool is_tensor_ = false;
		cudnnTensorDescriptor_t tensor_desc_;

		cudnnTensorDescriptor_t tensor()
		{
			if (is_tensor_)
			{
				return tensor_desc_;
			}

			cudnnCreateTensorDescriptor(&tensor_desc_);
			cudnnSetTensor4dDescriptor(tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_, c_, h_, w_);
			is_tensor_ = true;

			return tensor_desc_;
		}

		// Memory Control
		// get specified memory pointer
		ftype* ptr() { return h_ptr_; }

		void data(py::array_t<ftype> input) {
			py::buffer_info buf_info = input.request();
			// size check
			if (buf_info.size != len())
			{
				throw std::invalid_argument("Array of '" +
					std::to_string(buf_info.size) + "' elements must equal Tensor of '"
					+ std::to_string(len()) + "' elements!\n");
			}
			// get pointer from numpy array
			h_ptr_ = (ftype*)buf_info.ptr;
		}

		py::array_t<ftype> data() {
			// return numpy array with data from h_ptr_
			if (d_ptr_ != nullptr)
				to_(host);

			py::array_t<ftype> result(len(), h_ptr_);
			result.resize({ n_, c_, h_, w_ });
			return result;
		}

		// get cuda memory
		ftype* cuda()
		{
			if (d_ptr_ == nullptr)
			{
				cudaMalloc((void**)&d_ptr_, sizeof(ftype) * len());
			}

			return d_ptr_;
		}

		// transfer data between host and device memory
		ftype* to(DeviceType target)
		{
			ftype* ptr = nullptr;
			if (target == host)
			{
				cudaMemcpy(h_ptr_, cuda(), sizeof(ftype) * len(), cudaMemcpyDeviceToHost);
				ptr = h_ptr_;
			}

			else
			{
				cudaMemcpy(cuda(), h_ptr_, sizeof(ftype) * len(), cudaMemcpyHostToDevice);
				ptr = d_ptr_;
			}

			return ptr;
		}

		void to_(DeviceType target)
		{
			if (target == host)
			{
				cudaMemcpy(h_ptr_, cuda(), sizeof(ftype) * len(), cudaMemcpyDeviceToHost);
			}

			else
			{
				cudaMemcpy(cuda(), h_ptr_, sizeof(ftype) * len(), cudaMemcpyHostToDevice);
			}
		}
		
		std::string print(std::string name, bool view_param = false, int num_batch = 1, int width = 16)
		{
			std::stringstream ss;
			to(host);
			ss << "**" << name << " size: (" << size() << ")\t";
			ss << "n: " << n_ << ", c: " << c_ << ", h: " << h_ << ", w: " << w_;
			ss << std::hex << "\t(h:" << h_ptr_ << ", d:" << d_ptr_ << ")" << std::dec << '\n';
			
			if (view_param)
			{
				ss << std::fixed;
				ss.precision(6);

				int max_print_line = 4;
				if (width == 28)
				{
					ss.precision(3);
					max_print_line = 28;
				}

				int offset = 0;

				for (int n = 0; n < num_batch; n++)
				{
					if (num_batch > 1)
					{
						ss << "<----- Batch[" << n << "] ----->\n";
					}

					int count = 0;
					int print_line_count = 0;

					while (count < size() && print_line_count < max_print_line)
					{
						ss << "\t";
						for (int s = 0; s < width && count < size(); s++)
						{
							ss << h_ptr_[size() * n + count + offset] << "\t";
							count++;
						}

						ss << '\n';
						print_line_count++;
					}
				}
			}
			ss.unsetf(std::ios::fixed);
			return ss.str();
		}

		std::string repr()
		{
			std::stringstream ss;
			ss << "**Tensor_size: (" << size() << ")\t";
			ss << "Shape: (n: " << n_ << ", c: " << c_ << ", h: " << h_ << ", w: " << w_ << ")";
			return ss.str();
		}

		// pretrained parameter load and save
		int file_read(std::string filename)
		{
			std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);
			if (!file.is_open())
			{
				py::print("failed to open ", filename, '\n');
				return -1;
			}

			file.read((char*)h_ptr_, sizeof(float) * this->len());
			this->to(DeviceType::cuda);
			file.close();

			return 0;
		}

		int file_write(std::string filename)
		{
			std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
			if (!file.is_open())
			{
				py::print("failed to write ", filename, '\n');
				return -1;
			}

			file.write((char*)this->to(host), sizeof(float) * this->len());
			file.close();

			return 0;
		}

	private:

		ftype* h_ptr_ = nullptr;
		ftype* d_ptr_ = nullptr;

		int n_ = 1;
		int c_ = 1;
		int h_ = 1;
		int w_ = 1;
	};
}