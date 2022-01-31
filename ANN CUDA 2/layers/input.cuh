#pragma once

#include "layer.cuh"

namespace cudl
{
	class Input : public Layer
	{
	public:
		Input(std::string name)
		{
			name_ = name;
		};

		virtual ~Input()
		{
		};

		Input* call(std::array<int, 3> shape) {
			shape_ = shape;
			return this;
		}

		Input* call(int shape) {
			shape_ = std::array<int, 3> { 1, 1, shape };
			return this;
		}

		virtual void forward()
		{
		};

		virtual void backward()
		{
		};

		// operators
		virtual std::string repr() const
		{
			return name_ + "\tShape: { " + std::to_string(shape_[0]) + ", " + std::to_string(shape_[1]) + ", " + std::to_string(shape_[2]) + " }";
		}

	private:
		void fwd_initialize()
		{
			if (output_->shape_no_batch() != shape_)
				throw std::invalid_argument("Input Tensor shape must match user specified input shape.");
		};

		void bwd_initialize()
		{
		};

		std::array<int, 3> shape_;

		bool gradient_stop_ = true;
	};
}