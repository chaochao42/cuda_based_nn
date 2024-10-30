#include "leaky_relu_activation.h"
#include "../nn_utils/nn_exception.h"
#include <iostream>

__device__ float sigmoid(float x) {
	return 1.0f / (1.0f + expf(-x));
}


__global__ void sigmoidActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) A[index] = sigmoid(Z[index]);
}


__global__ void sigmoidActivationBackward(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) dZ[index] = dA[index] * sigmoid(Z[index]) * (1 - sigmoid(Z[index]));
}

LeakyReluActivation::LeakyReluActivation(std::string name) {
	this->name = name;
}

LeakyReluActivation::~LeakyReluActivation()
{ }

Matrix& LeakyReluActivation::forward(Matrix& Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	sigmoidActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(),
														   	Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform sigmoid forward propagation.");

	return A;
}

Matrix& LeakyReluActivation::backward(Matrix& dA, float learning_rate) {
	dZ.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	sigmoidActivationBackward<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(),
															 dZ.data_device.get(),
															 Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform sigmoid back propagation");

	return dZ;
}
