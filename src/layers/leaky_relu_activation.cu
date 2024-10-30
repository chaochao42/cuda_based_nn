#include "leaky_relu_activation.h"
#include "../nn_utils/nn_exception.h"
#include <iostream>



__global__ void LeakyReluActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim, float alpha) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) A[index] = fmaxf(Z[index], alpha * Z[index]);
}


__global__ void LeakyReluActivationBackward(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim, float alpha) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) dZ[index] = dA[index] * fmaxf((Z[index] > 0), alpha);
}

LeakyReluActivation::LeakyReluActivation(std::string name, float alpha) {
	this->name = name;
    this->alpha = alpha;
}

LeakyReluActivation::~LeakyReluActivation()
{ }

Matrix& LeakyReluActivation::forward(Matrix& Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	LeakyReluActivationForward<<<num_of_blocks, block_size>>>(Z.data_device.get(), A.data_device.get(),
														   	Z.shape.x, Z.shape.y, this->alpha);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform reaky_relu forward propagation.");

	return A;
}

Matrix& LeakyReluActivation::backward(Matrix& dA, float learning_rate) {
	dZ.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	LeakyReluActivationBackward<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(),
															 dZ.data_device.get(),
															 Z.shape.x, Z.shape.y, this->alpha);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform reaky_relu back propagation");

	return dZ;
}
