#pragma once

#include "nn_layer.h"

class LeakyReluActivation : public NNLayer {
private:
	Matrix A; // 输出

	Matrix Z; // 输入
	Matrix dZ;
	float alpha;
public:
	LeakyReluActivation(std::string name, float alpha = 0.01);
	~LeakyReluActivation();

	Matrix& forward(Matrix& Z);
	// dA = dCost/dsigma
	Matrix& backward(Matrix& dA, float learning_rate = 0.01);
};
