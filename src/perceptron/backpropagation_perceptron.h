/*
 * backpropagation_perceptron.h
 *
 *  Created on: Feb 4, 2012
 *      Author: dragoon
 */

#ifndef BACKPROPAGATION_PERCEPTRON_H_
#define BACKPROPAGATION_PERCEPTRON_H_

#include "../matrix/matrix.h"

#include <vector>

namespace perceptron {

using matrix::Matrix;
using std::vector;

class BackpropagationPerceptron {

public:
	BackpropagationPerceptron(vector<size_t> &layers);
	virtual ~BackpropagationPerceptron();

	double* evaluate(double* input_row);
	void teach(double* expected_row);

private:
	void init(vector<size_t> &layers);

	size_t layers_count_;
	size_t *layer_neuron_count_;
	Matrix** W_;
	Matrix** prev_direction_W_;
	double** descent_gradient_;
	double** answers_;
	double* temp_row_;
	bool add_const_x_;
	double learning_speed_;
	double inertia_;
};

} /* namespace hme_model */
#endif /* BACKPROPAGATION_PERCEPTRON_H_ */
