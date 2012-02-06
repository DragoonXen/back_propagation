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
	BackpropagationPerceptron(vector<size_t> &layers, double learning_speed = 0.05, double inertia = 0.5);
	virtual ~BackpropagationPerceptron();

	double learning_speed();
	void learning_speed(double value);

	double* evaluate(double* input_row);
	void teach(double* expected_row);
	void teach_by_errors_row(double* errors_row);
	void perceptron_learning(double** rows, double** results, size_t data_count, double** test_rows, double** test_results, size_t test_data_count);
	double evaluate_perceptron(double** test_rows, double** test_results, size_t test_data_count);

private:
	void init(vector<size_t> &layers);
	void status_remember();
	void status_recover();

	size_t layers_count_;
	size_t *layer_neuron_count_;
	Matrix** W_;
	Matrix** remembered_W_;
	Matrix** prev_direction_W_;
	double** descent_gradient_;
	double** answers_;
	double* temp_row_;
	double* errors_row_;
	bool add_const_x_;
	double learning_speed_;
	double inertia_;
};

} /* namespace perceptron */
#endif /* BACKPROPAGATION_PERCEPTRON_H_ */
