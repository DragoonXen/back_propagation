/*
 * backpropagation_perceptron.cpp
 *
 *  Created on: Feb 4, 2012
 *      Author: dragoon
 */

#include "backpropagation_perceptron.h"

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

namespace perceptron {

using std::cout;
using std::endl;
using std::copy;
using std::random_shuffle;
using std::fill;

BackpropagationPerceptron::BackpropagationPerceptron(vector<size_t> &layers, double learning_speed,
		double inertia) {
	remembered_W_ = NULL;
	learning_speed_ = learning_speed;
	inertia_ = inertia;
	add_const_x_ = true;
	init(layers);
}

BackpropagationPerceptron::~BackpropagationPerceptron() {
	if (remembered_W_ != NULL) {
		for (size_t i = 0; i < layers_count_ - 1; i++) {
			delete (remembered_W_[i]);
		}
		delete[] remembered_W_;
	}
	for (size_t i = 0; i < layers_count_ - 1; i++) {
		delete (W_[i]);
		delete (prev_direction_W_[i]);
		delete[] answers_[i];
		delete[] descent_gradient_[i];
	}
	delete[] answers_[layers_count_ - 1];

	delete[] W_;
	delete[] prev_direction_W_;
	delete[] answers_;
	delete[] descent_gradient_;
	delete[] temp_row_;
	delete[] errors_row_;
}

double BackpropagationPerceptron::learning_speed() {
	return learning_speed_;
}
void BackpropagationPerceptron::learning_speed(double value) {
	learning_speed_ = value;
}

double* BackpropagationPerceptron::evaluate(double* input_row) {
	copy(input_row, input_row + layer_neuron_count_[0], answers_[0]);
	for (size_t i = 0; i < layers_count_ - 1; i++) {
		if (add_const_x_) {
			answers_[i][layer_neuron_count_[i]] = 1.0;
		}

		//answers[i + 1] = (answers[i] * W)
		Matrix::multiply_row_with_rows_to_row(answers_[i], *W_[i], answers_[i + 1]);

		//1 / (1 + e^( -answers * W))
		for (size_t j = 0; j != layer_neuron_count_[i + 1]; j++) {
			answers_[i + 1][j] = 1.0 / (1 + pow(M_E, -answers_[i + 1][j]));
		}

	}

	double *result = new double[layer_neuron_count_[layers_count_ - 1]];
	for (size_t i = 0; i != layer_neuron_count_[layers_count_ - 1]; i++) {
		result[i] = answers_[layers_count_ - 1][i];
	}
	return result;
}

void BackpropagationPerceptron::teach(double* expected_row) {
	for (size_t i = 0; i != layer_neuron_count_[layers_count_ - 1]; i++) {
		errors_row_[i] = expected_row[i] - answers_[layers_count_ - 1][i];
	}
	teach_by_errors_row(errors_row_);
}

void BackpropagationPerceptron::teach_by_errors_row(double* errors_row) {
	size_t index = layers_count_ - 2;
	for (size_t i = 0; i != layer_neuron_count_[index + 1]; i++) {
		descent_gradient_[index][i] = answers_[index + 1][i] * (1.0 - answers_[index + 1][i])
				* errors_row[i];
	}

	while (index > 0) {
		--index;
		/*
		 * first variant:
		 * 		for (size_t i = 0; i != layer_neuron_count_[index + 1]; i++) {
		 * 			gradient_[index][i] = 0;
		 * 			for (size_t j = 0; j != layer_neuron_count_[index + 2]; j++) {
		 * 				gradient_[index][i] += (*W_[index + 1])[j][i] * gradient_[index + 1][j]
		 * 						* (1.0 - answers_[index + 1][j]) * answers_[index + 1][j];
		 * 			}
		 * 		}
		 */
		for (size_t j = 0; j != layer_neuron_count_[index + 2]; j++) {
			temp_row_[j] = descent_gradient_[index + 1][j] * (1.0 - answers_[index + 1][j])
					* answers_[index + 1][j];
		}
		for (size_t i = 0; i != layer_neuron_count_[index + 1]; i++) {
			descent_gradient_[index][i] = 0;
		}
		for (size_t j = 0; j != layer_neuron_count_[index + 2]; j++) {
			for (size_t i = 0; i != layer_neuron_count_[index + 1]; i++) {
				descent_gradient_[index][i] += (*W_[index + 1])[j][i] * temp_row_[j];
			}
		}
	}

	for (size_t i = 0; i != layers_count_ - 1; i++) {
		for (size_t j = 0; j != prev_direction_W_[i]->rows_count(); j++) {
			for (size_t k = 0; k != prev_direction_W_[i]->columns_count(); k++) {
				(*prev_direction_W_[i])[j][k] = (*prev_direction_W_[i])[j][k] * inertia_
						+ learning_speed_ * descent_gradient_[i][j] * answers_[i][k];
				(*W_[i])[j][k] += (*prev_direction_W_[i])[j][k];
			}
		}
	}

}

void BackpropagationPerceptron::perceptron_learning(double** rows, double** results,
		size_t data_count, double** test_rows, double** test_results, size_t test_data_count) {
	size_t order[data_count];
	for (size_t i = 0; i != data_count; i++) {
		order[i] = i;
	}

	double base_learning_speed = learning_speed_;

	double errors_sum = evaluate_perceptron(test_rows, test_results, test_data_count);
	cout << errors_sum << endl;
	status_remember();
	double best_errors_sum = errors_sum;
	size_t number_of_iterations = 0;
	size_t last_succsess_iteration = 0;
	while (last_succsess_iteration + 200 > number_of_iterations) {
		learning_speed_ = errors_sum * base_learning_speed;
		++number_of_iterations;
		random_shuffle(order, order + data_count);
		for (size_t i = 0; i != data_count; i++) {
			delete[] evaluate(rows[order[i]]);
			teach(results[order[i]]);

		}
		errors_sum = evaluate_perceptron(test_rows, test_results, test_data_count);
		if (errors_sum < best_errors_sum) {
			best_errors_sum = errors_sum;
			cout << "iteration #" << number_of_iterations << ": " << errors_sum << endl;
			status_remember();
			last_succsess_iteration = number_of_iterations;
		}
	}

	status_recover();
	learning_speed_ = base_learning_speed;
}

double BackpropagationPerceptron::evaluate_perceptron(double** test_rows, double** test_results,
		size_t test_data_count) {
	double rmse = 0;
	for (size_t i = 0; i != test_data_count; i++) {
		double* assessment = evaluate(test_rows[i]);
		for (size_t j = 0; j != layer_neuron_count_[layers_count_ - 1]; j++) {
			assessment[j] -= test_results[i][j];
			rmse += assessment[j] * assessment[j];
		}
		delete[] assessment;
	}
	return rmse / test_data_count;
}

//private

void BackpropagationPerceptron::init(vector<size_t> &layers) {
	layers_count_ = layers.size();
	layer_neuron_count_ = new size_t[layers_count_];
	size_t max_layer_neuron_count = layers[0];
	for (size_t i = 0; i < layers_count_; i++) {
		max_layer_neuron_count = fmax(max_layer_neuron_count, layers[i]);
		layer_neuron_count_[i] = layers[i];
	}
	temp_row_ = new double[max_layer_neuron_count + 1];

	W_ = new Matrix*[layers_count_ - 1];
	prev_direction_W_ = new Matrix*[layers_count_ - 1];

	descent_gradient_ = new double*[layers_count_ - 1];
	answers_ = new double*[layers_count_];

	for (size_t i = 0; i < layers_count_ - 1; i++) {
		W_[i] = Matrix::create_rand_matrix(layer_neuron_count_[i + 1],
				layer_neuron_count_[i] + add_const_x_);
		*W_[i] -= 0.5;
		prev_direction_W_[i] = new Matrix(layer_neuron_count_[i + 1],
				layer_neuron_count_[i] + add_const_x_);

		descent_gradient_[i] = new double[layer_neuron_count_[i + 1]];
		answers_[i] = new double[layer_neuron_count_[i] + add_const_x_];
	}

	answers_[layers_count_ - 1] = new double[layer_neuron_count_[layers_count_ - 1]];
	errors_row_ = new double[layer_neuron_count_[layers_count_ - 1]];
}

void BackpropagationPerceptron::status_remember() {
	if (remembered_W_ == NULL) {
		remembered_W_ = new Matrix*[layers_count_ - 1];
		for (size_t i = 0; i < layers_count_ - 1; i++) {
			remembered_W_[i] = W_[i]->clone();
		}
	} else {
		for (size_t i = 0; i < layers_count_ - 1; i++) {
			delete (remembered_W_[i]);
			remembered_W_[i] = W_[i]->clone();
		}
	}
}

void BackpropagationPerceptron::status_recover() {
	assert(remembered_W_ != NULL);
	for (size_t i = 0; i < layers_count_ - 1; i++) {
		delete (W_[i]);
		W_[i] = remembered_W_[i]->clone();
		for (size_t j = 0; j != prev_direction_W_[i]->rows_count(); j++) {
			fill((*prev_direction_W_[i])[j], (*prev_direction_W_[i])[j] + prev_direction_W_[i]->columns_count(), 0.0);
		}
	}
}

} /* namespace perceptron */
