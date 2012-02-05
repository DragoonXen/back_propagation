/*
 * backpropagation_perceptron.cpp
 *
 *  Created on: Feb 4, 2012
 *      Author: dragoon
 */

#include "backpropagation_perceptron.h"

#include <stdlib.h>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>

namespace perceptron {

using std::copy;

BackpropagationPerceptron::BackpropagationPerceptron(vector<size_t> &layers) {
	learning_speed_ = 0.3;
	inertia_ = 0.3;
	add_const_x_ = true;
	init(layers);
}

BackpropagationPerceptron::~BackpropagationPerceptron() {
	for (size_t i = 0; i < layers_count_ - 1; i++) {
		delete (W_[i]);
		delete (prev_direction_W_[i]);
		delete (answers_[i]);
		delete (descent_gradient_[i]);
	}
	delete (answers_[layers_count_ - 1]);

	delete[] W_;
	delete[] prev_direction_W_;
	delete[] answers_;
	delete[] descent_gradient_;
	delete[] temp_row_;
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
	size_t index = layers_count_ - 2;
	for (size_t i = 0; i != layer_neuron_count_[index + 1]; i++) {
		descent_gradient_[index][i] = answers_[index + 1][i] * (1.0 - answers_[index + 1][i])
				* (expected_row[i] - answers_[index + 1][i]);
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
		for (size_t j = 0; j != prev_direction_W_[i]->rows_count_; j++) {
			for (size_t k = 0; k != prev_direction_W_[i]->columns_count_; k++) {
				(*prev_direction_W_[i])[j][k] = (*prev_direction_W_[i])[j][k] * inertia_
						+ learning_speed_ * descent_gradient_[i][j] * answers_[i][k];
				(*W_[i])[j][k] += (*prev_direction_W_[i])[j][k];
			}
		}
	}

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
}

} /* namespace hme_model */
