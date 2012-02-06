#include "perceptron/backpropagation_perceptron.h"
#include <vector>
#include <fstream>
#include <iostream>
int main() {
	using std::cout;
	using std::endl;
	using std::vector;
	using std::ifstream;
	using std::fstream;
	using perceptron::BackpropagationPerceptron;

	vector<size_t> layers;
	layers.push_back(235);
	layers.push_back(10);
	layers.push_back(1);
	BackpropagationPerceptron per(layers, 0.05, 0.9, true);

	ifstream f("input.txt");
	double** data;
	const size_t rows_count = 97051;
	const size_t parameters_count = 235;
	data = new double*[rows_count];
	double** results;
	results = new double*[rows_count];
	for (size_t i = 0; i != rows_count; i++) {
		double tmp;
		f >> tmp;
		results[i] = new double[1];
		results[i][0] = tmp;

		data[i] = new double[parameters_count];
		for (size_t j = 0; j != parameters_count; j++) {
			f >> data[i][j];
		}
	}
	f.close();

	size_t test_count = rows_count / 10;
	size_t learn_count = rows_count - test_count;
	per.perceptron_learning(data, results, learn_count, data + learn_count, results + learn_count,
			test_count, 10, 0.7);

	fstream save_stream("base_model.bin", std::ios_base::out | std::ios_base::binary);
	per.save(save_stream);
	save_stream.close();

	double err_sum = 0.0;
	for (size_t i = 0; i != rows_count; i++) {
		double* res = per.evaluate(data[i]);
		double tmp = 0.0;
		tmp = (res[0] - results[i][0]) * (res[0] - results[i][0]);
		err_sum += tmp;
		delete[] res;
	}
	cout << err_sum / rows_count << endl;

	return 0;
}
