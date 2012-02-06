#include "perceptron/backpropagation_perceptron.h"
#include <vector>
#include <fstream>
#include <iostream>
int main() {
	using std::cout;
	using std::endl;
	using std::vector;
	using std::ifstream;
	using perceptron::BackpropagationPerceptron;

	vector<size_t> layers;
	layers.push_back(4);
	layers.push_back(5);
	layers.push_back(3);
	BackpropagationPerceptron per(layers, 0.5, 0.7);

	ifstream f("input.txt");
	double** data;
	data = new double*[150];
	double** results;
	results = new double*[150];
	for (size_t i = 0; i != 150; i++) {
		data[i] = new double[4];
		for (int j = 0; j != 4; j++) {
			f >> data[i][j];
		}
		int int_tmp;
		f >> int_tmp;
		results[i] = new double[3];
		results[i][int_tmp] = 1.0;
	}
	f.close();

	per.perceptron_learning(data, results, 120, data + 120, results + 120, 30);

	double err_sum = 0.0;
	for (size_t i = 0; i != 150; i++) {
		double* res = per.evaluate(data[i]);
		double tmp = 0.0;
		for (size_t j = 0; j != 3; j++) {
			tmp += (res[j] - results[i][j]) * (res[j] - results[i][j]);
		}
		if (tmp >= 0.5) {
			cout << i << endl;
		}
		err_sum += tmp;
		delete[] res;
	}
	cout << err_sum / 150.0 << endl;

	return 0;
}
