#include "perceptron/backpropagation_perceptron.h"
#include <vector>
int main() {
	using std::vector;
	using perceptron::BackpropagationPerceptron;

	vector<size_t> layers;
	layers.push_back(5);
	layers.push_back(4);
	layers.push_back(3);
	BackpropagationPerceptron *per = new BackpropagationPerceptron(layers);
	delete (per);
	return 0;
}
