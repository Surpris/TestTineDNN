#include "Header.h"

using namespace std;
using namespace tiny_dnn;
using namespace tiny_dnn::layers;

// Basic sequential model.
network<sequential> create_model_1(const vector<vec_t> X, const vector<vec_t> y) {
	network<sequential> model;
	size_t n_neural = 100;
	model << fc(X[0].size(), n_neural) << activation::tanh()
		<< fc(n_neural, y[0].size()) << activation::softmax();
	return model;
}

// multilayer sequantial model with a dropout layer.
network<sequential> create_model_2(const vector<vec_t> X, const vector<vec_t> y) {
	network<sequential> model;
	size_t n_neural = 100;
	model << fc(X[0].size(), n_neural) << activation::tanh()
		<< fc(n_neural, n_neural) << activation::sigmoid()
		<< dropout(n_neural, 0.3)
		<< fc(n_neural, y[0].size()) << activation::softmax();
	return model;
}