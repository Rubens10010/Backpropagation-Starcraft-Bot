#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <iostream>
#include <vector>
#include <math.h>       /* exp */
#include <fstream>
#include <algorithm>
#include <random>	// rng
#include <sstream>
#include <string>
#include <optional>
//#include <tuple>
#include "utils.h"

using sizes = std::vector<int>;
using name = std::string;

template<typename T>
using opt = std::optional<T>;

const double epsilon = 1e-8;

enum class Activation : char { IDENTITY, LOGISTIC, TANH, RELU}; // default Relu

enum class Solver : char { SGD, ADAM }; // stocasthic solvers, default Adam
enum class LearningRate : char { CONSTANT, INVSCALING, ADAPTIVE};	// default constant
/*
    MLPClassifier trains iteratively since at each time step
    the partial derivatives of the loss function with respect to the model
    parameters are computed to update the parameters.
    It can also have a regularization term added to the loss function
    that shrinks model parameters to prevent overfitting.
    This implementation works with data represented as dense numpy arrays or
    sparse scipy arrays of floating point values.
*/
class NeuralNetwork
{
	public:
	//private:
		size_t n_inputs;
		size_t n_layers;
		sizes n_neurons;

		const sizes hidden_layer_sizes;
		Activation activation;
		Solver solver;
		LearningRate learning_rate;
		int batch_size;
		double alpha;
		double learning_rate_init;
		double tol;
		double power_t;
		int max_iter;
		int n_iter_no_change;
		double beta1;
		double beta2;
		double validation_fraction;
		bool shuffle;
		bool verbose;
		bool early_stopping;

		// classes_: array of array of shape(n_classes,)
		float loss_;	// current loss computed
		neural_network coefs_; //: ith element in the list represents the weight matrix corresponding to layer i. list, length n_layers - 1
		layer intercepts_; //: list, length n_layers - 1. the ith element in list represents the bias vector corresponding to layer i + 1
		int t_;
		int n_iter_; // number of iterations the solver has ran
		int n_layers_;	// number of layers
		int n_outputs_;	// number of outputs
		Activation out_activation_;	// name of output activation function		

		values loss_curve_;
		int _no_improvement_count;
		double best_loss_;

		public:

		NeuralNetwork(opt<sizes> hls = std::nullopt, opt<Activation> act = std::nullopt, opt<Solver> sol = std::nullopt, opt<int> maxit = std::nullopt, opt<double> l_rate_init = std::nullopt, opt<double> val_frac = std::nullopt) :
						hidden_layer_sizes(hls.value_or(sizes(1,100))),
						activation(act.value_or(Activation::RELU)),
						solver(sol.value_or(Solver::ADAM)),
						alpha(0.0001), batch_size(200),
						learning_rate(LearningRate::CONSTANT),
						learning_rate_init(l_rate_init.value_or(0.001)),
						tol(1e-4), power_t(0.5), max_iter(maxit.value_or(200)),
						shuffle(true), verbose(false), early_stopping(false),
						validation_fraction(val_frac.value_or(0.1)),
						beta1(0.9), beta2(0.999), n_iter_no_change(10) {}

		~NeuralNetwork();

		// Main Functions
		bool _initialize(expected y, sizes layer_units);
		void _init_coef(layer &coef_init, weights &intercept_init, size_t n_in, size_t n_out);
		void print_network();
		//bool _init_coef(self, fan_in, fan_out):
		void _fit(training X, expected y, bool incremental=false);
		void _fit_stochastic(training X, expected y,chainedMatrix &_activations, chainedMatrix &deltas, chainedMatrix& coef_grads,grads &intercept_grads, sizes layer_units,bool incremental);
		void shuffle_data(training &X, expected &y);
		slices gen_batches(int n_samples, int batch_size);

		double _backprop( training &                       X, expected &y,chainedMatrix& activations,chainedMatrix& deltas, chainedMatrix& coef_grads, grads& intercept_grads);

		chainedMatrix _forward_pass(chainedMatrix& activations);
		/*def _validate_hyperparameters(self):
		def fit(self, X, y):
		def _predict(self, X):*/
};

#endif
