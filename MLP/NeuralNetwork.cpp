#include "NeuralNetwork.h"

// n_inputs: number of features
// n_lawyers: number of hidden lawyers + output lawyer
// n_neurons: neurons per lawyer

// hidden_layer_sizes: tuple, length = n_layers - 2, default(100,) the ith element represents the number of neurons in the ith hidden layer.

// activation: {'identity', 'logistic', 'tanh', 'relu'}, default 'relu'. Activation function for the hidden layer.

// solver: {'lbfgs', 'sgd', 'adam'}, default 'adam' The solver for weight optimization

// alpha: float, optional, default 0.0001. L2 penalty (regularization term) parameter.

// batch size: int, optional, default 'auto'. Size of minibatches for stochastic optimizers. 'auto': batch_size = min(200, n_samples)

// learning rate: {'constant', 'invscaling', 'adaptive'}, default 'constant'. Learning rate schedule for updates.

// learning rate init: double, optional, default 0.001. The initial learning rate used. It controls the step-size in updating the weights. Only used when solver='sgd' or 'adam'

// NO debe entrar
// power_t: double, optional, default 0.5. The exponent for inverse scaling learning rate. It is used in updating effective learning rate when the learning_rate is set to 'invscaling'. Only used when solver = 'sgd' ***** no used

// max_iter: int, optional, default 200. Maximun number of iterations. The solver iterates until convergence (determined by 'tol') or this number of iterations. For stochastic solvers ('sgd', 'adam'), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps.

// shuffle: bool, optional, default True. Wether to shuffle samples in each iteration. Only used when solver = 'sgd' or 'adam'

// random_state: Int, random state is the seed used by the random number generator, if randomState instance, is the random number generator. None the random number generator is the randomState instance used in random.

// tol: float, optional, default 1e-4. Tolerance for the optimization. When the loss or score is not improving by at least 'tol' for n_iter_no_change' consecutive iterations, unless 'learning rate' is set to 'adaptive'. convergence is considered to be reached and training stops.

// verbose: bool, optional, default False. Whether to print messages to stdout.

// warm start: default False
// momentum: default 0.9 for gradient descent update. when solver = 'sgd'
// nesterovs_momentum, default True. only used when solver = 'sgd' and momentum > 0
// early_stopping: bool, default False. Use early stopping to terminate training when validation score is not improving. If set to true, it will set aside 10% of training data as validation and terminate training when validation score is not improving by at least 'tol' for 'n_iter_no_change' consecutive epochs. Only effective when solver = 'sgd' or 'adam'.

// validation_fraction: float, optional, default 0.1 the proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True.

// beta_1: float, optional, default 0.9. Exponential decay rate for estimates of first moment vector in adam, should be in [0,1>. when solver = 'adam'

// beta_2: float, optional, default 0.999. Exponential decaay rate for estimates of second moment vector in adam.

// epsilon: Value for numerical stability in adam.

// n_iter_no_change: Int, optional, default 10. Maximun number of epochs to not meet 'tol' improvement. when solver is sgd or adam.

// @returns: neural network initialized with random weights
/*NeuralNetwork::NeuralNetwork(const sizes &n_neurons, Activation activation, Solver solver, double alpha, int batch_size, LearningRate learning_rate, double learning_rate_init, double tol, double power_t, int max_iter, bool shuffle, bool verbose, bool early_stopping, double validation_fraction, double beta1, double beta2, int n_iter_no_change)
{
	//this->n_inputs = n_inputs;
	//this->n_layers = n_layers;
	this->n_neurons = n_neurons;
	this->activation = activation;
	this->solver = solver;
	this->batch_size = batch_size;
	this->learning_rate = learning_rate;
	this->alpha = alpha;
	this->learning_rate_init = learning_rate_init;
	this->tol = tol;
	this->power_t = power_t;
	this->max_iter = max_iter;
	this->n_iter_no_change = n_iter_no_change;
	this->beta1 = beta1;
	this->beta2 = beta2;
	this->validation_fraction = validation_fraction;
	this->shuffle = shuffle;
	this->verbose = verbose;
	this->early_stopping = early_stopping;
	// loss: log_loss
}*/

NeuralNetwork::~NeuralNetwork()
{

}

//_initialize: set all attributes, allocate weights etc first call
// Initialize parameters
// y:	expected outputs from network
// layer_units: Array with number of neurons per layer [input, hidden, output]
bool NeuralNetwork::_initialize(expected y, sizes layer_units)
{
	this->n_iter_ = 0;
	this->t_ = 0;
	this->n_outputs_ = y[0].size();

	// Total number of layers	
	this->n_layers_ = layer_units.size();

	// out activation 
	this->out_activation_ = Activation::LOGISTIC;

	// Initialize coefficient and intercepts layers
	this->coefs_.clear();
	this->intercepts_.clear();

	layer coef_init;
	weights intercept_init;
	for(int i = 0; i < this->n_layers_ - 1; i++)
	{
		coef_init.resize(layer_units[i]);
		intercept_init.resize(layer_units[i+1]);
		_init_coef(coef_init, intercept_init, layer_units[i], layer_units[i+1]);
		coefs_.push_back(coef_init);
		intercepts_.push_back(intercept_init);		
	}

	loss_curve_.clear();
	this->_no_improvement_count = 0;
	if(!this->early_stopping)
		this->best_loss_ = 99999999999;
}

void NeuralNetwork::_init_coef(layer &coef_init, weights &intercept_init, size_t n_in, size_t n_out)
{
	double factor = 6.;
	if(this->activation == Activation::LOGISTIC)
		factor = 2.;

	double init_bound = sqrt(factor/(n_in + n_out));
	// generate weights and bias
	//layer coef_init(fan_in);
	//weights intercept_init(fan_out);

	//coef_init.clear();
	//intercept_init.clear();

	for(int i = 0; i < n_in; i++){
		coef_init[i] = weights(n_out);
		for(int j = 0; j < n_out; j++){
			//coef_init[i][j] = random(-init_bound, init_bound);
			coef_init[i][j] = fRand(-init_bound, init_bound);
		}
	}
	
	for(int j = 0; j < n_out; j++)
		intercept_init[j] = fRand(-init_bound, init_bound);
}

// Receive an initialized neural network
void NeuralNetwork::print_network()
{
	// -- Prints out a neural network in console --
	std::cout << "----------------------------------------------------------------------";
	std::cout << "\n\t\tNeural Network\n";
	/*for(int i = 0; i < this->n_layers_ - 1; i++)
	{
		coef_init.resize(layer_units[i]);
		intercept_init.resize(layer_units[i+1]);
		_init_coef(coef_init, intercept_init, layer_units[i], layer_units[i+1]);
		coefs_.push_back(coef_init);
		intercepts_.push_back(intercept_init);		
	}*/
	for (int k = 0; k < this->n_layers_ - 1; k++)	// Lawyer by lawyer
	{
		std::cout << "\nLayer " << k << ":\n";
		for (int j = 0; j < coefs_[k].size(); j++)		// Neuron by neuron
		{
			std::cout << "\tNeuron" << j << ": [ Weights: {";
			for (int i = 0; i < coefs_[k][j].size(); i++)		// Print weights
			{
				if(i > 4)
				{
					i = coefs_[k][j].size();
					std::cout << ",...(" << i << ")";
					continue;
				}
				std::cout << coefs_[k][j][i] << ", ";
			}
			std::cout << "} Bias: " << intercepts_[k][j] <<" ]\n";
		}
	}
	std::cout << "----------------------------------------------------------------------";
	std::cout << std::endl;
}

void NeuralNetwork::_fit(training X, expected y, bool incremental)
{
	// list
	auto hidden_layer_sz = this->hidden_layer_sizes;
	
	// validar parametros
	// validar sizes in hidden layers
	// Validate Inputs
	// Ensure is 2d	
	
	int n_samples = X.size();
	int n_features = X[0].size();

	this->n_outputs_ = y[0].size();
	sizes layer_units = {n_features};
	for(int i = 0; i < hidden_layer_sizes.size(); i++)
		layer_units.push_back(hidden_layer_sizes[i]);
	layer_units.push_back(this->n_outputs_);

	// check random state, seed
	// first time training model
	if(coefs_.empty())
		_initialize(y, layer_units);

	// lbfgs doesnt support mini-batches
	this->batch_size = (n_samples < 200)?n_samples:200;
	
	// Initialize activations lists	
	// initialize deltas, coef_grads, intercept_grads
	chainedMatrix _activations;
	_activations.push_back(X);
	// add activations matrix for each layer default to cero
	for(int i = 1; i < layer_units.size(); i++)
		_activations.push_back(layer(batch_size, weights(layer_units[i])));

	chainedMatrix deltas = _activations; // copy
	chainedMatrix coef_grads;
	for(int i = 0; i < layer_units.size() - 1; i++)
	{
		// [1,2,3] -> matrices([(1,2),(2,3)])
		coef_grads.push_back(layer(layer_units[i],weights(layer_units[i+1])));
	}
	
	grads intercept_grads;
	for(int i = 1; i < layer_units.size(); i++)
	{
		intercept_grads.push_back(std::vector<double>(layer_units[i],0));	
	}
	
	_fit_stochastic(X,y,_activations,deltas,coef_grads,intercept_grads,layer_units,incremental);
}

void NeuralNetwork::_fit_stochastic(training X, expected y,chainedMatrix& _activations, chainedMatrix &deltas, chainedMatrix& coef_grads,grads &intercept_grads, sizes layer_units,bool incremental)
{
	if(!incremental)
	{
		// get/set optimizer values
	}

	// if early_stopping divide in training validation with validation_fract
	int n_samples = X.size();
	// make sure batch_size is settle
	training X_ = X;
	expected y_ = y;
	for(int it = 0; it < this->max_iter; it++)
	{
		shuffle_data(X_,y_);
		double accumulated_loss = 0.0;
		slices s = gen_batches(n_samples, batch_size);
		training slice;
		expected slice_y;	
		for(int i = 0; i < s.size(); i++)
		{
			std::vector<int> batch_slice = s[i];
			cut_slice(X_,y_, slice,slice_y, batch_slice);
			//backprop in batch slice
			//accumulated_loss += batch_loss*
			//update weights, grads
		}
		this->n_iter_ += 1;
		this->loss_ = accumulated_loss/X.size();
		this->t_ += n_samples;
		this->loss_curve_.push_back(this->loss_);
		if(this->verbose)
			std::cout << "Iteration " << n_iter_ << ", loss = " << loss_ <<std::endl;

		//# update no_improvement_count based on training loss or
	}	
}

double NeuralNetwork::_backprop( training &X, expected &y, chainedMatrix& activations,chainedMatrix& deltas, chainedMatrix& coef_grads, grads& intercept_grads)
{
	int n_samples = X.size();
	activations = this->_forward_pass(activations);
	// Loss binary_log_loss
	loss = loss_func(y,activations.back());
	return 0.0;
}

// Perform a forward pass on the network computing the values of the neurons in hidden layers and output layer
chainedMatrix NeuralNetwork::_forward_pass(chainedMatrix& activations)
{
	//hidden_activation = function(this->activation);
	if(this->activation == Activation::LOGISTIC)
	{
		//pointer to function	
	}
	// iterate over hidden layers
	int i;
	for(i = 0; i < this->n_layers_ - 1; i++)
	{
		activations[i + 1] = safe_sparse_dot(activations[i],this->coefs_[i]);
		sum_intercepts_(activations[i + 1], this->intercepts_[i]);
		// calculate hidden layer activations
		if(this->activation == Activation::LOGISTIC)
		{
			calculate_activations(activations[i+1], 0); // 0: LOGISTIC
		}
	}
	// output layer activations
	if(this->out_activation_ == Activation::LOGISTIC){
		calculate_activations(activations[i+1], 0); // 0: LOGISTIC
	}
	return activations;
}


void NeuralNetwork::shuffle_data(training &X, expected &y)
{
	std::vector<size_t> idx(X.size());
	std::iota(idx.begin(), idx.end(), 0);
	
	std::random_shuffle(idx.begin(), idx.end());

	reorder<features>(X, idx);
	reorder<features>(y,idx);
}

slices NeuralNetwork::gen_batches(int n_samples, int batch_size)
{
	int start = 0;
	int end;
	slices s;
	for(int i = 0; i < int(n_samples/batch_size); i++)
	{
		end = start + batch_size;
		s.push_back({start,end});
		start = end;
	}
	if(start < n_samples)
		s.push_back({start, n_samples});

	return s;
}

// ---------------- file input -----------------
void read_file(std::string f_name, dataset &data, expected &out)
{
	std::ifstream file(f_name);
	std::string line;

	int features, classes;
	file >> features;
	file >> classes;
	std::cout << features << " -> " << classes << std::endl;
	std::getline(file,line);

        while(std::getline(file, line))
        {
	       std::vector<double> lineData;
		std::vector<double> lineOutputs;
		std::stringstream lineStream(line);
	       double value;
		int f = features;

	       while(f > 0 && lineStream >> value)
	       {
		   f--;
	           lineData.push_back(value);
	       }

		int c  = classes;
		while(c > 0 && lineStream >> value)
	       {
		   c--;
	           lineOutputs.push_back(value);
	       }

	       data.push_back(lineData);
	       out.push_back(lineOutputs);
       }
}
