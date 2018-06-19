#ifndef UTILS_H
#define UTILS_H
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
//#include <>

using weights = std::vector<double>;
using features = std::vector<double>;
using layer = std::vector<weights>;
using neural_network = std::vector<layer>;
using outputs = std::vector<features>;
using training = std::vector<features>;
using testing = std::vector<features>;
using expected = std::vector<features>;
using dataset = std::vector<std::vector<double>>;
using values = std::vector<double>;
using slices = std::vector<std::vector<int>>;
using chainedMatrix = std::vector<layer>;
using grads = std::vector<std::vector<double>>;
//srand(time(0));

// returns a random float number between a and b
inline float random(float a, float b)		
{
	return  ((double)rand()  / (RAND_MAX)) + a + rand() % int(b-a);
}

inline double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

template< class T >
inline void reorder(std::vector<T> &v, std::vector<size_t> &order )  {   
    for ( int s = 1, d; s < order.size(); ++ s ) {
        for ( d = order[s]; d < s; d = order[d] ) ;
        if ( d == s ) while ( d = order[d], d != s ) swap( v[s], v[d] );
    }
}

inline void cut_slice(training &X_,expected &y_, training &slice,   expected &slice_y, std::vector<int> &batch_slice)
{
	slice.clear();
	for(int i = batch_slice[0]; i < batch_slice[1]; i++)
	{
			slice.push_back(X_[i]);
			slice_y.push_back(y_[i]);
	}
}

inline double dot_product(std::vector<double> &a, std::vector<double> &b){
	double result = 0.0;
	for(int i = 0; i < a.size(); i++)
	{
		result += a[i]*b[i];
	}
	return result;
}

inline layer safe_sparse_dot(layer &activations, layer &coefs)
{
	// dot products between each row of activation and each from coefs (neuron)
	layer activation_out(activations.size());
	int i, j;
	for(i = 0; i < activations.size(); i++)
	{
		activation_out.resize(coefs.size());
		for(j = 0; j < coefs.size(); j++)
		{
			activation_out[i][j] = dot_product(activations[i],coefs[j]);
		}		
	}
	return activation_out;
}

inline void sum_intercepts_(layer& activations, weights& intercepts)
{
	for(int i = 0; i < activations.size(); i++)
		for(int j =0; j < activations[i].size(); j++)
			activations[i][j] += intercepts[j];
}

// Neuron activation function
// Uses sigmoid function to calculate the output value of the neuron
// energy_activation:	Value of energy from neuron (weights*input)
// logistic activation
inline float activation_sigm(float energy_activation)
{
	return 1.0 / (1.0 + exp(-energy_activation));
}

// calculate activation function for values inplace
inline void calculate_activations(layer& activations, int type)
{
	switch(type)
	{
		case 0: //logistic
		{
			for(int i = 0; i < activations.size(); i++)
				for(int j =0; j < activations[i].size(); j++)
					activations[i][j] = activation_sigm(activations[i][j]);
			break;
		}
		default:
			break;
	}
}

/*inline double loss_func(y,activations[activations.size() -1])
{

}*/


float nn_error(features &outputs, features &expected)
{
	float total = 0;
	for (int i = 0; i < outputs.size(); i++)
		total += ((expected[i] - outputs[i])*(expected[i] - outputs[i]))*1/2;
	return total;
}

#endif
