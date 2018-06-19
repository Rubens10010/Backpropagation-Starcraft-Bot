#include <iostream>
#include <vector>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>       /* exp */
#include <fstream>
#include <algorithm>
#include <sstream>
#include <string>

using weights = std::vector<double>;
using features = std::vector<double>;
using layer = std::vector<weights>;
using neural_network = std::vector<layer>;
using outputs = std::vector<features>;
using training = std::vector<features>;
using testing = std::vector<features>;
using expected = std::vector<features>;
using dataset = std::vector<std::vector<double>>;

std::ofstream f;

void save_data(std::vector<double> &ms, std::string fileName){
    f.open(fileName);

    for (unsigned int i = 0; i < ms.size(); i++) {
    	f << i << " " << ms[i] << '\n';
    }
    f.close();
}

// returns a random float number between a and b
float random(float a, float b)		
{
	return  ((double)rand()  / (RAND_MAX)) + a + rand() % int(b-a);
}

// n_inputs: number of features
// n_lawyers: number of hidden lawyers + output lawyer
// n_neurons: neurons per lawyer
// @returns: neural network initialized with random weights
neural_network create_neural_network(int n_inputs, int n_layers, std::vector<int> & n_neurons)
{
	if (n_layers != n_neurons.size())	std::cerr << "Bad neural network parameters!" << std::endl;
	// -- Neural network basic structure --
	neural_network network(n_layers);

	// -- Neural network hidden lawyers and output lawyer initialization -- 
	for (int k = 0; k < n_layers; k++)
	{
		network[k] = layer(n_neurons[k]);		//! allocate space for neurons
		for (int j = 0; j < network[k].size(); j++)	//! neuron by neuron
		{
			// initialize weights with random values
			network[k][j] = (k == 0)?weights(n_inputs + 1): weights(n_neurons[k - 1] + 1);		//! allocate space for weights vector + bias (last index)
			for (int i = 0; i < network[k][j].size(); i++)
			{
				network[k][j][i] = random(0, 1.0);
			}
		}
	}
	return network;
}

// Receive an initialized neural network
void print_network(neural_network &nn)
{
	// -- Prints out a neural network in console --
	std::cout << "----------------------------------------------------------------------";
	std::cout << "\n\t\tNeural Network\n";
	for (int k = 0; k < nn.size(); k++)	// Lawyer by lawyer
	{
		std::cout << "Layer" << k << ":\n";
		for (int j = 0; j < nn[k].size(); j++)		// Neuron by neuron
		{
			std::cout << "\tNeuron" << j << ": [ Weights: {";
			for (int i = 0; i < nn[k][j].size(); i++)		// Print weights
			{
				std::cout << nn[k][j][i] << ", ";
			}
			std::cout << "} ]\n";
		}
	}
	std::cout << "----------------------------------------------------------------------";
	std::cout << std::endl;
}

// w: Weights vector of a neuron
// in: Input vector to a neuron
float energy(weights &w, features &in)
{
	// Calculates total energy flowing into the neuron
	if (w.size() != in.size() + 1) 	std::cerr << "Can't compute dot product because vectors are not of same length" << std::endl;

	float activation = w.back();	// bias is multiply by 1
	for (int i = 0; i < w.size() - 1; i++)
	{
		activation += w[i] * in[i];
	}
	return activation;
}

// Neuron activation function
// Uses sigmoid function to calculate the output value of the neuron
// energy_activation:	Value of energy from neuron (weights*input)
float activation_sigm(float energy_activation)
{
	return 1.0 / (1.0 + exp(-energy_activation));
}

// -- Forward propagation function --
// Forward propagate input to a network output
// network: Initialized neural network
// input: Training or test sample for feeding the network
// @ returns: Prediction in shape of vector from neural network
outputs forward_propagate(neural_network &network, features &input)
{
	features output = input;		// Here the output of the network will be stored
	outputs outputs_m;
	for (int l = 0; l < network.size(); l++)		// for lawyer in network
	{
		features curr_inputs;
		for (int j = 0; j < network[l].size(); j++)
		{
			float act_energy = energy(network[l][j] , output);
			float n_output = activation_sigm(act_energy);			// current neuron output
			curr_inputs.push_back(n_output);
		}
		output = curr_inputs;		// This vector stores the current outputs of the lawyer
		outputs_m.push_back(output);
	}
	return outputs_m;		// -- Final Output (Prediction) --
}

// -- Deltas for output lawyer --
// output: outputs from output lawyer
// target: targeted outputs from training sample
// @returns: deltas from output lawyer
features delta_output(features &output, features &target)
{
	features deltas = output;
	for (int i = 0; i < deltas.size(); i++)
	{
		deltas[i] = (target[i] - output[i])*output[i] * (1.0 - output[i]);
		//std::cout << "delta output " << i << ":" << target[i]  <<" "<< output[i] << std::endl;	
	}
	return deltas;
}

// -- Deltas for hidden lawyer --
// output: outputs from output lawyer
// target: targeted outputs from training sample
// w_lawyer: matrix of weights from next lawyer
// deltas: deltas from next lawyer
// @returns: delta from a neuron in hidden lawyer considering next lawyer
features delta_hidden(features &output, layer &w_layer, features &deltas)
{
	features delta = output;
	for (int j = 0; j < output.size(); j++)	// neuron by neuron in current lawyer
	{
		float error = 0.0;
		for (int n = 0; n < w_layer.size(); n++)	// neuron by neuron in next lawyer
			error += w_layer[n][j] * deltas[n];
		delta[j] = error*output[j] * (1.0 - output[j]);
	}
	return delta;
}

// Calculates deltas(change factor for changing weights) for each neuron in neural network
outputs calculate_deltas(neural_network &network, features &real, outputs &output)
{
	outputs deltas(network.size());			// deltas for each neuron in each lawyer
	for (int l = network.size()-1; l >= 0; l--)	// Loop through all lawyers
	{
		if (l == network.size() - 1)			// -- output lawyer delta -- 
		{
			deltas[l] = delta_output(output[l],real);
		}
		else
		{
			// -- hidden lawyers deltas --
			deltas[l] = delta_hidden(output[l], network[l+1], deltas[l+1]);
		}
	}
	return deltas;
}

// Backward propagate output error
void backward_propagate(neural_network &network, features &sample, outputs &output, outputs &deltas, float l_rate)
{
	features input = sample;

	for (int l = 0; l < network.size(); l++)	// Loop through all lawyers
	{
		if (l != 0)
			input = output[l - 1];
		for (int j = 0; j < network[l].size(); j++)
		{
			for (int i = 0; i < input.size(); i++)
			{
				network[l][j][i] += l_rate*deltas[l][j]*input[i];	// decrease error for weights
			}
			network[l][j].back() += l_rate*deltas[l][j];	// decrease error for bias
		}
	}
}

// Neural network error
float nn_error(features &outputs, features &expected)
{
	float total = 0;
	for (int i = 0; i < outputs.size(); i++)
		total += ((expected[i] - outputs[i])*(expected[i] - outputs[i]))*1/2;
	return total;
}

// -- Specific for example --
int belongs_to(features &output)
{
	// setosa
	if(round(output[0]) && !round(output[1]) && !round(output[2]))
	{	return 0;
	}else if(!round(output[0]) && round(output[1]) && !round(output[2]))
	{	return 1;	// versicolor
	}else if(!round(output[0]) && !round(output[1]) && round(output[2]))
	{	return 2;	// virgilia
	}else{
		return 3;	// unknown
	}
}

// -- specific for example
void confusion_matrix(std::vector<int> &results, std::vector<int> &real)
{
	// confusion matrix for iris database
	std::cout << "CONFUSION MATRIX IRIS" << std::endl;
	std::cout << "*****************************" << std::endl;
	std::vector<std::vector<int>> r(3,std::vector<int>(3,0));
	double sum = 0.;
	for(int i = 0; i < results.size(); i++)
	{
		r[real[i]][results[i]]++;
	}
	std::cout << "     | 0 | 1 | 2 | \n";
	for(int i = 0; i < 3; i++)
	{
		std::cout << "------------------" << std::endl;
		std::cout << "| " << i << " |";
		for(int j = 0; j < 3; j++)
		{
			if(i==j)
				sum+= r[i][j]*1e2/25.;
			std::cout << r[i][j]*1e2/25. << "% / ";
		}
		std::cout << "\n------------------" << std::endl;
	}
	std::cout << "Average accuracy: " << sum/3 << std::endl;
	std::cout << "\n*****************************" << std::endl;
}

// -- --
int print_prediction(features &output)
{
	std::cout << " ................................. " << std::endl;
	std::cout << "\nPrediction for current iteration: \n" << std::endl;
	int c = belongs_to(output);

	if(c == 0)
		std::cout << "setosa" << std::endl;
	else if(c == 1)
		std::cout << "versicolor" << std::endl;
	else if(c == 2)
 		std::cout << "vergilia" << std::endl;
	else
		std::cout << "unknown" << std::endl;
	for (int i = 0; i < output.size(); i++)
	{
		std::cout << output[i] << " ";
	}
	std::cout << "\n ................................." << std::endl;
	return c;
}

/*
int print_prediction(features &output){
	std::cout << "------------------------------" << std::endl;
	std::cout << "Prediction for current iteratiion" << std::endl;
	if((round(output[0]) || round(output[1])) && !(round(output[0]) && round(output[1])))
	{
		std::cout << "1" << std::endl;
		return 1;
	}
	else{
		std::cout << "0" << std::endl;
		return 0;
	}
	std::cout << "\n----------------------------------" << std::endl;
}*/

// Returns the clases that belongs to the output vector
std::vector<int> prediction(outputs &output)
{
	std::vector<int> classes(output.size());
	for(int i = 0; i < output.size(); i++)
	{
		//std::cout << output[i][0] << " " << output[1] << " " << output[2] << std::endl;
		int c = belongs_to(output[i]);
		classes[i] = c;
	}
	return classes;
}

// Use dataset to train neural network
void train_network(neural_network &network, training &examples, expected &r_outputs, float l_rate, int n_iterations)	// + n_outputs
{
	std::vector<double> ms;
	//std::vector<int> results(examples.size());	// predictions
	for (int i = 0; i < n_iterations; i++)
	{ //.................................

		std::cout << "---------------------------------" << std::endl;
		std::cout << "iteration " << i << ":" << std::endl;
		std::cout << "examples size " << examples.size() << std::endl;		
		float err = 0;
		outputs output;
		for (int j = 0; j < examples.size(); j++) 
		{
			output = forward_propagate(network, examples[j]);
			err = nn_error(output.back(), r_outputs[j]);
			auto deltas = calculate_deltas(network, r_outputs[j], output);
			backward_propagate(network, examples[j],output,deltas,l_rate);
		}
		print_prediction(output.back());		// final output for prediction of last example
		std::cout << "Current error: " << err << std::endl;
		std::cout << "---------------------------------\n" << std::endl;
		ms.push_back(err);
	}
	save_data(ms, "error_data.csv");
	//return results;
}

// -- Test Network --
//  Use data to test network

std::vector<int> test_network(neural_network &network, testing &test_samples/*, expected &r_outputs*/)
{
	std::vector<int> results;
	std::cout << "************************************\n";
	std::cout << "Testing Network\n";
	for(int i = 0; i < test_samples.size(); i++)
	{
		std::cout << "Input: {";
		for(int j = 0; j < test_samples[i].size(); j++)
			std::cout << test_samples[i][j] << ",";
		std::cout << "}\nOutput: {";
		outputs output = forward_propagate(network, test_samples[i]);
		int c = print_prediction(output.back());
		results.push_back(c);
		std::cout << "}\n";

	}
	std::cout << "************************************\n";
	return results;
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

//-------------------------------------------------------
void normalizer(dataset &tr){

    for(int j = 0; j < tr[0].size(); j++){
    	std::vector<double> tmp;
		for(int i = 0; i < tr.size(); i++){
			tmp.push_back( tr[i][j] );
		}
		std::sort( tmp.begin(), tmp.end() );
		double max = tmp[tmp.size()-1];
		double min = tmp[0];
		for(unsigned i=0; i<tr.size(); ++i){
			tr[i][j] = (tmp[i]-min) / (max - min); 
		}
	}
}

// ------------------------------------------------------
void split_dataset(dataset &d, expected &e, training &tr, testing &te, expected &e1, expected &e2, float for_train, int n_clases)
{
	int n = d.size();	// total of examples
	std::cout << n << std::endl;
	// -- Split for training neural network --
	int chunk = n/n_clases;
	std::cout << "chunks: "<< chunk << std::endl;
	int t = chunk*(for_train/100);	// for training
	std::cout << "For training: " << t << std::endl;
	int diff = chunk - t;	// for testing
	std::cout << "dif: " << diff << std::endl;

	tr.resize(t*n_clases);
	te.resize(diff*n_clases);
	e1.resize(t*n_clases);
	e2.resize(diff*n_clases);

	int t_start = 0;
	int te_start = 0;
	for(int i = 0; i < n_clases; i++)
	{
		int start = i*chunk;
		int fin = (i+1)*chunk;
		int s = start + t;

		//std::cout << "start: "<< start << " fin: " << fin << " s: "<< s << std::endl;
		std::cout << "i: " << i << std::endl;
		for(int j = start; j < s; j++)
		{
			std::cout << "j: " << j << std::endl;
			tr[(j - start) + t_start] = d[j];
			//std::cout << "\nTraining index:  "<< (j-start) + t_start << " "<< tr[(j-start) + t_start][0] << " " << tr[j-start][1] << " " << tr[j-start][2] << " " << tr[j-start][3] << " " << std::endl;
			//std::cout << "ej: "<< e1.size() << " "<<e.size()<<std::endl;
			e1[(j - start) + t_start] = e[j];	// problem
			//std::cout << "\nt_real index:  "<< (j-start) + t_start << " "<< e1[(j - start) + t_start][0] << " " << e1[(j - start) + t_start][1] << " " << e1[(j - start) + t_start][2] << std::endl;	
		}
		//std::cout << "next" << s << " "<<fin <<" "<<t_start<<std::endl;
		for(int k = s; k < fin; k++)
		{
			//std::cout << "k: " << k << " "<<t_start <<" "<< e.size() << " " << d.size()<< std::endl;
			te[(k - s) + te_start] = d[k];
			//std::cout << "\nTesting index:  "<< (k-s) + t_start << " "<< te[(k - s) + t_start][0] << " " << tr[(k - s) + t_start][1] << " " << te[(k - s) + t_start][2] << " " << te[(k - s) + t_start][3] << " " << std::endl;
			//std::cout << "\nTesting index:  "<< (k-s) + t_start << " "<< d[k][0] << " " << d[k][1] << " " << d[k][2] << " " << d[k][3] << " " << std::endl;
			e2[(k - s) + te_start] = e[k];
			//std::cout << "\nreal index:  "<< (k-s) + t_start << " "<< e2[(k - s) + t_start][0] << " " << e2[(k - s) + t_start][1] << " " << e2[(k - s) + t_start][2] << std::endl;
			//std::cout << "out " << std::endl;
		}
		t_start += t;
		
		te_start += diff;
	}
	//std::cout << tr.size() << " " << te.size() << " " << e1.size() << " " << e2.size() << std::endl;
}



int main()
{
	/* initialize random seed: */
	srand(time(NULL));


	/*training examples = { {2.7,2.5},{1.4,2.3},{1.3,1.8} };
	testing test_examples = {{1.2,1.5},{3,2.8},{0.8,1.3}};
	expected r_outputs = { {1,0} ,{0,1},{0,1}};*/
	/*training examples = { {0,0}, {1,0}, {0,1}, {1,1}};
	testing test_examples = { {0,0}, {1,0}, {0,1},{1,1}};
	expected r_outputs = { {0,0}, {1,0}, {1,0}, {0,0}};

	int n_inputs = examples[0].size();
	int n_outputs = r_outputs[0].size();	// use set

	std::vector<int> n_neurons = { 2,2,n_outputs };
	neural_network nn = create_neural_network(n_inputs, n_neurons.size(), n_neurons);
	print_network(nn);

	int i;
	std::cout << "Number of iterations: ";
	std::cin >> i;
	train_network(nn, examples, r_outputs, 1.5, i);
	test_network(nn,test_examples);
	print_network(nn);
*/
	// -------------------------------------
	dataset d;
    expected e;
	read_file("Iris.data", d, e);

	training examples;
	testing test_examples;
	expected out_tr;
	expected out_te;

	std::cout << "sizes: " << d.size() <<" "<< e.size()<< std::endl;
	// 7th parameter: percentaje for training 
	split_dataset(d, e, examples, test_examples, out_tr, out_te, 50, 3);
	normalizer(examples);
	normalizer(test_examples);

	int n_inputs = examples[0].size();
	int n_outputs = out_tr[0].size();	// use set

	std::cout << "Inputs " << n_inputs << " Outputs" << n_outputs << std::endl;
	std::vector<int> n_neurons = {6,6,n_outputs};
	neural_network nn = create_neural_network(n_inputs, n_neurons.size(), n_neurons);
	print_network(nn);

	int i;
	std::cout << "Number of iterations: ";
	std::cin >> i;
	train_network(nn, examples, out_tr, 0.5, i);
	std::vector<int> results = test_network(nn,test_examples);
	std::vector<int> reals = prediction(out_te);
	//for(int i = 0; i < results.size(); i++)
	//	std::cout << "real: " << reals[i] <<" output: " << results[i] << std::endl; 
	confusion_matrix(results,reals);
	//print_network(nn);

    system("gnuplot -p 'plotFile'");

	return 0;
}
