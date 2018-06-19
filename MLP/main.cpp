#include <iostream>
#include "NeuralNetwork.h"

int main()
{
	srand(time(0));
	//NeuralNetwork nn(Activation activation = Activation::IDENTITY, int n_iter_no_change = 100);
	//NeuralNetwork nn(std::nullopt,std::nullopt,std::nullopt,std::nullopt,std::nullopt,std::nullopt);
	//NeuralNetwork args: sizes hidden_layer_sizes, Activation act, Solver sol, int maxit, double l_rate_init,double val_frac
	NeuralNetwork nn({},Activation::IDENTITY,{},500,0.1,{});
	std::cout << nn.n_iter_no_change << std::endl;
	std::cout << nn.hidden_layer_sizes.size() << std::endl;
	std::cout << nn.validation_fraction << std::endl;
	nn._initialize(expected(10), {3,4,2,2});
	//nn.print_network();

	training X = {{1,0,0},{1,0,1},{1,1,0},{1,1,1}};
	expected y = {{1,0},{1,0},{1,0},{0,1}};
	nn._fit(X,y);
	nn._predict("irisdataset");
	nn._plotLoss();
	return 0;
}
