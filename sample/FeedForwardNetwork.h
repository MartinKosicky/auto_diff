#include "../AD.h"
#include <random>
#include <boost/numeric/ublas/matrix.hpp>
#include <vector>

namespace AD {

	// simple feed forward network
	struct FeedForwardNetwork
	{
		std::random_device rd;

		std::vector<size_t> m_units;
		std::vector<ExprWrapper> m_weightExpressions;
		ExprWrapper m_inputExpression;
		ExprWrapper m_outputExpression;
		ExprWrapper m_errorExpression;
		ExprWrapper m_expectedOutput;

		typedef boost::numeric::ublas::matrix<float> matrix;

		/**
		* Create a network with unit specified in units.  If I want a neural network with 3 inputs 2 hidden layers with 5 neurons and 6 neurons and 7 outputs the input is
		* {3,5,6,7}.  All units are sigmoids, the last is softmax
		* The inputs and values in hidden layers are stored in row matrices (1,N)
		*/
		FeedForwardNetwork(std::vector<size_t>&& units);

		/**
		* Updates the input data  (when we pass a different input)
		*/
		void SetInput(matrix&& newInput);

		/**
		* Calculates output based on the input set by SetInput
		*/
		ResultType GetOutput();

		/**
		*  Backpropagate error. If you used GetOutput before the sub expression is cached and only the error will be calculated.
		*  @return the last error based on expectedOutput and what was calculated. Weights will be adjusted by learningRate
		*/
		ResultType FeedExpectedValue(matrix&& expectedOutput, float learningRate);
	};

}