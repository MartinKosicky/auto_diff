#include <iostream>
#include "FeedForwardNetwork.h"
#include <random>
#include <windows.h>

void TestFF()
{
	using namespace AD;
	std::random_device rd;
	typedef boost::numeric::ublas::matrix<float> matrix;

	std::vector<std::pair<matrix, matrix> > resultsMapping;
	// Generate some input output pairs

	resultsMapping.push_back(std::make_pair(
		Utils::CreateRowMatrix({ 1,0,0,0,0 }),   // input
		Utils::CreateRowMatrix({ 1,0,0 })        // expected output
		));

	resultsMapping.push_back(std::make_pair(
		Utils::CreateRowMatrix({ 0,1,0,0,0 }),   // input
		Utils::CreateRowMatrix({ 0,1,0 })        // expected output
		));

	resultsMapping.push_back(std::make_pair(
		Utils::CreateRowMatrix({ 0,0,1,0,0 }),   // input
		Utils::CreateRowMatrix({ 1,0,0 })        // expected output
		));

	resultsMapping.push_back(std::make_pair(
		Utils::CreateRowMatrix({ 0,0,0,1,0 }),   // input
		Utils::CreateRowMatrix({ 0,0,1 })        // expected output
		));

	resultsMapping.push_back(std::make_pair(
		Utils::CreateRowMatrix({ 0,0,0,0,1 }),   // input
		Utils::CreateRowMatrix({ 0,1,0 })        // expected output
		));


	FeedForwardNetwork nn({ 5,400,304,3 });

	DWORD start = GetTickCount();
	// Learning loop is this:
	for (size_t i = 0; i < 1000; i++)
	{
		for (auto&v : resultsMapping)
		{
			nn.SetInput(matrix(v.first));
			ResultType curOutput = nn.GetOutput();
			ResultType curError = nn.FeedExpectedValue(matrix(v.second), 1.f);
		}
	}
	// If i wanted to paralize, ie feed batches and these batches would be in seperate threads I would probably make the FeedForwardNetwork copies 1 for each thread
	// , gather the gradients...  average them, or sum or whatever and apply same Update on all threads
	// the expressions have some caching which means this is totally not thread safe

	DWORD end = GetTickCount();
	std::cout << "Duration: " << (float)(end - start) / 1000.f << std::endl;

	// validate results
	for (auto&v : resultsMapping)
	{
		nn.SetInput(matrix(v.first));
		ResultType curOutput = nn.GetOutput();

		std::cout << "current outout :" << curOutput << std::endl;
		ResultType curError = nn.FeedExpectedValue(matrix(v.second), 0.2f);
		std::cout << "current error :" << curError << std::endl << std::endl;
	}
}

void main()
{
	using namespace AD;
	
	// basic usage
	// suppose we want to maximize A = xy
	// we have the constraint      2x + y = 2400
	// this can be rewritten as    A = xy - error
	//                             2400 - 2x - y = 0
	//                             error = (2400 - 2x - y)^2
	//                             A = x*y - error*(some big constant, so that error will hurt the result)
	ExprWrapper x(1.f);  // random value
	ExprWrapper y(1200.f);  // random value
	ExprWrapper error = ExprWrapper(2400) - x * 2 - y;
	ExprWrapper A = x*y - error*error*1000;

	for (size_t i = 0; i < 20000; i++)
	{
		ResultType curOutput = A.Calc();
		std::cout << curOutput << std::endl;
		std::cout << "constraint error = " << error.Calc() << std::endl;
		ResultType dx = A.GetGradBy(x);
		ResultType dy = A.GetGradBy(y);

		x.Update(x.Calc() + dx*0.00002f);   // i am maximizing so +
		y.Update(y.Calc() + dy*0.00002f);
	}

	std::cout << "x = " << x.Calc() << std::endl;
	std::cout << "y = " << y.Calc() << std::endl;
	std::cout << "constraint error = " << error.Calc() << std::endl;

	// test a simple neural network created with this library
	TestFF();



}

