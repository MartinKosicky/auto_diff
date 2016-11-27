#include "FeedForwardNetwork.h"

namespace AD {

	FeedForwardNetwork::FeedForwardNetwork(std::vector<size_t>&& units) :
		m_units(units)
	{
		matrix inputMatrix(1, m_units[0]);
		// Creating expression from constant will make an updatable ExprWrapper
		m_inputExpression = ExprWrapper(inputMatrix);
		m_expectedOutput = ExprWrapper(matrix(1, m_units.back()));

		for (size_t i = 1; i < m_units.size(); i++)
		{
			matrix weightMatrix(m_units[i - 1], m_units[i]);
			for (auto& val : weightMatrix.data())
			{
				val = (float)(rd() % 1000) / 1000.f;
			}
			m_weightExpressions.push_back(ExprWrapper(weightMatrix));
		}

		ExprWrapper curUnit = m_inputExpression;
		for (size_t i = 0; i < m_weightExpressions.size(); i++)
		{
			if (i == m_weightExpressions.size() - 1)
				curUnit = curUnit.MatrixMult(m_weightExpressions[i]).Softmax();
			else
				curUnit = curUnit.MatrixMult(m_weightExpressions[i]).Sigmoid();
		}

		m_outputExpression = curUnit;
		// Creating expression based on operators will create expression tree
		m_errorExpression = ExprWrapper(0.5f) * (m_expectedOutput - m_outputExpression).Pow(2);
	}

	void FeedForwardNetwork::SetInput(matrix&& newInput)
	{
		// This will trash cache in the call tree only where required
		m_inputExpression.Update(std::move(newInput));
	}

	ResultType FeedForwardNetwork::GetOutput()
	{
		return m_outputExpression.Calc();
	}

	ResultType FeedForwardNetwork::FeedExpectedValue(matrix&& expectedOutput, float learningRate)
	{
		m_expectedOutput.Update(std::move(expectedOutput));
		std::vector<ResultType> m_weightUpdates;
		ResultType curError = m_errorExpression.Calc();
		for (size_t i = 0; i < m_weightExpressions.size(); i++)
		{
			// This is all that is required to get gradient updates for gradient descent
			ResultType curGrad = m_errorExpression.GetGradBy(m_weightExpressions[i]) * learningRate;
			ResultType newValue = (m_weightExpressions[i].Calc() - curGrad);
			m_weightUpdates.push_back(std::move(newValue));
		}

		for (size_t i = 0; i < m_weightExpressions.size(); i++)
		{
			m_weightExpressions[i].Update(std::move(m_weightUpdates[i]));
		}

		return curError;
	}

}