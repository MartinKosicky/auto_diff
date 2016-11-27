#pragma once

#include <boost/numeric/ublas/matrix.hpp>
#include <map>
#include <vector>
#include <memory>
#include <iostream>
#include <atomic>
#include <boost/container/flat_map.hpp>


class ResultType
{
public:
	typedef boost::numeric::ublas::matrix<float> matrix;

private:

public:

	float m_curValueFloat;
	matrix m_curValueMatrix;
	bool m_movedAway = false;
	enum class CurValueType { scalar, matrix };
	CurValueType m_curValueType = CurValueType::scalar;


	ResultType();
	ResultType(matrix&& m);
	ResultType(const matrix& m);
	ResultType(float v);
	ResultType(int v);
	ResultType(ResultType&& other);

	ResultType(const ResultType& other);
	void operator=(const ResultType& other);
	void operator=(ResultType&& other);

	~ResultType();

	ResultType Transpose() const;



	ResultType Product(const ResultType& other) const;

	ResultType operator*(const ResultType& other) const;

	// 1 to all
	template<typename T>
	ResultType ApplyUnaryFunc(const T& t) const
	{
		if (m_curValueType == CurValueType::scalar)
			return t(m_curValueFloat);

		//matrix m = m_curValueMatrix;
		matrix m = MatrixCache::Get().GetMatrix(m_curValueMatrix.size1(), m_curValueMatrix.size2());

		const size_t target = m.data().size();
		for (size_t i = 0; i < target;i++ )
		{
			m.data()[i] = t(m_curValueMatrix.data()[i]);
		}
		return m;
	}


	ResultType operator+(const ResultType& other) const;

	ResultType operator-(const ResultType& other) const;

	friend std::ostream & operator<<(std::ostream &os, const ResultType& p);

};



class Differentiable;
class ExprWrapper
{
	std::shared_ptr<Differentiable> m_holdedValue;

	template<typename T, typename ...Args>
	static std::shared_ptr<T> Make(Args&&... args)
	{
		return std::make_shared<T>(std::forward<Args>(args)...);
	}

public:
	ExprWrapper() {}
	ExprWrapper(ResultType v);



	ExprWrapper(std::shared_ptr<Differentiable> diff);
	ExprWrapper operator+(const ExprWrapper& other) const;
	ExprWrapper operator+(const ResultType& other) const;
	ExprWrapper operator-(const ExprWrapper& other) const;
	ExprWrapper operator-(const ResultType& other) const;
	ExprWrapper operator*(const ExprWrapper& other) const;
	ExprWrapper operator*(const ResultType& other) const;
	ExprWrapper MatrixMult(const ResultType& other) const;
	ExprWrapper MatrixMult(const ExprWrapper& other) const;
	ExprWrapper Sigmoid() const;
	ExprWrapper Pow(int powValue) const;
	ExprWrapper Softmax() const;
	ResultType Calc();

	ResultType GetGradBy(ExprWrapper& other);

	void Update(ResultType&& newValue);
};
