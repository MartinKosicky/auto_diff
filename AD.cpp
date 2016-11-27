#include "AD.h"
#include <boost/numeric/ublas/io.hpp>
class IExpression
{
	static std::atomic<size_t> m_counter;
	size_t m_id;
	ResultType m_calcCached;
	bool m_isCalcCached = false;
protected:
	virtual ResultType DoCalc() = 0;
	virtual bool TrashCalcCache();

public:
	virtual ~IExpression() {}

	IExpression();
	virtual const ResultType& Calc();
	virtual ResultType GetGradBy(size_t id) = 0;
	virtual size_t GetId() const;

};

struct MatrixCache
{
private:
	thread_local static MatrixCache m_matrixCacheSingleton;

	typedef boost::numeric::ublas::matrix<float> matrix;

	boost::container::flat_map<std::tuple<size_t, size_t>,
		std::vector<matrix> > m_matrixCache;
public:

	static MatrixCache& Get()
	{
		return m_matrixCacheSingleton;
	}

	matrix GetMatrix(const matrix& other)
	{
		matrix m = GetMatrix(other.size1(), other.size2());
		const size_t maxSize = other.data().size();
		for (size_t i = 0; i < maxSize; i++)
		{
			m.data()[i] = other.data()[i];
		}
		return m;
	}

	matrix GetMatrix(size_t s1, size_t s2)
	{
		auto& c = m_matrixCache[std::make_tuple(s1, s2)];
		if (c.empty())
		{
			return matrix(s1, s2);
		}
		else
		{
			matrix x = std::move(c.back());
			c.pop_back();
			return std::move(x);
		}
	}

	void ReturnMatrix(matrix&& m)
	{
		auto& mc = m_matrixCache[std::make_tuple(m.size1(), m.size2())];
		if (mc.size() < 40)
			mc.push_back(std::move(m));

	}
};



class Differentiable :
	public IExpression
{
	std::vector<std::shared_ptr<Differentiable> > m_children;
	std::vector<Differentiable*> m_parents;

	std::map<size_t, ResultType> m_differentiationsCached;

protected:
	virtual ResultType CombineDiffChain(size_t childId, size_t targetId);

	void TrashCacheForParents();
	virtual ResultType GetGradByLocal(size_t id) = 0;

	ResultType DoGetGradByParent(size_t id);

public:
	Differentiable(std::vector<std::shared_ptr<Differentiable> >&& children);
	ResultType GetGradBy(size_t id);
};

class ConstExpression :
	public Differentiable
{
	ResultType m_value;

protected:
	virtual ResultType GetGradByLocal(size_t id) override;

	// Inherited via IExpression
	virtual ResultType DoCalc() override;

public:

	bool TrashCalcCache()
	{
		return true;
	}

	ConstExpression(ResultType&& v);
	void UpdateValue(ResultType&& v);
	virtual const ResultType& Calc() override;
};

class MultExpression :
	public Differentiable
{
	std::shared_ptr<Differentiable> m_left;
	std::shared_ptr<Differentiable> m_right;
public:

	MultExpression(std::shared_ptr<Differentiable> left, std::shared_ptr<Differentiable> right);

	ResultType DoCalc() override;

protected:
	ResultType GetGradByLocal(size_t id);
};

class MatrixMultExpression :
	public Differentiable
{
	std::shared_ptr<Differentiable> m_left;
	std::shared_ptr<Differentiable> m_right;

protected:

	virtual ResultType CombineDiffChain(size_t localId, size_t targetId);

public:

	MatrixMultExpression(std::shared_ptr<Differentiable> left, std::shared_ptr<Differentiable> right);
	ResultType DoCalc() override;

protected:
	ResultType GetGradByLocal(size_t id);
};

class SumExpression :
	public Differentiable
{
	std::shared_ptr<Differentiable> m_left;
	std::shared_ptr<Differentiable> m_right;
public:

	SumExpression(std::shared_ptr<Differentiable> left, std::shared_ptr<Differentiable> right);
	ResultType DoCalc() override;

protected:
	ResultType GetGradByLocal(size_t id);
};

class MinusExpression :
	public Differentiable
{
	std::shared_ptr<Differentiable> m_left;
	std::shared_ptr<Differentiable> m_right;
public:
	MinusExpression(std::shared_ptr<Differentiable> left, std::shared_ptr<Differentiable> right);
	ResultType DoCalc() override;

protected:
	ResultType GetGradByLocal(size_t id);
};

class PowerExpression :
	public Differentiable
{
	std::shared_ptr<Differentiable> m_val;
	int m_powValue;
public:

	PowerExpression(std::shared_ptr<Differentiable> val, int powValue);
	ResultType GetPow(const ResultType& v, int p);
	ResultType DoCalc() override;

protected:
	ResultType GetGradByLocal(size_t id);
};

class SigmoidExpression :
	public Differentiable
{
	std::shared_ptr<Differentiable> m_val;
public:
	SigmoidExpression(std::shared_ptr<Differentiable> val);
	ResultType GetGradByLocal(size_t id);
	ResultType DoCalc() override;

};

class SoftmaxExpression :
	public Differentiable
{
	std::shared_ptr<Differentiable> m_val;
	ResultType m_prevGrad;
	ResultType m_prevJacobian;
public:
	SoftmaxExpression(std::shared_ptr<Differentiable> val):
		Differentiable({ val }), m_val(val)
	{

	}

	ResultType CombineDiffChain(size_t localId, size_t targetId) override
	{
		assert(m_val->GetId() == localId);
		ResultType thisResults = DoGetGradByParent(targetId);  //  xW ->  Y  ... this is Y

		assert(thisResults.m_curValueType == ResultType::CurValueType::matrix);
		return thisResults.Product(m_prevJacobian);
	}


	ResultType GetGradByLocal(size_t id)
	{
		return ResultType();
	}

	ResultType DoCalc() override
	{
		const ResultType& rs = m_val->Calc();
		assert(rs.m_curValueType == ResultType::CurValueType::matrix);
		assert(rs.m_curValueMatrix.size1() == 1);

		float sum = 0;
		std::vector<float> tempCalcs;  tempCalcs.reserve(rs.m_curValueMatrix.size2());
		boost::numeric::ublas::matrix<float> resultMatrix = MatrixCache::Get().GetMatrix(1, rs.m_curValueMatrix.size2());

		for (auto& v : rs.m_curValueMatrix.data())
		{
			float foo = std::exp(v);
			tempCalcs.push_back(foo);
			sum += foo;
		}

		for (size_t i = 0; i < tempCalcs.size(); i++)
		{
			resultMatrix(0, i) = tempCalcs[i] / sum;
		}

		if (m_prevJacobian.m_curValueMatrix.size1() * m_prevJacobian.m_curValueMatrix.size2() == 0)
			m_prevJacobian = MatrixCache::Get().GetMatrix(resultMatrix.size2(), resultMatrix.size2());


		for (size_t i = 0; i < resultMatrix.size2(); i++)
		{
			for (size_t j = 0; j < resultMatrix.size2(); j++)
			{
				float adder = 0;
				if (i == j)					
					adder = (resultMatrix(0,i))*(1.f - resultMatrix(0, i));
				else
					adder = (-resultMatrix(0, i) * resultMatrix(0,j));

				m_prevJacobian.m_curValueMatrix(i, j) = adder;
			}
		}
		return resultMatrix;
	}

};

thread_local MatrixCache MatrixCache::m_matrixCacheSingleton;

ResultType::ResultType() {
	m_curValueFloat = (float)0;
	m_curValueType = CurValueType::scalar;
}

ResultType::ResultType(ResultType&& other)
{
	operator=(std::move(other));
}

ResultType::ResultType(const ResultType& other)
{
	operator=(other);
}

void ResultType::operator=(const ResultType& other)
{
	m_curValueFloat = other.m_curValueFloat;
	m_curValueMatrix = MatrixCache::Get().GetMatrix(other.m_curValueMatrix);
	m_curValueType = other.m_curValueType;
}

void ResultType::operator=(ResultType&& other)
{
	m_curValueFloat = other.m_curValueFloat;
	m_curValueMatrix = std::move(other.m_curValueMatrix);
	m_curValueType = other.m_curValueType;
	other.m_movedAway = true;
}

ResultType::ResultType(matrix && m) {
	m_curValueMatrix = m;
	m_curValueType = CurValueType::matrix;
}

ResultType::ResultType(const matrix & m) {
	m_curValueMatrix = MatrixCache::Get().GetMatrix(m);
	m_curValueType = CurValueType::matrix;
}

ResultType::~ResultType()
{
	if (m_curValueType == CurValueType::matrix && !m_movedAway)
	{
		MatrixCache::Get().ReturnMatrix(std::move(m_curValueMatrix));
	}
}


ResultType ResultType::Transpose() const
{
	assert(this->m_curValueType == CurValueType::matrix);
	matrix newMatrix = MatrixCache::Get().GetMatrix(this->m_curValueMatrix.size2(), this->m_curValueMatrix.size1());

	const size_t blocksize = 16;
	const size_t n = this->m_curValueMatrix.size1();
	const size_t m = this->m_curValueMatrix.size2();

	for (int i = 0; i < n; i += blocksize) {
		for (int j = 0; j < m; j += blocksize) {
			// transpose the block beginning at [i,j]

			const size_t maxK = std::min(i + blocksize, n);
			const size_t maxL = std::min(j + blocksize, m);
			for (int k = i; k < maxK; ++k) {
				for (int l = j; l < maxL; ++l) {
					newMatrix(l, k) = this->m_curValueMatrix(k, l);
				}
			}
		}
	}
		
	return newMatrix;
}


ResultType::ResultType(float v)
{
	m_curValueFloat = v;
	m_curValueType = CurValueType::scalar;
}



ResultType::ResultType(int v)
{
	m_curValueFloat = (float)v;
	m_curValueType = CurValueType::scalar;
}

ResultType ResultType::Product(const ResultType & other) const
{
	assert(m_curValueType == CurValueType::matrix && other.m_curValueType == CurValueType::matrix);
	
	const int newHeight = (int)m_curValueMatrix.size1();
	const int newWidth = (int)other.m_curValueMatrix.size2();
	const int sz = (int)m_curValueMatrix.size2();
	assert(sz == (int)other.m_curValueMatrix.size1());

	typedef boost::numeric::ublas::matrix<float, boost::numeric::ublas::column_major> col_matrix;
	typedef boost::numeric::ublas::matrix<float> matrix;

	matrix newMatrix = MatrixCache::Get().GetMatrix(m_curValueMatrix.size1(), other.m_curValueMatrix.size2());

	const size_t T = 16;
	const size_t N = m_curValueMatrix.size1();
	const size_t M = m_curValueMatrix.size2();
	const size_t P = other.m_curValueMatrix.size2();

	for (size_t I = 0; I < N; I+=T)
	{
		for (size_t J = 0; J < P; J+=T)
		{
			for (size_t K = 0; K < (size_t)M; K+=T)
			{
				const size_t iDest = std::min(I + T, N);
				const size_t jDest = std::min(J + T, P);
				const size_t kDest = std::min(K + T, M);

				for (size_t i = I; i < iDest; i++)
				{
					for (size_t j = J; j < jDest; j++)
					{
						float sum = 0;
						for (size_t k = K; k < kDest; k++)
						{
							sum += m_curValueMatrix(i, k)*
								other.m_curValueMatrix(k, j);
						}
						newMatrix(i, j) = sum;
					}
				}		
			}
		}
	}

	return newMatrix;
}

ResultType ResultType::operator*(const ResultType& other) const
{
	if (m_curValueType == CurValueType::scalar && other.m_curValueType == CurValueType::scalar)
	{
		return m_curValueFloat * other.m_curValueFloat;
	}
	else if (m_curValueType == CurValueType::scalar && other.m_curValueType == CurValueType::matrix)
	{
		float f = m_curValueFloat;
		return other.ApplyUnaryFunc([f](float v)
		{
			return v*f;
		});
	}
	else if (other.m_curValueType == CurValueType::scalar)
	{
		float f = other.m_curValueFloat;
		return ApplyUnaryFunc([f](float v)
		{
			return v*f;
		});
	}
	assert(m_curValueMatrix.size1() == other.m_curValueMatrix.size1() && m_curValueMatrix.size2() == other.m_curValueMatrix.size2());
	matrix result = MatrixCache::Get().GetMatrix(m_curValueMatrix);

	auto it1 = result.data().begin();
	auto it2 = other.m_curValueMatrix.data().begin();
	for (; it1 != result.data().end(); it1++, it2++)
	{
		(*it1) *= (*it2);
	}
	return ResultType(std::move(result));
}

ResultType ResultType::operator+(const ResultType& other) const
{
	if (m_curValueType == CurValueType::scalar && other.m_curValueType == CurValueType::scalar)
	{
		return m_curValueFloat + other.m_curValueFloat;
	}
	else if (m_curValueType == CurValueType::scalar && other.m_curValueType == CurValueType::matrix)
	{
		float f = m_curValueFloat;
		return other.ApplyUnaryFunc([f](float v)
		{
			return v + f;
		});
	}
	else if (other.m_curValueType == CurValueType::scalar)
	{
		float f = other.m_curValueFloat;
		return ApplyUnaryFunc([f](float v)
		{
			return v + f;
		});
	}
	matrix result = MatrixCache::Get().GetMatrix(m_curValueMatrix);
	const size_t maxSize = m_curValueMatrix.data().size();

	for (size_t i = 0; i < maxSize; i++)
	{
		result.data()[i] += other.m_curValueMatrix.data()[i];
	}
	return ResultType(std::move(result));
}

ResultType ResultType::operator-(const ResultType& other) const
{
	if (m_curValueType == CurValueType::scalar && other.m_curValueType == CurValueType::scalar)
	{
		return m_curValueFloat - other.m_curValueFloat;
	}
	else if (m_curValueType == CurValueType::scalar && other.m_curValueType == CurValueType::matrix)
	{
		// f - v = -v + f
		float f = m_curValueFloat;
		return other.ApplyUnaryFunc([f](float v)
		{
			return -v + f;
		});
	}
	else if (other.m_curValueType == CurValueType::scalar)
	{
		// f - v = -v + f
		float f = other.m_curValueFloat;
		return ApplyUnaryFunc([f](float v)
		{
			return v-f;
		});
	}
	matrix result = MatrixCache::Get().GetMatrix(m_curValueMatrix);
	const size_t maxSize = m_curValueMatrix.data().size();

	for (size_t i = 0; i < maxSize; i++)
	{
		result.data()[i] -= other.m_curValueMatrix.data()[i];
	}
	return ResultType(std::move(result));
}

std::ostream & operator<<(std::ostream & os, const ResultType & p)
{
	if (p.m_curValueType == ResultType::CurValueType::scalar)
		os << p.m_curValueFloat;
	else
		os << p.m_curValueMatrix;
	return os;
}

bool IExpression::TrashCalcCache()
{
	bool prevValue = m_isCalcCached;
	m_isCalcCached = false;
	return prevValue;
}

IExpression::IExpression()
{
	m_id = m_counter.fetch_add(1);
}

const ResultType& IExpression::Calc() {
	if (!m_isCalcCached)
	{
		m_calcCached = DoCalc();
		m_isCalcCached = true;
	}
	return m_calcCached;
};

size_t IExpression::GetId() const
{
	return m_id;
}

std::atomic<size_t> IExpression::m_counter(0);

ResultType Differentiable::CombineDiffChain(size_t childId, size_t targetId)
{
	ResultType curValue = GetGradByLocal(childId);  // dParent/dSelf
	if (GetId() != targetId)
		return curValue * DoGetGradByParent(targetId);
	else
		return curValue;
}

void Differentiable::TrashCacheForParents()
{
	if (!TrashCalcCache())
		return;   // no need to trash parents

	
	m_differentiationsCached.clear();


	for (auto& parent : m_parents)
	{
		parent->TrashCacheForParents();
	}
}

ResultType Differentiable::DoGetGradByParent(size_t id)
{
	if (GetId() == id)
		return 0;

	auto it = m_differentiationsCached.find(id);
	if (it != m_differentiationsCached.end())
		return it->second;

	ResultType totalSum;

	for (auto parent : m_parents)
	{
		totalSum = totalSum + parent->CombineDiffChain(GetId(), id);
	}

	m_differentiationsCached[id] = totalSum;
	return totalSum;
}

Differentiable::Differentiable(std::vector<std::shared_ptr<Differentiable> >&& children) :
	m_children(std::move(children))
{
	for (auto ch : m_children)
	{
		ch->m_parents.push_back(this);
	}
}

ResultType Differentiable::GetGradBy(size_t id)
{
	return DoGetGradByParent(id);
}

ResultType ConstExpression::GetGradByLocal(size_t id)
{
	return 0;
}

ResultType ConstExpression::DoCalc() 
{
	return m_value;
}

ConstExpression::ConstExpression(ResultType&& v) :
	Differentiable(std::initializer_list<std::shared_ptr<Differentiable> >{}),
	m_value(std::move(v))
{}

void ConstExpression::UpdateValue(ResultType&& v)
{
	m_value = std::move(v);
	TrashCacheForParents();
}

const ResultType& ConstExpression::Calc() 
{
	return m_value;
}

MultExpression::MultExpression(std::shared_ptr<Differentiable> left, std::shared_ptr<Differentiable> right) :
	Differentiable(std::initializer_list<std::shared_ptr<Differentiable> >{left, right}),
	m_left(left), m_right(right)
{}

ResultType MultExpression::DoCalc()
{
	return m_left->Calc() * m_right->Calc();
}

ResultType MultExpression::GetGradByLocal(size_t id)
{
	if (m_left->GetId() == id)
	{
		return m_right->Calc();
	}
	else if (m_right->GetId() == id)
	{
		return m_left->Calc();
	}
	throw std::runtime_error("Unknown id");
}

ResultType MatrixMultExpression::CombineDiffChain(size_t localId, size_t targetId)
{
	ResultType thisResults = DoGetGradByParent(targetId);  //  xW ->  Y  ... this is Y
	assert(thisResults.m_curValueType == ResultType::CurValueType::matrix && thisResults.m_curValueMatrix.size1() == 1);

	if (m_left->GetId() == localId)
	{
		namespace ublas = boost::numeric::ublas;
		const auto& rightCalc = m_right->Calc();
		assert(rightCalc.m_curValueType == ResultType::CurValueType::matrix);
	
		return thisResults.Product(rightCalc.Transpose());
		//return ublas::prod(thisResults.m_curValueMatrix, ublas::trans(rightCalc.m_curValueMatrix));
	}
	else if (m_right->GetId() == localId)
	{
		namespace ublas = boost::numeric::ublas;
		const auto& leftCalc = m_left->Calc();
		assert(leftCalc.m_curValueType == ResultType::CurValueType::matrix);

		ResultType temp = leftCalc.Transpose();
		return temp.Product(thisResults);
	}
	else
		throw std::runtime_error("WTF");
}

MatrixMultExpression::MatrixMultExpression(std::shared_ptr<Differentiable> left, std::shared_ptr<Differentiable> right) :
	Differentiable(std::initializer_list<std::shared_ptr<Differentiable> >{left, right}),
	m_left(left), m_right(right)
{}

ResultType MatrixMultExpression::DoCalc()
{
	return m_left->Calc().Product(m_right->Calc());
}

ResultType MatrixMultExpression::GetGradByLocal(size_t id)
{
	throw std::runtime_error("Unsupported");
}

SumExpression::SumExpression(std::shared_ptr<Differentiable> left, std::shared_ptr<Differentiable> right) :
	Differentiable(std::initializer_list<std::shared_ptr<Differentiable> >{left, right}),
	m_left(left), m_right(right)
{}

ResultType SumExpression::DoCalc() 
{
	return m_left->Calc() + m_right->Calc();
}

ResultType SumExpression::GetGradByLocal(size_t id)
{
	if (m_left->GetId() == id)
	{
		return 1;
	}
	else if (m_right->GetId() == id)
	{
		return 1;
	}
	throw std::runtime_error("Unknown id");
}

MinusExpression::MinusExpression(std::shared_ptr<Differentiable> left, std::shared_ptr<Differentiable> right) :
	Differentiable(std::initializer_list<std::shared_ptr<Differentiable> >{left, right}),
	m_left(left), m_right(right)
{}

ResultType MinusExpression::DoCalc() 
{
	return m_left->Calc() - m_right->Calc();
}

ResultType MinusExpression::GetGradByLocal(size_t id)
{
	if (m_left->GetId() == id)
	{
		return 1;
	}
	else if (m_right->GetId() == id)
	{
		return -1;
	}
	throw std::runtime_error("Unknown id");
}

PowerExpression::PowerExpression(std::shared_ptr<Differentiable> val, int powValue) :
	Differentiable(std::initializer_list<std::shared_ptr<Differentiable> >{val}),
	m_val(val), m_powValue(powValue)
{}

ResultType PowerExpression::GetPow(const ResultType& v, int p)
{
	ResultType result = 1;
	for (int i = 0; i < p; i++)
	{
		result = result * v;
	}
	return result;
}

ResultType PowerExpression::DoCalc()
{
	const ResultType& v = m_val->Calc();
	return GetPow(v, m_powValue);
}

ResultType PowerExpression::GetGradByLocal(size_t id)
{
	if (m_val->GetId() == id)
	{
		const ResultType& v = m_val->Calc();
		return GetPow(v, m_powValue - 1)*m_powValue;
	}
	throw std::runtime_error("Unknown id");
}

SigmoidExpression::SigmoidExpression(std::shared_ptr<Differentiable> val):
	Differentiable({val}),m_val(val)
{

}

ResultType SigmoidExpression::GetGradByLocal(size_t id)
{
	assert(m_val->GetId() == id);
	return Calc().ApplyUnaryFunc([](float f)
	{
		return f*(1 - f);
	});
}

ResultType SigmoidExpression::DoCalc()
{
	return m_val->Calc().ApplyUnaryFunc([](float f)
	{
		return 1.f / (1 + std::exp(-f));
	});
}

ExprWrapper::ExprWrapper(ResultType v)
{
	m_holdedValue = std::make_shared<ConstExpression>(std::move(v));
}

ExprWrapper::ExprWrapper(std::shared_ptr<Differentiable> diff)
{
	m_holdedValue = diff;
}

ExprWrapper ExprWrapper::operator+(const ExprWrapper& other) const
{
	return ExprWrapper(Make<SumExpression>(m_holdedValue, other.m_holdedValue));
}

ExprWrapper ExprWrapper::operator+(const ResultType& other) const
{
	return ExprWrapper(Make<SumExpression>(m_holdedValue, ExprWrapper(other).m_holdedValue));
}

ExprWrapper ExprWrapper::operator-(const ExprWrapper& other) const
{
	return ExprWrapper::ExprWrapper(Make<MinusExpression>(m_holdedValue, other.m_holdedValue));
}

ExprWrapper ExprWrapper::operator-(const ResultType& other) const
{
	return ExprWrapper(Make<MinusExpression>(m_holdedValue, ExprWrapper(other).m_holdedValue));
}

ExprWrapper ExprWrapper::operator*(const ExprWrapper& other) const
{
	return ExprWrapper(Make<MultExpression>(m_holdedValue, other.m_holdedValue));
}

ExprWrapper ExprWrapper::operator*(const ResultType& other) const
{
	return ExprWrapper(Make<MultExpression>(m_holdedValue, ExprWrapper(other).m_holdedValue));
}

ExprWrapper ExprWrapper::MatrixMult(const ResultType& other) const
{
	return ExprWrapper(Make<MatrixMultExpression>(m_holdedValue, ExprWrapper(other).m_holdedValue));
}

ExprWrapper ExprWrapper::MatrixMult(const ExprWrapper& other) const
{
	return ExprWrapper(Make<MatrixMultExpression>(m_holdedValue, other.m_holdedValue));
}

ExprWrapper ExprWrapper::Pow(int powValue) const
{
	return ExprWrapper(Make<PowerExpression>(m_holdedValue, powValue));
}

ResultType ExprWrapper::Calc()
{
	return m_holdedValue->Calc();
}

ResultType ExprWrapper::GetGradBy(ExprWrapper& other)
{
	return other.m_holdedValue->GetGradBy(m_holdedValue->GetId());
}

ExprWrapper ExprWrapper::Sigmoid() const
{
	return ExprWrapper(Make<SigmoidExpression>(m_holdedValue));
}

ExprWrapper ExprWrapper::Softmax() const
{
	return ExprWrapper(Make<SoftmaxExpression>(m_holdedValue));
}

void ExprWrapper::Update(ResultType&& newValue)
{
	auto val = std::dynamic_pointer_cast<ConstExpression>(m_holdedValue);
	assert(val.get() != nullptr);

	val->UpdateValue(std::move(newValue));
}


