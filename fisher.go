package main

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyfwd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/num-analysis/linalg/ludecomp"
)

// Fisher computes the Fisher information matrix for a
// softmax distribution.
func Fisher(params anyvec.Vector) *anyvec.Matrix {
	creator := &anyfwd.Creator{
		ValueCreator: params.Creator(),
		GradSize:     params.Len(),
	}
	inVec := creator.MakeVector(params.Len()).(*anyfwd.Vector)
	inVec.Values.Set(params)
	constIn := anydiff.NewConst(inVec.Copy())
	for i, g := range inVec.Jacobian {
		g.Slice(i, i+1).AddScalar(params.Creator().MakeNumeric(1))
	}
	varIn := anydiff.NewVar(inVec)

	logProbs1 := anydiff.LogSoftmax(constIn, 0)
	logProbs2 := anydiff.LogSoftmax(varIn, 0)
	kl := anydiff.Dot(anydiff.Exp(logProbs1), anydiff.Sub(logProbs1, logProbs2))

	grad := anydiff.NewGrad(varIn)
	kl.Propagate(anyvec.Ones(creator, 1), grad)

	rows := grad[varIn].(*anyfwd.Vector).Jacobian
	return &anyvec.Matrix{
		Data: params.Creator().Concat(rows...),
		Rows: len(rows),
		Cols: len(rows),
	}
}

// NaturalGradient solves for the natural gradient.
//
// This may modify m.
func NaturalGradient(m *anyvec.Matrix, grad anyvec.Vector) anyvec.Vector {
	c := m.Data.Creator()

	// The fisher matrix has a null-space with the vector
	// of all ones.
	// By adding all ones to the matrix, we remove this
	// null-space without changing the behavior of the
	// matrix for the gradient (which is orthogonal to
	// the vector of all ones).
	m.Data.AddScalar(c.MakeNumeric(1))

	lu := ludecomp.Decompose(&linalg.Matrix{
		Rows: m.Rows,
		Cols: m.Cols,
		Data: linalgVector(m.Data),
	})

	// Once we get close to the solution, the null-space
	// will expand beyond just the vector of all ones.
	// Specifically, any vector besides the maximum one
	// will have virtually no effect on the output.
	if lu.PivotScale() < 1e-12 {
		return grad
	}

	solution := lu.Solve(linalgVector(grad))
	return c.MakeVectorData(c.MakeNumericList(solution))
}

func linalgVector(v anyvec.Vector) linalg.Vector {
	switch data := v.Data().(type) {
	case []float64:
		return data
	default:
		panic("unsupported numeric type")
	}
}
