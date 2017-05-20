package main

import (
	"flag"
	"log"
	"math/rand"
	"time"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	var initStddev float64
	var numActions int
	var stepSize float64
	var maxSteps int

	flag.Float64Var(&initStddev, "stddev", 1, "standard deviation for initial params")
	flag.IntVar(&numActions, "actions", 10, "number of different actions")
	flag.Float64Var(&stepSize, "step", 0.01, "SGD step size")
	flag.IntVar(&maxSteps, "maxsteps", 0, "max SGD steps")

	flag.Parse()

	c := anyvec64.CurrentCreator()
	params := c.MakeVector(numActions)
	anyvec.Rand(params, anyvec.Normal, nil)
	params.Scale(c.MakeNumeric(initStddev))

	rewards := c.MakeVector(numActions)
	anyvec.Rand(rewards, anyvec.Uniform, nil)

	for step := 0; step < maxSteps || maxSteps == 0; step++ {
		paramVar := anydiff.NewVar(params)
		probs := anydiff.Exp(anydiff.LogSoftmax(paramVar, 0))
		reward := anydiff.Dot(probs, anydiff.NewConst(rewards))

		grad := anydiff.NewGrad(paramVar)
		reward.Propagate(anyvec.Ones(c, 1), grad)

		grad.Scale(c.MakeNumeric(stepSize))
		grad.AddToVars()

		log.Printf("step=%d reward=%f max=%f correct=%v", step, anyvec.Sum(reward.Output()),
			anyvec.Max(rewards),
			anyvec.MaxIndex(rewards) == anyvec.MaxIndex(probs.Output()))
	}
}
