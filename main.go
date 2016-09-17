package main

import (
	"errors"
	"fmt"
	"math/rand"
)

var (
	errVectorsNotEqual = errors.New("Vectors are not equal")
)

// Scalar -
type Scalar float64

// Vector -
type Vector []Scalar

// InnerProduct -
func (v Vector) InnerProduct(m Vector) (p Scalar, err error) {
	if len(v) != len(m) {
		return p, errVectorsNotEqual
	}
	for i := range v {
		p += v[i] * m[i]
	}
	return p, nil
}

// Scale -
func (v Vector) Scale(s Scalar) (r Vector) {
	r = make(Vector, len(v))
	for i := range v {
		r[i] = v[i] * s
	}
	return r
}

// Add -
func (v Vector) Add(m Vector) (r Vector, err error) {
	if len(v) != len(m) {
		return nil, errVectorsNotEqual
	}
	r = make(Vector, len(v))
	for i, s := range v {
		r[i] = s + m[i]
	}
	return r, nil
}

// Matrix -
type Matrix []Vector

func main() {
	// our ds
	ds := make(Matrix, 10)
	ds[0] = Vector{3, 2, 1}
	ds[1] = Vector{4.5, 3, 1}
	ds[2] = Vector{3, 3.5, 1}
	ds[3] = Vector{5, 4.5, 1}
	ds[4] = Vector{6, 4, 1}
	ds[5] = Vector{2.5, 5, 1}
	ds[6] = Vector{3, 6, 1}
	ds[7] = Vector{2, 6.5, 1}
	ds[8] = Vector{3, 7, 1}
	ds[9] = Vector{3.5, 7.5, 1}

	// weights
	w := Vector{0, 0, 0}

	// learning rate
	lr := Scalar(1)

	// labels
	ls := Vector{-1, -1, -1, -1, -1, 1, 1, 1, 1, 1}

	// rounds
	for r := 0; r < 10; r++ {
		// shuffle order
		or := rand.Perm(10)
		for _, i := range or {
			// prediction
			p, _ := ds[i].InnerProduct(w)
			fmt.Printf("> Predicted; p=%f; l=%f; w=%v;\n", p, ls[i], w)

			// weight adjustment
			if p*ls[i] <= 0 {
				w, _ = w.Add(ds[i].Scale(lr).Scale(ls[i]))
				fmt.Printf("> > Adjusting weight; w=%v;\n", w)
			}
		}
	}
}
