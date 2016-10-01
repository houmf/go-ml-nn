package main

import (
	"errors"
	"fmt"
	"math/rand"
	"time"
)

var (
	errVectorsNotEqual  = errors.New("Vectors are not equal")
	errMatricesNotEqual = errors.New("Matrices are not equal")
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

// Row -
func (m Matrix) Row(i int) (Vector, error) {
	return m[i], nil
}

// Col -
func (m Matrix) Col(i int) (Vector, error) {
	v := Vector{}
	for j := range m {
		v = append(v, m[j][i])
	}
	return v, nil
}

// Zeroes -
func Zeroes(m, n int) Matrix {
	r := make(Matrix, m)
	for i := 0; i < m; i++ {
		r[i] = make(Vector, n)
		// for j := 0; j < n; j++ {
		// 	r[i][j] = 0
		// }
	}
	return r
}

// RandomMatrix -
func RandomMatrix(m, n int, sc float64) Matrix {
	sr := rand.New(rand.NewSource(time.Now().Unix()))
	r := make(Matrix, m)
	for i := 0; i < m; i++ {
		r[i] = make(Vector, n)
		for j := 0; j < n; j++ {
			r[i][j] = Scalar((sr.Float64() - 0.5) * sc)
		}
	}
	return r
}

// Multiply -
func (m Matrix) Multiply(m2 Matrix) (Matrix, error) {
	// TODO(geoah) Handle error
	r := Zeroes(len(m), len(m2[0]))
	for i := range m {
		for j := range m2[0] {
			col, _ := m2.Col(j)
			row, _ := m.Row(i)
			ip, _ := row.InnerProduct(col)
			// fmt.Println(i, j, row, col, ip)
			r[i][j] = ip
		}
	}
	return r, nil
}

// Add -
func (m Matrix) Add(m2 Matrix) Matrix {
	// TODO(geoah) Handle error
	r := Zeroes(len(m), len(m[0]))
	for i := range m {
		for j := range m[0] {
			r[i][j] = m[i][j] + m2[i][j]
		}
	}
	return r
}

// Scale -
func (m Matrix) Scale(s Scalar) Matrix {
	// TODO(geoah) Handle error
	r := Zeroes(len(m), len(m[0]))
	for i := range m {
		for j := range m[0] {
			r[i][j] = m[i][j] * s
		}
	}
	return r
}

// FrobeniusProduct -
func (m Matrix) FrobeniusProduct(m2 Matrix) (Matrix, error) {
	// TODO(geoah) Handle error
	r := Zeroes(len(m), len(m[0]))
	for i := range m {
		for j := range m[i] {
			r[i][j] = m[i][j] * m2[i][j]
		}
	}
	return r, nil
}

// Transpose -
func (m Matrix) Transpose() Matrix {
	m2 := Zeroes(len(m[0]), len(m))
	for i := range m[0] {
		for j := range m {
			m2[i][j] = m[j][i]
		}
	}
	return m2
}

// MultiplyWithVector -
func (m Matrix) MultiplyWithVector(v Vector) (Vector, error) {
	r := Vector{}
	for i := range m {
		iv, _ := m[i].InnerProduct(v) // TODO(geoah) Handle error
		r = append(r, iv)
	}
	return r, nil
}

// func main() {
// 	m1 := make(Matrix, 3)
// 	m1[0] = Vector{1, 2}
// 	m1[1] = Vector{3, 4}
// 	m1[2] = Vector{5, 6}

// 	m2 := make(Matrix, 2)
// 	m2[0] = Vector{2, 3}
// 	m2[1] = Vector{4, 5}

// 	mm, _ := m1.Multiply(m2)
// 	fmt.Printf("%+v", mm)
// }

// Derivative -
type Derivative func(Scalar) Scalar

// ReluDerivative -
func ReluDerivative(s Scalar) Scalar {
	if s > 0 {
		return 1
	}
	return 0
}

// Activator -
type Activator func(Scalar) Scalar

// Activate -
func Activate(m Matrix, fn Activator) Matrix {
	r := Zeroes(len(m), len(m[0]))
	for i := range m {
		for j := range m[0] {
			r[i][j] = fn(m[i][j])
		}
	}
	return r
}

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
	// w := Vector{0, 0, 0}

	// learning rate
	// lr := Scalar(1)

	// labels
	ls := Vector{-1, -1, -1, -1, -1, 1, 1, 1, 1, 1}

	// layer 1
	l1 := RandomMatrix(3, 4, 1)
	fmt.Println("l1=", l1)

	// feed forward
	a1, _ := ds.Multiply(l1)
	fmt.Println("a1=", a1)

	// rectified linear activation function
	rl := func(f Scalar) Scalar {
		if f > 0 {
			return f
		}
		return 0
	}

	// activate
	o1 := Activate(a1, rl)
	fmt.Println("o1=", o1)

	// calculate output error
	l2 := RandomMatrix(4, 1, 1)
	fmt.Println("l2=", l2)

	o2, _ := o1.Multiply(l2)
	fmt.Println("o2=", o2)

	e2 := Zeroes(len(ds), 1)
	for i := range o2 {
		// e1[i][0] = o2[i][0] - ls[i]
		e2[i][0] = (ls[i] - o2[i][0]) * (ls[i] - o2[i][0])
	}
	fmt.Println("e2=", e2)

	de2 := Zeroes(len(ds), 1)
	for i := range o2 {
		de2[i][0] = ls[i] - o2[i][0]
	}
	fmt.Println("de2=", de2)

	// back propagate
	// calculate derivative
	// d2, _ := o2.FrobeniusProduct(e1)
	// fmt.Println("d2=", d2)

	//calculate gradient
	g2 := Zeroes(len(o1), len(o1[0]))
	for i := range g2 {
		for j := range g2[i] {
			g2[i][j] = o1[i][j] * de2[i][0]
		}
	}
	fmt.Println("g2=", g2)

	d1 := Activate(o1, ReluDerivative)
	fp, _ := d1.FrobeniusProduct(g2)
	g1, _ := ds.Transpose().Multiply(fp)
	fmt.Println("g1=", g1)

	l1 = l1.Add(g1).Scale(0.00001)
	fmt.Println("l1=", l1)

	l2 = l2.Add(g2).Scale(0.00001)
	fmt.Println("l2=", l2)

	// pass 2

	// feed forward
	a1, _ = ds.Multiply(l1)
	fmt.Println("a1=", a1)

	// activate
	o1 = Activate(a1, rl)
	fmt.Println("o1=", o1)

	// calculate output error
	o2, _ = o1.Multiply(l2)
	fmt.Println("o2=", o2)

	e2 = Zeroes(len(ds), 1)
	for i := range o2 {
		e2[i][0] = (ls[i] - o2[i][0]) * (ls[i] - o2[i][0])
	}
	fmt.Println("e2=", e2)

	// rounds
	// for r := 0; r < 10; r++ {
	// 	// shuffle order
	// 	or := rand.Perm(10)
	// 	for _, i := range or {
	// 		// prediction
	// 		p, _ := ds[i].InnerProduct(w)
	// 		fmt.Printf("> Predicted; p=%f; l=%f; w=%v;\n", p, ls[i], w)

	// 		// weight adjustment
	// 		if p*ls[i] <= 0 {
	// 			w, _ = w.Add(ds[i].Scale(lr).Scale(ls[i]))
	// 			fmt.Printf("> > Adjusting weight; w=%v;\n", w)
	// 		}
	// 	}
	// }
}
