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
func (v Vector) InnerProduct(m Vector) (p Scalar) {
	if len(v) != len(m) {
		panic(errVectorsNotEqual)
	}
	for i := range v {
		p += v[i] * m[i]
	}
	return p
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
func (v Vector) Add(m Vector) (r Vector) {
	if len(v) != len(m) {
		panic(errVectorsNotEqual)
	}
	r = make(Vector, len(v))
	for i, s := range v {
		r[i] = s + m[i]
	}
	return r
}

// Matrix -
type Matrix []Vector

// Row -
func (m Matrix) Row(i int) Vector {
	return m[i]
}

// Col -
func (m Matrix) Col(i int) Vector {
	v := Vector{}
	for j := range m {
		v = append(v, m[j][i])
	}
	return v
}

// Zeroes -
func Zeroes(m, n int) Matrix {
	r := make(Matrix, m)
	for i := 0; i < m; i++ {
		r[i] = make(Vector, n)
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
func (m Matrix) Multiply(m2 Matrix) Matrix {
	// TODO(geoah) Handle error
	r := Zeroes(len(m), len(m2[0]))
	for i := range m {
		for j := range m2[0] {
			col := m2.Col(j)
			row := m.Row(i)
			ip := row.InnerProduct(col)
			// fmt.Println(i, j, row, col, ip)
			r[i][j] = ip
		}
	}
	return r
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
func (m Matrix) FrobeniusProduct(m2 Matrix) Matrix {
	// TODO(geoah) Handle error
	r := Zeroes(len(m), len(m[0]))
	for i := range m {
		for j := range m[i] {
			r[i][j] = m[i][j] * m2[i][j]
		}
	}
	return r
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
func (m Matrix) MultiplyWithVector(v Vector) Vector {
	r := Vector{}
	for i := range m {
		iv := m[i].InnerProduct(v)
		r = append(r, iv)
	}
	return r
}

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

	// labels
	ls := Vector{-1, -1, -1, -1, -1, 1, 1, 1, 1, 1}

	// Define layer 1 size and initialization type:
	//	layer1_size=100;
	//	layer1= (1/sqrt(size(XX,2))) * randn(size(XX,2),layer1_size);

	// layer 1
	l1 := RandomMatrix(3, 4, 1)
	fmt.Println("l1=", l1)

	// feed forward layer 1:
	//z1=thisX*layer1;
	//a1=[max(z1,0), ones(size(thisX,1),1)];
	z1 := ds.Multiply(l1)
	fmt.Println("z1=", z1)

	// rectified linear activation function
	rl := func(f Scalar) Scalar {
		if f > 0 {
			return f
		}
		return 0
	}

	//relu derivative:
	drl := func(f Scalar) Scalar {
		if f > 0 {
			return 1
		}
		return 0
	}
	// activate
	a1 := Activate(z1, rl)
	fmt.Println("a1=", a1)

	// %feed forward again
	//         z2=a1*layer2;
	//         a2=max(0,z2);
	l2 := RandomMatrix(4, 1, 1)
	fmt.Println("l2=", l2)

	z2 := a1.Multiply(l2)
	fmt.Println("z2=", z2)
	a2 := Activate(z2, rl)
	fmt.Println("a2=", a2)

	// calculate output error derivative
	// %back prop
	//         %d2 = (a2-thisY).*(z2>0);
	//         d2=a2-thisY;
	d2 := Zeroes(len(ds), 1)
	for i := range a2 {
		// e1[i][0] = o2[i][0] - ls[i]
		d2[i][0] = (a2[i][0] - ls[i])
	}
	fmt.Println("d2=", d2)

	//calculate gradient for layer 2:
	//df2=a1'* d2;
	df2 := a1.Transpose().Multiply(d2)
	fmt.Println("df2=", df2)

	// %backprop again
	// d1 = ( d2 * layer2( 1: end-1,:)' ).* (z1>0);
	// df1 = thisX'*d1;

	d1 := d2.Multiply(l2.Transpose()).FrobeniusProduct(Activate(z1, drl))
	df1 := ds.Transpose().Multiply(d1)
	fmt.Println("df1=", df1)

	// back propagate
	// calculate derivative
	// d2, _ := o2.FrobeniusProduct(e1)
	// fmt.Println("d2=", d2)

	//calculate gradient
	//g2 := Zeroes(len(o1), len(o1[0]))
	// for i := range g2 {
	// 	for j := range g2[i] {
	// 		g2[i][j] = o1[i][j] * de2[i][0]
	// 	}
	// }
	// fmt.Println("g2=", g2)
	//
	// d1 := Activate(o1, ReluDerivative)
	// fp, _ := d1.FrobeniusProduct(g2)
	// g1, _ := ds.Transpose().Multiply(fp)
	// fmt.Println("g1=", g1)

	// l1 = l1.Add(g1).Scale(0.00001)
	// fmt.Println("l1=", l1)
	//
	// l2 = l2.Add(g2).Scale(0.00001)
	// fmt.Println("l2=", l2)
	//

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
