package layer
import "fmt"


//layer should set data, calculate transfer and activation for feeding forward
//calculate derivative of activation faction for propagating backward
type Layer interface{
 GetType()  int//to enumerate: input/output, hidden, convolutional,dropout other?
 SetSize() 
 Transfer() [][]float
 Activate() [][]float
 BackProp() [][]float
}

//transfer function

//activation function

//derivative of activation function

