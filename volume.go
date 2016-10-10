package main
import "fmt"
import "strconv"
//Volume is a basic three-dimensional Volume passed between layers
//dimensionality is MxNxD
type Volume struct{
    m,n,d int
    data [][][]float64
}

//create volume with specific size
func Init(m int,n int,d int) Volume{
  V:=Volume{m,n,d,nil}
  V.data =make([][][]float64, d)
  for i:=0;i<d;i++{
    V.data[i]=make([][]float64,m)
    for j:=0; j<m;j++{
      V.data[i][j]=make([]float64,n)
    }
  }
  return V
}

//print contents of a volume
func (V Volume) String() string{
  rtn:=""
  for i:=0;i<V.d;i++{
    for j:=0;j<V.m;j++{
      rtn += strconv.Itoa(i)+": "+strconv.Itoa(j)+": "
      for k:=0;k<V.m;k++{
        rtn+=strconv.FormatFloat(V.data[i][j][k],'e',-1,64)+" "
      }
      rtn+="\n"
    }
  }
  return rtn
}

//add to volume in place

//func (V Volume)Adj(V)

//add two volumes

//element-wise multiply of volumes

//scale a volume

//return a new volume after applying function f on it
func main(){
  V:=Init(2,2,2)
  //V.d=2
  fmt.Println("YO BABY!",V)
}
