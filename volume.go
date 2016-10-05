package layer 

//Volume is a basic three-dimensional Volume passed between layers
//dimensionality is MxNxD
type Volume struct{
    m,n,d int
    data [][][]float
}

//create volume with specific size

//add to volume in place

//add two volumes

//element-wise multiply of volumes

//scale a volume

//return a new volume after applying function f on it

