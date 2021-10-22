# Tiled Matrix Multiplication - OpenMP

Tiling is an important technique for extraction of parallelism. Informally, tiling consists of partitioning the iteration space into several chunk of computation called tiles (blocks) such that sequential traversal of the tiles covers the entire iteration space. Hence tiling increases the granularity of computation and decreases the amount of communication incurred between processors.

The OpenMP-enabled parallel code exploits coarse grain parallelism, which makes use of the cores available in a multicore machine. During experimentation multiplication of matrices were carried out serially as well as in parallel and results were verified. 


## Block/ Tile Size

Tile size of 64 was declared.  

```#define tile_size 64```
 
 
## Compilation and Execution 

```
g++ -fopenmp TiledMatrixMultiplication.cpp -o matmul

./matmul 256
```

For matrix dimension of NXN  
```
./matmul N
```

### Output
```
                ........................................................................
 	                      Matrix Multiplication (Serial VS Tiled)                          
                ........................................................................


		 Matrix into Matrix Multiplication (Serial) ......Done 

		 Time in Seconds (T)        : 0.125315 Seconds 

		 ( T represents the Time taken for computation )

		 Matrix into Matrix Multiplication (Tiled) ......Done 

		 Time in Seconds (T)        : 0.086112 Seconds 

		 ( T represents the Time taken for computation )
		..........................................................................
Done Checking : Both Computations are same!! 

```

## Results

DIMENSION  | SERIAL | PARALLEL | SPEED UP
--- | --- | :---:|:---:
64 |0.002516| 0.002136| 1.177902622
128| 0.011201| 0.011197| 1.000357239
256 |0.112253| 0.075235 |1.492031634
512 |1.488146 |0.695457 |2.139810226
1024| 12.283523 |5.515929| 2.226918258
2048 |151.039676| 42.364133 |3.565272444
4096| 1505.435873| 381.263239 |3.948547143
8192| 7659.253458 |2516.833489 |3.04321024


![image](https://user-images.githubusercontent.com/67643621/138514062-cf05b9a1-f94c-444b-bacb-24f032241ba0.png) ![image](https://user-images.githubusercontent.com/67643621/138514172-97c9059e-7b2c-4abf-9e27-9d6934a0e0ad.png)

