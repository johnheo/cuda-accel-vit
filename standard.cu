#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define n 1024
#define T 196
#define D 384
#define H 6
__global__ void matrixmult_Q_K(int *a, int *b, int *c){
	int X = blockIdx.x*blockDim.x + threadIdx.x;
	int Y = blockIdx.y*blockDim.y + threadIdx.y;
	int local_c = 0;
	if(X < T && Y < T){
		for (int i = 0; i< D/H; i++)
			local_c += a[X*D/H + i] * b[i*T + Y];
		c[X*T + Y] = local_c;  
	}
}
__global__ void matrixmult_QK_V(int *a, int *b, int *c){
	int X = blockIdx.x*blockDim.x + threadIdx.x;
	int Y = blockIdx.y*blockDim.y + threadIdx.y;
	int local_c = 0;
	if(X < T && Y < T){
		for (int i = 0; i< D/H; i++)
			local_c += a[X*D/H + i] * b[i*T + Y];
		c[X*T + Y] = local_c;  
	}
}



int main(){	
	//H=atoi(argv[1]);
	
    int i;
    int *Q = (int*)malloc(sizeof(int)*T*D/H);          
	int *K = (int*)malloc(sizeof(int)*D/H*T);          
    int *QK= (int*)malloc(sizeof(int)*T*T);           	

	for(i=0; i<T*D/H; i++){
		Q[i]=1;
		K[i]=2;
	}
	int *gpu_Q, *gpu_K, *gpu_QK;
	cudaMalloc((void**)&gpu_Q, sizeof(int)*T*D/H); 
	cudaMalloc((void**)&gpu_K, sizeof(int)*D/H*T);
	cudaMalloc((void**)&gpu_QK, sizeof(int)*T*T);
		
	struct timespec start, stop; 
	double time;
  
  
	cudaMemcpy(gpu_Q, Q, sizeof(int)*T*D/H, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_K, K, sizeof(int)*D/H*T, cudaMemcpyHostToDevice);
	
	dim3 dimGrid(16,16);
	dim3 dimBlock(16,16);
	
	if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}
	matrixmult_Q_K<<<dimGrid, dimBlock>>>(gpu_Q, gpu_K, gpu_QK);//matrixmult_QK_V<<<, >>>(gpu_a, gpu_b, gpu_c);				
	cudaMemcpy(QK, gpu_QK, sizeof(int)*T*T, cudaMemcpyDeviceToHost);
	
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("time is %f ns\n", time*1e9);	 
	

	printf("QK[%d]=%d ", i, QK[5*T + 5], 'should equal 2*D/H = 128');


	return 0;
}	
