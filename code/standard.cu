#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define n 1024
#define T 196
#define D 384
#define H 12
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
	if(X < T && Y < D/H){
		for (int i = 0; i< T; i++)
			local_c += a[X*T + i] * b[i*D/H + Y];
		c[X*D/H + Y] = local_c;  
	}
}



int main(){	
	float ms; // elapsed time in milliseconds

    cudaEvent_t startEvent,stopEvent;
	cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    int i;
    int *Q = (int*)malloc(sizeof(int)*T*D/H);         
	int *K = (int*)malloc(sizeof(int)*D/H*T);         
    int *QK= (int*)malloc(sizeof(int)*T*T);           
    int *QKV=(int*)malloc(sizeof(int)*T*D/H);        
    int *V = (int*)malloc(sizeof(int)*T*D/H);           	

	for(i=0; i<T*D/H; i++){
		Q[i]=1;
		K[i]=2;
		V[i]=1;
	}
	
	struct timespec start, stop; 
	double time;
	if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}

	cudaEventRecord(startEvent,0);


	int *gpu_Q, *gpu_K, *gpu_QK, *gpu_V, *gpu_QKV;
	cudaMalloc((void**)&gpu_Q, sizeof(int)*T*D/H); 
	cudaMalloc((void**)&gpu_K, sizeof(int)*D/H*T);
	cudaMalloc((void**)&gpu_QK, sizeof(int)*T*T);
	cudaMalloc((void**)&gpu_V, sizeof(int)*T*D/H);
	cudaMalloc((void**)&gpu_QKV, sizeof(int)*T*D/H);
		

  
  
	cudaMemcpy(gpu_Q, Q, sizeof(int)*T*D/H, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_K, K, sizeof(int)*D/H*T, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_V, V, sizeof(int)*D/H*T, cudaMemcpyHostToDevice);
	
	dim3 dimGrid_1(T/16,T/16);
	dim3 dimBlock_1(16,16);
	dim3 dimGrid_2(T/16,D/H/16);
	dim3 dimBlock_2(16,16);
	
	matrixmult_Q_K<<<dimGrid_1, dimBlock_1>>>(gpu_Q, gpu_K, gpu_QK);
	
	matrixmult_QK_V<<<dimGrid_2, dimBlock_2>>>(gpu_QK, gpu_V, gpu_QKV);

	
	cudaEventRecord(stopEvent, 0) ;
	cudaEventSynchronize(stopEvent) ;
	cudaEventElapsedTime(&ms, startEvent, stopEvent) ;
	printf("Standard Time execute using cuda api(ms): %f   \n", ms);
	
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	//printf("standard attention time is %f ns\n", time*1e9);	 
	
	cudaMemcpy(QK, gpu_QK, sizeof(int)*T*T, cudaMemcpyDeviceToHost);
	cudaMemcpy(QKV, gpu_QKV, sizeof(int)*T*D/H, cudaMemcpyDeviceToHost);


	/*
	printf("QK[100][100] = %f\n",QK[100*T+100]);
	printf("should equal 2*D/H = 128\n");
	printf("QKV[100][30] = %f\n",QKV[100*D/H+30]);
	printf("should equal 196* 128 = 25088");
	*/
	return 0;
}	
