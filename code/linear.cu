#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define n 1024
#define T 196
#define D 384
#define H 12
__global__ void matrixmult_K_V(int *a, int *b, int *c){
	int X = blockIdx.x*blockDim.x + threadIdx.x;
	int Y = blockIdx.y*blockDim.y + threadIdx.y;
	int local_c = 0;
	if(X < D/H && Y < D/H){
		for (int i = 0; i< T; i++)
			local_c += a[X*T + i] * b[i*D/H + Y];
		c[X*D/H + Y] = local_c;  
	}
}
__global__ void matrixmult_Q_KV(int *a, int *b, int *c){
	int X = blockIdx.x*blockDim.x + threadIdx.x;
	int Y = blockIdx.y*blockDim.y + threadIdx.y;
	int local_c = 0;
	if(X < T && Y < D/H){
		for (int i = 0; i< D/H; i++)
			local_c += a[X*D/H + i] * b[i*D/H + Y];
		c[X*D/H + Y] = local_c;  
	}
}



int main(){	
	
    int i;
    int *Q = (int*)malloc(sizeof(int)*T*D/H);          
	int *K = (int*)malloc(sizeof(int)*D/H*T);          
    int *KV= (int*)malloc(sizeof(int)*D/H*D/H);           	
    int *QKV=(int*)malloc(sizeof(int)*T*D/H);           	
    int *V = (int*)malloc(sizeof(int)*T*D/H);           	

	for(i=0; i<T*D/H; i++){
		Q[i]=1;
		K[i]=2;
		V[i]=1;
	}
	
	int *gpu_Q, *gpu_K, *gpu_KV, *gpu_V, *gpu_QKV;
	cudaMalloc((void**)&gpu_Q, sizeof(int)*T*D/H); 
	cudaMalloc((void**)&gpu_K, sizeof(int)*D/H*T);
	cudaMalloc((void**)&gpu_KV, sizeof(int)*D/H*D/H);
	cudaMalloc((void**)&gpu_V, sizeof(int)*T*D/H);
	cudaMalloc((void**)&gpu_QKV, sizeof(int)*T*D/H);
		
	struct timespec start, stop; 
	double time;
  
  
	cudaMemcpy(gpu_Q, Q, sizeof(int)*T*D/H, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_K, K, sizeof(int)*D/H*T, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_V, V, sizeof(int)*D/H*T, cudaMemcpyHostToDevice);
	
	dim3 dimGrid_1(4,4);
	dim3 dimBlock_1(16,16);
	dim3 dimGrid_2(16,4);
	dim3 dimBlock_2(16,16);
	
	if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}
	matrixmult_K_V<<<dimGrid_1, dimBlock_1>>>(gpu_K, gpu_V, gpu_KV);
	
	matrixmult_Q_KV<<<dimGrid_2, dimBlock_2>>>(gpu_Q, gpu_KV, gpu_QKV);				
	
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("Linear attention time is %f ns\n", time*1e9);	 
	
	cudaMemcpy(KV, gpu_KV, sizeof(int)*D/H*D/H, cudaMemcpyDeviceToHost);
	cudaMemcpy(QKV, gpu_QKV, sizeof(int)*T*D/H, cudaMemcpyDeviceToHost);


/*
	printf("QK[50][50] = %d\n",KV[50*D/H+50]);
	printf("should equal 2*T = 392\n");
	printf("QKV[100][30] = %d\n",QKV[100*D/H+30]);
	printf("should equal 392* 64 = 25088");
*/
	return 0;
}	
