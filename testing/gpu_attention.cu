#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

//#define n 1024
//#define T 196
//#define D 384
//#define H 384
/*
__global__ void matrixmult_K_V(double *a, double *b, double *c, int H){
	int X = blockIdx.x*blockDim.x + threadIdx.x;
	int Y = blockIdx.y*blockDim.y + threadIdx.y;
	double local_c = 0;
	if(X < D/H && Y < D/H){
		for (int i = 0; i< T; i++)
			local_c += a[X*T + i] * b[i*D/H + Y];
		c[X*D/H + Y] = local_c;  
	}
}
__global__ void matrixmult_Q_KV(double *a, double *b, double *c, int H ){
	int X = blockIdx.x*blockDim.x + threadIdx.x;
	int Y = blockIdx.y*blockDim.y + threadIdx.y;
	double local_c = 0;
	if(X < T && Y < D/H){
		for (int i = 0; i< D/H; i++)
			local_c += a[X*D/H + i] * b[i*D/H + Y];
		c[X*D/H + Y] = local_c;  
	}
}
*/
__global__ void mmult(double *a, double *b, double *c, int m, int n, int k){
	int X = blockIdx.x*blockDim.x + threadIdx.x;
	int Y = blockIdx.y*blockDim.y + threadIdx.y;
	double local_c = 0;
	if(X < m && Y < k){
		for (int i = 0; i< n; i++)
			local_c += a[X*n + i] * b[i*k + Y];
		c[X*k + Y] = local_c;  
	}
}

__global__ void softmax(double *tempc, double *c, bool parallel, int row, int col){
	//int X = blockIdx.x;
	//int Y = threadIdx.x;
	int X = blockIdx.x*blockDim.x + threadIdx.x;
	int Y = blockIdx.y*blockDim.y + threadIdx.y;
	//softmax 
	if (X < row && Y < col){
		//parallel sum
		if (parallel){
			__shared__ double sum;

			double temp_sum = tempc[X*row+Y];
			double temp = tempc[X*row+Y];
			int count = 0;
			for (int i= (int)log2((double)col); i >=0; i--){
				
				if(Y<pow(2,i)){
					
					if((Y+pow(2,i)) < row){    //sum with [X][Y+2^i] 
						temp_sum += tempc[X* row + Y + (int)pow(2,i)];  
						count += 1;
					}
					//if(Y>=pow(2,i-1)||(i==0 && Y ==0))   //save to global so others can get it 
					tempc[X*row+Y] = temp_sum;
				}
				if(Y == 0 && i == 0)
					tempc[X*row] = temp_sum;
				__syncthreads(); 
			}
			c[X*row + Y] = temp/tempc[X*row];
		}
		//c[X*T + Y] = count ;
		else {
		
		//naive softmax 
		
			double local_sum = 0; 
			for(int i = 0; i < col ; i++)
				local_sum += tempc[X*row + i];
			c[X*row + Y] = tempc[X*row + Y]/local_sum;
		
		}
	}
}


int main(int argc, char const *argv[]){	
	float ms, ms_soft, ms_m1, ms_m2; // elapsed time in milliseconds
	int T,D,H;
    // receive input parameters
    sscanf(argv[1], "%d", &T);
    sscanf(argv[2], "%d", &D);
    sscanf(argv[3], "%d", &H);
    cudaEvent_t startEvent,stopEvent,startsoft, stopsoft;
	cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
	cudaEventCreate(&startsoft);
    cudaEventCreate(&stopsoft);
    int i;
    double *Q = (double*)malloc(sizeof(double)*T*D/H);          
	double *K = (double*)malloc(sizeof(double)*D/H*T);          
    double *KV= (double*)malloc(sizeof(double)*D/H*D/H);           	
    double *QKV=(double*)malloc(sizeof(double)*T*D/H);           	
    double *V = (double*)malloc(sizeof(double)*T*D/H);           	

	for(i=0; i<T*D/H; i++){
		Q[i]=1;
		K[i]=2;
		V[i]=1;
	}
		
	struct timespec start, stop; 
	double time;
	if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}

	cudaEventRecord(startEvent,0);

	double *gpu_Q, *gpu_K, *gpu_KV, *gpu_V, *gpu_QKV, *gpu_temp1, *gpu_temp2, *gpu_temp3, * gpu_QK;
	cudaMalloc((void**)&gpu_Q, sizeof(double)*T*D/H); 
	cudaMalloc((void**)&gpu_K, sizeof(double)*D/H*T);
	cudaMalloc((void**)&gpu_KV, sizeof(double)*D/H*D/H);
	cudaMalloc((void**)&gpu_V, sizeof(double)*T*D/H);
	cudaMalloc((void**)&gpu_QKV, sizeof(double)*T*D/H);
	cudaMalloc((void**)&gpu_temp1, sizeof(double)*T*D/H);
	cudaMalloc((void**)&gpu_temp2, sizeof(double)*T*D/H);
	cudaMalloc((void**)&gpu_temp3, sizeof(double)*T*T);
	cudaMalloc((void**)&gpu_QK, sizeof(double)*T*T);

	cudaMemcpy(gpu_Q, Q, sizeof(double)*T*D/H, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_K, K, sizeof(double)*D/H*T, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_V, V, sizeof(double)*D/H*T, cudaMemcpyHostToDevice);
	
	
	int max;
	if (T >= (D/H))
		max = T;
	else 
		max = D/H;
	dim3 dimGrid_1(max/16,max/16);
	dim3 dimBlock_1(16,16);
	//linear
/*
	softmax<<<dimGrid_1, dimBlock_1>>>(gpu_K,gpu_temp1,true, T,D/H);
	softmax<<<dimGrid_1, dimBlock_1>>>(gpu_Q,gpu_temp2,true, D/H,T);
	mmult<<<dimGrid_1, dimBlock_1>>>(gpu_temp1, gpu_V, gpu_KV, D/H,T,D/H);
	mmult<<<dimGrid_1, dimBlock_1>>>(gpu_temp2, gpu_KV, gpu_QKV, T,D/H,D/H);				
	*/
	
	//standard


	mmult<<<dimGrid_1, dimBlock_1>>>(gpu_Q, gpu_K, gpu_temp3,T,D/H,T);
	cudaEventRecord(startsoft, 0) ;

	softmax<<<dimGrid_1, dimBlock_1>>>(gpu_temp3,gpu_QK,true, T,T);
	cudaEventRecord(stopsoft, 0) ;


	mmult<<<dimGrid_1, dimBlock_1>>>(gpu_QK, gpu_V, gpu_QKV,T,T,D/H);	

	
	
	cudaEventRecord(stopEvent, 0) ;
	cudaEventSynchronize(stopEvent) ;
	cudaEventElapsedTime(&ms, startEvent, stopEvent) ;
	cudaEventElapsedTime(&ms_soft, startsoft, stopsoft) ;
	printf("Total time = %f,  Softmax time = %f,  percentage of softmax = %f with T = %d, H = %d\n", ms, ms_soft, ms_soft/ms, T,H);
	
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	//printf("Linear attention time is %f ns\n", time*1e9);	 
	
	cudaMemcpy(KV, gpu_KV, sizeof(double)*D/H*D/H, cudaMemcpyDeviceToHost);
	cudaMemcpy(QKV, gpu_QKV, sizeof(double)*T*D/H, cudaMemcpyDeviceToHost);

	return 0;
}	
