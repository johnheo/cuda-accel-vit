#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>
#include <nvToolsExt.h>

#define TILE_WIDTH 16

__global__ void mmult(double *a, double *b, double *c, int m, int n, int k){
	int X = blockIdx.x*blockDim.x + threadIdx.x;
	int Y = blockIdx.y*blockDim.y + threadIdx.y;
	//double aa = 1;
	//double bb = 1;
	double local_c = 0;
	if(X < m && Y < k){
		for (int i = 0; i< n; i++){
			 //aa = a[X*n + i]; bb = b[i*k + Y];
			 local_c += a[X*n + i]*b[i*k + Y];
			 //local_c += aa * bb;
		}
		c[X*k + Y] = local_c;  
	}
}

__global__ void mmult_s(double *a, double *b, double *c, int m, int n, int k){
	//int X = blockIdx.x*blockDim.x + threadIdx.x;
	//int Y = blockIdx.y*blockDim.y + threadIdx.y;
	//double aa = 1;
	//double bb = 1;
	double local_c = 0;
	for (int Y = 0; Y< k; Y++){
		for (int X = 0; X< m; X++){
			for (int i = 0; i< n; i++){
				 //aa = a[X*n + i]; bb = b[i*k + Y];
				 local_c += a[X*n + i]*b[i*k + Y];
				 //local_c += aa * bb;
			}
			c[X*k + Y] = local_c;  
		}
	}
}

__global__ void mmult_shared(double * A, double * B, double * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    __shared__ double ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ double ds_N[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y,
       Row = by * TILE_WIDTH + ty,
       Col = bx * TILE_WIDTH + tx;
    double Pvalue = 0;

    for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
		
       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
          ds_M[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
		  //ds_M[ty][tx] = 0;
       else
          ds_M[ty][tx] = 0;
       if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
          ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
		  //ds_N[ty][tx] = 0;
       else
          ds_N[ty][tx] = 0;
		
       __syncthreads();
	   
       for (int k = 0; k < TILE_WIDTH; ++k)
          Pvalue += ds_M[ty][k] * ds_N[k][tx];
       __syncthreads();
	   
    }
	
    if (Row < numCRows && Col < numCColumns)
       C[Row*numCColumns+Col] = Pvalue;
   
}

__global__ void softmax_all_p(double *c, int row, int col, double *sum, double *tempc){
	int X = blockIdx.x*blockDim.x + threadIdx.x;
	int Y = blockIdx.y*blockDim.y + threadIdx.y;
	//softmax 
	if (X < row && Y < col){
		//__shared__ double ssum[400];

		double temp_sum = c[X*col+Y];
		//double temp = c[X*col+Y];
		tempc[X*col+Y] = c[X*col+Y];
		//ssum[Y] = c[X*col +Y];
		int count = 0;
		for (int i= (int)log2((double)col); i >=0; i--){
			
			if(Y<pow(2,i)){
				
				if((Y+pow(2,i)) < col){    //sum with [X][Y+2^i] 
					temp_sum += tempc[X* col + Y + (int)pow(2,i)];
					//temp_sum += ssum[Y + (int)pow(2,i)];
					count += 1;
				}
				if(Y>=pow(2,i-1 && Y != 0))   //save to global so others can get it 
					tempc[X*col+Y] = temp_sum;
				//ssum[col] = temp_sum;
			}
			if(Y == 0 && i == 0)
				sum[X] = temp_sum;
			__syncthreads(); 
		}	
		//c[X*col + Y] = exp(tempc[X*col + Y])/sum[X];
		__syncthreads();
		c[X*col + Y] = exp(c[X*col + Y])/sum[X];

	}
}
__global__ void softmax_sum(double *c, int row, int col, double *sum){
	int X = blockIdx.x*blockDim.x + threadIdx.x;
	int Y = blockIdx.y*blockDim.y + threadIdx.y;
	//softmax 
	if (X < row && Y < col){
		if(Y == 0){

			double local_sum = 0; 
			for(int i = 0; i < col ; i++)
				local_sum += exp(c[X*col + i]);
			sum[X] = local_sum;
		}
	}
}
__global__ void softmax_probability(double *c, int row, int col, double *sum){
	int X = blockIdx.x*blockDim.x + threadIdx.x;
	int Y = blockIdx.y*blockDim.y + threadIdx.y;
	//softmax 
	if (X < row && Y < col){
		c[X*col + Y] = exp(c[X*col + Y])/sum[X];
	}
}

int main(int argc, char const *argv[]){	
	float ms, ms_soft, ms1, ms2; // elapsed time in milliseconds
	int T,D,H;
    // receive input parameters
    sscanf(argv[1], "%d", &T);
    sscanf(argv[2], "%d", &D);
    sscanf(argv[3], "%d", &H);

    cudaEvent_t startEvent,stopEvent,startsoft, stopsoft, startm1, startm2, stopm1, stopm2;
	cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
	cudaEventCreate(&startsoft);
    cudaEventCreate(&stopsoft);
	cudaEventCreate(&startm1);
    cudaEventCreate(&stopm1);
	cudaEventCreate(&startm2);
    cudaEventCreate(&stopm2);
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
		
	//struct timespec start, stop; 
	//double time;
	//if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}

	cudaEventRecord(startEvent,0);
	int max;
	if (T >= (D/H))
		max = T;
	else 
		max = D/H;
	double *gpu_Q, *gpu_K, *gpu_KV, *gpu_V, *gpu_QKV, *gpu_temp1, * gpu_QK, *sum;
	cudaMalloc((void**)&gpu_Q, sizeof(double)*T*D/H); 
	cudaMalloc((void**)&gpu_K, sizeof(double)*D/H*T);
	cudaMalloc((void**)&gpu_KV, sizeof(double)*D/H*D/H);
	cudaMalloc((void**)&gpu_V, sizeof(double)*T*D/H);
	cudaMalloc((void**)&gpu_QKV, sizeof(double)*T*D/H);
	cudaMalloc((void**)&gpu_temp1, sizeof(double)*T*D/H);
	cudaMalloc((void**)&gpu_QK, sizeof(double)*T*T);
	cudaMalloc((void**)&sum, sizeof(double)*max);
	cudaMemcpy(gpu_Q, Q, sizeof(double)*T*D/H, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_K, K, sizeof(double)*D/H*T, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_V, V, sizeof(double)*D/H*T, cudaMemcpyHostToDevice);
	
	

	dim3 dimGrid_1((max+TILE_WIDTH)/TILE_WIDTH,(max+TILE_WIDTH)/TILE_WIDTH);
	dim3 dimBlock_1(TILE_WIDTH,TILE_WIDTH);
	
	//linear attention
	//---------------------------------------------------------
	/*
	//softmax 
	cudaEventRecord(startsoft, 0) ;
	//softmax_sum<<<dimGrid_1, dimBlock_1>>>(gpu_K, T,D/H, sum);
	//softmax_probability<<<dimGrid_1, dimBlock_1>>>(gpu_K, T,D/H, sum);
	softmax_all_p<<<dimGrid_1, dimBlock_1>>>(gpu_K, T,D/H, sum,gpu_temp1);

	//softmax_sum<<<dimGrid_1, dimBlock_1>>>(gpu_Q,false, D/H,T, sum);
	//softmax_probability<<<dimGrid_1, dimBlock_1>>>(gpu_Q, D/H,T, sum);
	softmax_all_p<<<dimGrid_1, dimBlock_1>>>(gpu_Q, D/H,T, sum,gpu_temp1);

	cudaEventRecord(stopsoft, 0) ;

	//matmult1
	dim3 dimGrid_2((D/H+TILE_WIDTH)/TILE_WIDTH,(D/H+TILE_WIDTH)/TILE_WIDTH);
	dim3 dimBlock_2(TILE_WIDTH,TILE_WIDTH);
	
	cudaEventRecord(startm1, 0) ;
	mmult<<<dimGrid_2, dimBlock_2>>>(gpu_K, gpu_V, gpu_KV, D/H,T,D/H);
	//mmult_shared<<<dimGrid_2, dimBlock_2>>>(gpu_K, gpu_V, gpu_V,D/H,T,T,D/H,D/H,D/H);	
	cudaEventRecord(stopm1, 0) ;
	
	//matmult2
	dim3 dimGrid_3((T+TILE_WIDTH)/TILE_WIDTH,(D/H+TILE_WIDTH)/TILE_WIDTH);
	dim3 dimBlock_3(TILE_WIDTH,TILE_WIDTH);
	
	cudaEventRecord(startm2, 0) ;
	mmult<<<dimGrid_3, dimBlock_3>>>(gpu_Q, gpu_KV, gpu_QKV, T,D/H,D/H);
	//mmult_shared<<<dimGrid_3, dimBlock_3>>>(gpu_Q, gpu_KV, gpu_QKV,T,D/H,D/H,D/H,T,D/H);	
	cudaEventRecord(stopm2, 0) ;
	*/
	
	

	//standard attention
	// --------------------------------------------------------

	dim3 dimGrid_2((T+TILE_WIDTH)/TILE_WIDTH,(T+TILE_WIDTH)/TILE_WIDTH);
	dim3 dimBlock_2(TILE_WIDTH,TILE_WIDTH);
	
	cudaEventRecord(startm1, 0) ;
	
	mmult<<<dimGrid_2, dimBlock_2>>>(gpu_Q, gpu_K, gpu_QK,T,D/H,T);
	//mmult_shared<<<dimGrid_2, dimBlock_2>>>(gpu_Q, gpu_K, gpu_QK,T,D/H,D/H,T,T,T);
	cudaEventRecord(stopm1, 0) ;

	
	cudaEventRecord(startsoft, 0) ;
	//softmax<<<dimGrid_1, dimBlock_1>>>(gpu_temp3,gpu_QK,false, T,T, sum);
	softmax_sum<<<dimGrid_1, dimBlock_1>>>(gpu_QK, T,T, sum);
	softmax_probability<<<dimGrid_1, dimBlock_1>>>(gpu_QK, T,T, sum);
	cudaEventRecord(stopsoft, 0) ;
	
	
	dim3 dimGrid_3((T+TILE_WIDTH)/TILE_WIDTH,(D/H+TILE_WIDTH)/TILE_WIDTH);
	dim3 dimBlock_3(TILE_WIDTH,TILE_WIDTH);

	cudaEventRecord(startm2, 0) ;
	mmult<<<dimGrid_3, dimBlock_3>>>(gpu_QK, gpu_V, gpu_QKV,T,T,D/H);	
	//mmult_shared<<<dimGrid_3, dimBlock_3>>>(gpu_QK, gpu_V, gpu_QKV,T,T,T,D/H,T,D/H);	
	cudaEventRecord(stopm2, 0) ;
	

	//-----------------------------------------------------------
	cudaEventRecord(stopEvent, 0) ;
	cudaEventSynchronize(stopEvent) ;
	cudaEventElapsedTime(&ms, startEvent, stopEvent) ;
	cudaEventElapsedTime(&ms_soft, startsoft, stopsoft) ;
	cudaEventElapsedTime(&ms1, startm1, stopm1) ;
	cudaEventElapsedTime(&ms2, startm2, stopm2) ;
	//printf("Total time = %f,  mmult1 = %f, mmult2 = %f, Softmax time = %f,  percentage of softmax = %f with T = %d, H = %d\n", ms, ms1,ms2,ms_soft, ms_soft/ms, T,H);
	printf("%d  %f  %f  %f  %f \n", T, ms, ms1,ms_soft,ms2);
	
	//if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
	//time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;


	return 0;
}	
