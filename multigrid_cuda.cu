#include <stdio.h> 
#include <string.h> 
#include <stddef.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


#define N 100
#define MAX_LEVELS 10
#define TOLERANCE 0.0001

__global__ void init_pade(double *A){
    int i = blockIdx.x, j = threadIdx.x;
    if (i == j){
        if (i==0 || i==(N-1))
            A[i*N + j] = 1.0;

        else 
            A[i*N + j] = 4.0;
    }
    else if (abs(i - j) == 1){
        if (i==0 || i==(N-1))
            A[i*N + j] = 2.0;

        else 
            A[i*N + j] = 1.0;
    }
    else
        A[i*N + j] = 0.0;
}

__global__ void init_tridiag(double *A){
    int i = blockIdx.x, j = threadIdx.x;
    if (i == j)
        A[i*N + j] = 2.0;
    else if (abs(i - j) == 1)
        A[i*N + j] = -1.0;
    else
        A[i*N + j] = 0.0;
}

__global__ void init_b(double *b, double *f){
    double delta = (double)3/N;
    for(int i=0;i<N;i++)
    {
        f[i]=sin(5*i*delta);
        printf("%f , ",f[i]);
    }

    for (int i = 1; i < N-1; i++) {
        b[i] = 3*(f[i+1]-f[i-1])/delta;
    }

    b[N-1]=(2.5*f[N-1]-2*f[N-2]-0.5*f[N-3])/delta;
    b[0]=(-2.5*f[0]+2*f[1]+0.5*f[2])/delta;
}

__global__ void init_b(double* b){
    int i = threadIdx.x;
    b[i] = 1;
}

__global__ void init_M(double* M, double *restriction, double *A){
    int i = blockIdx.x, j = threadIdx.x;
    for (int k=0; k<N; k++) {
        M[i*N + j] += restriction[i*N + k]*A[k*N + j];
    }
}

__global__ void init_pro_rest(double *prolongation, double *restriction){
    int i = blockIdx.x, j = threadIdx.x;
    if ((i%2)!=0 && j==(i-1)/2)
        prolongation[i*(N/2) + j]=1;
    if ((i%2)==0 && j==(i/2))
        prolongation[i*(N/2) + j]=1;
    else 
        prolongation[i*(N/2) + j]=0;

    restriction[j*N + i]=prolongation[i*(N/2) + j];
}

__global__ void init_coarse(double *A, double *A_coarse){
    int i = blockIdx.x, j = threadIdx.x;
    A_coarse[(i/2)*(N/2)+(j/2)]= A[i*N+j]+ A[(i+1)*N+j]+A[i*N+j+1]+A[(i+1)*N+j+1];
}

__global__ void init_coarse(double *A_coarse, double *M, double *prolongation){
    int i = blockIdx.x, j = threadIdx.x;
    for( int k=0; k<N; k++ ){
        A_coarse[i*(N/2)+j]+= M[i*N+k]*prolongation[k*(N/2) + j];
    }
}

__global__ void gauss_seidel(double *A, double *b, double *x, double* residual, bool fl) {
    int i = threadIdx.x;
    double sum = b[i];
    for ( j = 0; j < N; j++) {
        if (j != i)
            sum -= A[i*N+j] * x[j];
    }
    if(fl)
        x[i] = sum / A[i*N+i];
    else
        residual[i] = sum;
}


__global__ void gauss_seidel_coarse(double *A_coarse, double *residual_coarse, double *correction_coarse) {
    int i = threadIdx.x;
    double sum = residual_coarse[i];
    for ( j = 0; j < N/2; j++) {
        if (j != i)
            sum -= A_coarse[i*(N/2)+j] * correction_coarse [j];
    }
    correction_coarse [i] = sum / A_coarse[i*(N/2)+i];
}

__global__ void calculate_res(double* residual_coarse, double* restriction, double* residual){
    int i = blockIdx.x, k = threadIdx.x;
    residual_coarse[i] += restriction[i*N+k]*residual[k];
}

__global__ void calculate_cor(double* correction, double* prolongation, double* correction_coarse){
    int i = blockIdx.x, k = threadIdx.x;
    correction[i]= prolongation[i*(N/2)+k]*correction_coarse[k];
}

__global__ void update_x(double* x, double* prev_x, double* correction, bool fl){
    int i = threadIdx.x;
    if(fl)
        x[i] += correction[i];
    else
        prev_x[i] = x[i];
}

int main(int argc, char* argv[]) {
    cudaMalloc((void**)&A, N*N*sizeof(double));
    cudaMalloc((void**)&b, N*sizeof(double));
    cudaMalloc((void**)&x, N*sizeof(double));
    cudaMalloc((void**)&restriction, (N/2)*N*sizeof(double));
    cudaMalloc((void**)&prolongation, N*(N/2)*sizeof(double));
    cudaMalloc((void**)&A_coarse, (N/2)*(N/2)*sizeof(double));
    cudaMemset(x, 0, N*sizeof(double));

    init_pade<<< N,N >>>(A);
    cudaMalloc((void**)&f, N*sizeof(double));
    init_b<<< 1,1 >>>(b,f);

    // init_tridiag<<< N,N >>>(A);
    // init_b<<< 1,N >>>(b);

    init_pro_rest<<< N,N/2 >>>(prolongation, restriction);

    // cudaMalloc((void**)&M, (N/2)*N*sizeof(double));
    // init_M<<< N/2,N >>>(M,restriction,A);
    // init_coarse<<< N/2,N/2 >>>(A_coarse,M,prolongation)

    init_coarse<<< N,N >>>(A, A_coarse);

    cudaMalloc((void**)&residual, N*sizeof(double));
    cudaMalloc((void**)&correction, N*sizeof(double));
    cudaMalloc((void**)&residual_coarse, (N/2)*sizeof(double));
    cudaMalloc((void**)&correction_coarse, (N/2)*sizeof(double));
    cudaMalloc((void**)&prev_x, N*sizeof(double));
    cudaMalloc((void**)&error, sizeof(double));
 
    gauss_seidel<<< 1,N >>>(A, b, x, residual, true);

    do {
        update_x<<< 1,N >>>(x, correction, false);
        gauss_seidel<<< 1,N >>>(A, b, x, residual, false);
       
        for (int level = 1; level < MAX_LEVELS; level++) {
            cudaMemset(residual_coarse, 0, (N/2)*sizeof(double));
            cudaMemset(correction_coarse, 0, (N/2)*sizeof(double));
            cudaMemset(correction, 0, N*sizeof(double));
            calculate_res<<< N/2,N >>>(residual_coarse,restriction,residual);
            
            gauss_seidel_coarse<<< 1,N/2 >>>(A_coarse, residual_coarse, correction_coarse);

            calculate_cor<<< N,N/2 >>>(correction,prolongation,correction_coarse);
            update_x<<< 1,N >>>(x, correction, true);

            gauss_seidel<<< 1,N >>>(A, b, x, residual, true);
            gauss_seidel<<< 1,N >>>(A, b, x, residual, false);
        }

        cudaMemset(error, 0, sizeof(double));
        for (int i = 0; i < N; i++) {
            error += fabs(x[i] - prev_x[i]);
        }
        // printf("Final x vector:\n");
        // for (int i = 0; i < N; i++) {
        //     printf("%.6f , ", x[i]);
        // }printf("\n");

    } while (error > TOLERANCE);

    double x_host[N];
    cudaMemcpy(x_host, x, N*sizeof(double), cudaMemcpyDeviceToHost);
    printf("Final x vector:\n");
    for (int i = 0; i < N; i++) {
        printf("%.6f , ", x_host[i]);
    }

  return 0;
}