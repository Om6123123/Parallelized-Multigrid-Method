#include <stdio.h> 
#include <string.h> 
#include <stddef.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include<omp.h>
#endif


#define N 100
#define MAX_LEVELS 10
#define TOLERANCE 0.0001

double func(double x)
{
    double q= sin(5*x);
    return q;
}

void initialize_system(double A[N][N], double b[N] , double prolongation[N][N/2], double restriction[N/2][N],  double A_coarse[N/2][N/2]) {
    
    double delta = (double)3/N;
    for (int i = 0; i < N; i++) { 
        for (int j = 0; j < N; j++) {
            if (i == j){
                if (i==0 || i==(N-1))
                A[i][j] = 1.0;

                else 
                A[i][j] = 4.0;
            }
            else if (abs(i - j) == 1){
                 if (i==0 || i==(N-1))
                  A[i][j] = 2.0;

                 else 
                 A[i][j] = 1.0;

            }
            else

                A[i][j] = 0.0;
        }
    }
    
    double f[N];
    for(int i=0;i<N;i++)
    {
        f[i]=func(i*delta);
        printf("%f , ",f[i]);
    }

    for (int i = 1; i < N-1; i++) {
        b[i] = 3*(f[i+1]-f[i-1])/delta;
    }

    b[N-1]=(2.5*f[N-1]-2*f[N-2]-0.5*f[N-3])/delta;
    b[0]=(-2.5*f[0]+2*f[1]+0.5*f[2])/delta;

     
    // for (int i = 0; i < N; i++) {
    //  for (int j = 0; j < N; j++) {
    //         if (i == j)
    //             A[i][j] = 2.0;
    //         else if (abs(i - j) == 1)
    //             A[i][j] = -1.0;
    //         else
    //        A[i][j] = 0.0;
    //     }
    // }

    // for (int i = 0; i < N; i++) {
    //     b[i] = 1.0;
    // }

     for (int i=0; i<N; i++){

        for (int j=0; j<N/2; j++){
          
          if ((i%2)!=0 && j==(i-1)/2)
            prolongation[i][j]=1;
          if ((i%2)==0 && j==(i/2))
             prolongation[i][j]=1;
          else 
            prolongation[i][j]=0;
          
        }
    }

    for (int i=0; i<N ;i++){
        for (int j=0; j<N/2; j++){

           restriction[j][i]=prolongation[i][j];

        }
    }
    // double M[N/2][N];
    // for (int i=0; i<N/2; i++){
    //     for (int j=0; j<N; j++){
    //         for (int k=0; k<N; k++) {
         
    //       M[i][j] += restriction[i][k]*A[k][j];

    //         }
    //     }
    // }

    

    // for(int i=0; i<N/2; i++){
    //     for(int j=0; j<N/2; j++){
    //       for( int k=0; k<N; k++ ){
    //       A_coarse[i][j]+= M[i][k]*prolongation[k][j];
    //       }
    //     }
    // }

    for(int i=0; i<N; i+=2){
        for(int j=0; j<N; j+=2){
            A_coarse[i/2][j/2]= A[i][j]+ A[i+1][j]+A[i][j+1]+A[i+1][j+1];
        }
    }
}


void gauss_seidel(double A[N][N], double b[N], double x[N], int thread_count) {
  int i, j;
    #pragma omp parallel for num_threads(thread_count) default(none) private(i,j) shared(A, b , x)
    for ( i = 0; i < N; i++) {
        double sum = b[i];
        for ( j = 0; j < N; j++) {
            if (j != i)
                sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }
}


void gauss_seidel_coarse(double A_coarse[N/2][N/2], double residual_coarse[N/2], double correction_coarse [N/2], int thread_count) {
    int i, j;
     #pragma omp parallel for num_threads(thread_count) default(none) private(i,j) shared(A_coarse , correction_coarse, residual_coarse)
    for ( i = 0; i < N/2; i++) {
        double sum = residual_coarse[i];
        for ( j = 0; j < N/2; j++) {
            if (j != i)
                sum -= A_coarse[i][j] * correction_coarse [j];
        }
        correction_coarse [i] = sum / A_coarse[i][i];
    }
}

void multigrid_solver(double A[N][N], double b[N], double x[N] , double A_coarse[N/2][N/2], double prolongation[N][N/2], double restriction[N/2][N], int thread_count) {
    double residual[N], correction[N], residual_coarse[N/2], correction_coarse[N/2];
    double prev_x[N];
    int i, j, k;

 
    gauss_seidel(A, b, x, thread_count);

    double error;
    do {
       #pragma omp parallel for num_threads(thread_count) default(none) private(i) shared(prev_x, x)
        for (i = 0; i < N; i++) {
            prev_x[i] = x[i];
        }

       #pragma omp parallel for num_threads(thread_count) default(none) private(i,j) shared(A, b, x, residual)
        for ( i = 0; i < N; i++) {
            double sum = b[i];
            for ( j = 0; j < N; j++) {
                sum -= A[i][j] * x[j];
            }
            residual[i] = sum;
        }

       
        for (int level = 1; level < MAX_LEVELS; level++) {
        #pragma omp parallel for num_threads(thread_count) default(none) private(i,k) shared(residual_coarse, correction_coarse, restriction, residual)   
            for ( i = 0; i < N/2; i++) {
                residual_coarse[i] = 0.0;
                for ( k=0; k<N; k++){
                  residual_coarse[i] += restriction[i][k]*residual[k];
                }
                correction_coarse[i] = 0.0;
            }
            
         gauss_seidel_coarse(A_coarse, residual_coarse, correction_coarse, thread_count);

           #pragma omp parallel for num_threads(thread_count) default(none) private(i,k) shared(correction, prolongation, correction_coarse)
            for ( i = 0; i < N; i++) {
                correction[i] = 0.0;
                for( k=0; k<N/2; k++){
                    correction[i]= prolongation[i][k]*correction_coarse[k];
                }
            }
            #pragma omp parallel for num_threads(thread_count) default(none) private(i) shared(x,correction)
            for ( i = 0; i < N; i++) {
                x[i] += correction[i];
            }
            

            
            gauss_seidel(A, b, x, thread_count);

           #pragma omp parallel for num_threads(thread_count) default(none) private(i,j) shared(A, b, x, residual)
            for (i = 0; i < N; i++) {
                double sum = b[i];
                for ( j = 0; j < N; j++) {
                    sum -= A[i][j] * x[j];
                }
                residual[i] = sum;
            }
        }

        error = 0.0;
        for (int i = 0; i < N; i++) {
            error += fabs(x[i] - prev_x[i]);
        }
        // printf("Final x vector:\n");
        // for (int i = 0; i < N; i++) {
        //     printf("%.6f , ", x[i]);
        // }printf("\n");

    } while (error > TOLERANCE);
}

int main(int argc, char* argv[]) {
    double A[N][N];
    double b[N];
    double x[N] = {0};  
    double restriction[N/2][N];
    double prolongation [N][N/2];
    double A_coarse[N/2][N/2];

    int thread_count = 1;
    double start=omp_get_wtime();
    if (argc == 2)
        {
        thread_count = strtol(argv[1], NULL, 10);
        }
    else
        {
        printf("\n A command line argument other than name of the executable is required...exiting the program..");
        return 1;
        }
    
    initialize_system(A, b, prolongation, restriction, A_coarse);

    
    multigrid_solver(A, b, x, A_coarse , prolongation, restriction, thread_count);

    printf("Final x vector:\n");
    for (int i = 0; i < N; i++) {
        printf("%.6f , ", x[i]);
    }

  return 0;
}