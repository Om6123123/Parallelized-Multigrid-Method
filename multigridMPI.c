#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
int N=100;

// #define N 80
// #define MAX_LEVELS 10
// #define TOLERANCE 0.0001

double func(double x)
{
    double q= sin(5*x);
    return q;
}

void initialize_system(double A[N][N], double b[N], double prolongation[N][N / 2], double restriction[N / 2][N], double A_coarse[N / 2][N / 2], int rank, int size) {

    // double delta = (double) 3/N;
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         if (i == j){
    //             if (i==0 || i==(N-1))
    //             A[i][j] = 1.0;

    //             else 
    //             A[i][j] = 4.0;
    //         }
    //         else if (abs(i - j) == 1){
    //              if (i==0 || i==(N-1))
    //               A[i][j] = 2.0;

    //              else 
    //              A[i][j] = 1.0;

    //         }
    //         else

    //             A[i][j] = 0.0;
    //     }
    // }
    
    // double f[N];
    // for(int i=0;i<N;i++)
    // {
    //     f[i]=func(i*delta);
    // }

    // for (int i = 1; i < N-1; i++) {
    //     b[i] = 3*(f[i+1]-f[i-1])/delta;
    // }

    // b[N-1]=(2.5*f[N-1]-2*f[N-2]-0.5*f[N-3])/delta;
    // b[0]=(-2.5*f[0]+2*f[1]+0.5*f[2])/delta;

    for (int i = 0; i < N; i++) {
     for (int j = 0; j < N; j++) {
            if (i == j)
                A[i][j] = 2.0;
            else if (abs(i - j) == 1)
                A[i][j] = -1.0;
            else
           A[i][j] = 0.0;
        }
    }

    for (int i = 0; i < N; i++) {
        b[i] = 1.0;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N / 2; j++) {
          if ((i % 2) != 0 && j == (i - 1) / 2)
                prolongation[i][j] = 1;
         if ((i % 2) == 0 && j == (i / 2))
                prolongation[i][j] = 1;
            else
                prolongation[i][j] = 0;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N / 2; j++) {
            restriction[j][i] = prolongation[i][j];
        }
    }

    double M[N / 2][N];
    for (int i = 0; i < N / 2; i++) {
     for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
                M[i][j] += restriction[i][k] * A[k][j];
            }
        }
    }

    for (int i = 0; i < N / 2; i++) {
      for (int j = 0; j < N / 2; j++) {
        for (int k = 0; k < N; k++) {
                A_coarse[i][j] += M[i][k] * prolongation[k][j];
            }
        }
    }
    
    // for(int i=0; i< size; i++){
    // MPI_Bcast( A, N*N, MPI_DOUBLE, A, i, MPI_COMM_WORLD);
    // MPI_Bcast( prolongation, N*N/2, MPI_DOUBLE, prolongation, i, MPI_COMM_WORLD);
    // MPI_Bcast( restriction , N*N/2, MPI_DOUBLE, prolongation, i, MPI_COMM_WORLD);
    // }
}

void gauss_seidel(double A[N][N], double b[N], double x[N], int rank, int size) {
    int start = rank * (N / size);
    int end = (rank + 1) * (N / size);
     MPI_Status status;
    
    double x_local[N/size];

    for (int i = start; i < end; i++) {
        double sum = b[i];
        for (int j = 0; j < N; j++) {
        if (j != i)
                sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
        x_local[i-start]= x[i];
    }

    for (int i=0; i< size ; i++)
    {
     if (i!=rank){

            MPI_Send(x_local ,(N/size), MPI_DOUBLE , i, 1, MPI_COMM_WORLD);
         
     }
     else{
        for (int k=0; k< size ; k++){
         if (k!=i){
         MPI_Recv(x + (k)*(N/size) ,(N/size), MPI_DOUBLE , k , 1, MPI_COMM_WORLD ,&status);
         
         }
        }
     } 
    }
}

void gauss_seidel_coarse(double A_coarse[N / 2][N / 2], double residual_coarse[N / 2], double correction_coarse[N / 2], int rank, int size) {
    int start = rank * (N /(size * 2));
    int end = (rank + 1) * (N /(size * 2));
     MPI_Status status;

    double correction_coarse_local[N/(size*2)];

    for (int i = start; i < end; i++) {
        double sum = residual_coarse[i];
        for (int j = 0; j < N / 2; j++) {
        if (j != i)
                sum -= A_coarse[i][j] * correction_coarse[j];
        }
        correction_coarse[i] = sum / A_coarse[i][i];
        correction_coarse_local[i-start]=correction_coarse[i];
    }
     
    for (int i=0; i< size ; i++)
    {
     if (i!=rank){

            MPI_Send(correction_coarse_local ,(N/(2*size)), MPI_DOUBLE , i, 1, MPI_COMM_WORLD);
         
     }
     else{
        for (int k=0; k< size ; k++){
         if (k!=i){
         MPI_Recv(correction_coarse + (k)*(N/(2*size)) ,(N/(2*size)), MPI_DOUBLE , k , 1, MPI_COMM_WORLD ,&status);
         
         }
        }
     } 
    }
}

void multigrid_solver(double A[N][N], double b[N], double x[N], double A_coarse[N / 2][N / 2], double prolongation[N][N / 2], double restriction[N / 2][N], int rank, int size, double TOLERANCE , int MAX_LEVELS) {
    double residual[N], correction[N], residual_coarse[N / 2], correction_coarse[N / 2];
    double prev_x[N];
    double prev_x_local[N/size];
    double residual_local[N/size];
    double residual_coarse_local[N/(2*size)];
    double correction_coarse_local[N/(2*size)];
    double correction_local[N/(size)];
    double x_local[N/(size)];
    MPI_Status status;

    int start1 = rank * (N / size);
    int end1 = (rank + 1) * (N / size);

    int start2 = rank * (N /( size * 2));
    int end2 = (rank + 1) * (N /( size * 2));

    gauss_seidel(A, b, x, rank, size);

    double error, total_error = 0.0;
    do {

        for (int i = start1; i < end1; i++) {
            prev_x[i] = x[i];
            prev_x_local[i-start1]= prev_x[i];
        }
    
    
    for (int i=0; i< size ; i++)
    {
     if (i!=rank){

            MPI_Send(prev_x_local ,(N/size), MPI_DOUBLE , i, 1, MPI_COMM_WORLD);
         
     }
     else{
        for (int k=0; k< size ; k++){
         if (k!=i){
         MPI_Recv(prev_x + (k)*(N/size) ,(N/size), MPI_DOUBLE , k , 1, MPI_COMM_WORLD ,&status);
         
         }
        }
     } 
    }

        for (int i = start1; i < end1; i++) {
            double sum = b[i];
            for (int j = 0; j < N; j++) {
                sum -= A[i][j] * x[j];
            }
            residual[i] = sum;
            residual_local[i-start1]= residual[i];
        }
   
    for (int i=0; i< size ; i++)
    {
     if (i!=rank){

            MPI_Send(residual_local ,(N/size), MPI_DOUBLE , i, 1, MPI_COMM_WORLD);
         
     }
     else{
        for (int k=0; k< size ; k++){
         if (k!=i){
         MPI_Recv(residual + (k)*(N/size) ,(N/size), MPI_DOUBLE , k , 1, MPI_COMM_WORLD ,&status);
         
         }
        }
     } 
    }

        for (int level = 1; level < MAX_LEVELS; level++) {
            for (int i = start2; i < end2; i++) {
                residual_coarse[i] = 0.0;
                for (int k = 0; k < N; k++) {
                    residual_coarse[i] += restriction[i][k] * residual[k];
                    
                }
                residual_coarse_local[i-start2]=residual_coarse[i];
                correction_coarse[i] = 0.0;
                correction_coarse_local[i-start2]=correction_coarse[i];
            }

    
    for (int i=0; i< size ; i++)
    {
     if (i!=rank){

            MPI_Send(correction_coarse_local ,(N/(2*size)), MPI_DOUBLE , i, 1, MPI_COMM_WORLD);
         
     }
     else{
        for (int k=0; k< size ; k++){
         if (k!=i){
         MPI_Recv(correction_coarse + (k)*(N/(2*size)) ,(N/(2*size)), MPI_DOUBLE , k , 1, MPI_COMM_WORLD ,&status);
         
         }
        }
     } 
    }   

            gauss_seidel_coarse(A_coarse, residual_coarse, correction_coarse, rank, size);

            for (int i = start1; i < end1; i++) {
                correction[i] = 0.0;
                for (int k = 0; k < N / 2; k++) {
                    correction[i] += prolongation[i][k] * correction_coarse[k];
                    correction_local[i-start1]=correction[i];
                }
            }

    
    for (int i=0; i< size ; i++)
    {
     if (i!=rank){

            MPI_Send(correction_local ,(N/size), MPI_DOUBLE , i, 1, MPI_COMM_WORLD);
         
     }
     else{
        for (int k=0; k< size ; k++){
         if (k!=i){
         MPI_Recv(correction + (k)*(N/size) ,(N/size), MPI_DOUBLE , k , 1, MPI_COMM_WORLD ,&status);
         
         }
        }
     } 
    }    

            for (int i = start1; i < end1; i++) {
                x[i] += correction[i];
                x_local[i-start1]=x[i];
            }
    
    
    for (int i=0; i< size ; i++)
    {
     if (i!=rank){

            MPI_Send(x_local ,(N/size), MPI_DOUBLE , i, 1, MPI_COMM_WORLD);
         
     }
     else{
        for (int k=0; k< size ; k++){
         if (k!=i){
         MPI_Recv(x + (k)*(N/size) ,(N/size), MPI_DOUBLE , k , 1, MPI_COMM_WORLD ,&status);
         
         }
        }
     } 
    }

            gauss_seidel(A, b, x, rank, size);

            for (int i = start1; i < end1; i++) {
                double sum = b[i];
                for (int j = 0; j < N; j++) {
                    sum -= A[i][j] * x[j];
                }
                residual[i] = sum;
                residual_local[i-start1]=residual[i];
            }
        }
    
    
    for (int i=0; i< size ; i++)
    {
     if (i!=rank){

            MPI_Send(residual_local ,(N/size), MPI_DOUBLE , i, 1, MPI_COMM_WORLD);
         
     }
     else{
        for (int k=0; k< size ; k++){
         if (k!=i){
         MPI_Recv(residual + (k)*(N/size) ,(N/size), MPI_DOUBLE , k , 1, MPI_COMM_WORLD ,&status);
         
         }
        }
     } 
    }

        error = 0.0; total_error=0;
        for (int i = start1; i < end1; i++) {
            error += fabs(x[i] - prev_x[i]);
        }
        MPI_Allreduce(&error, &total_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // printf("%f %f\n", error, total_error);

    } while (total_error > TOLERANCE);
}

int main(int argc, char *argv[]) {
    int rank, size;
   
   
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    

    double A[N][N];
    double b[N];
    double x[N];
    double restriction[N / 2][N];
    double prolongation[N][N / 2];
    double A_coarse[N / 2][N / 2];
    int MAX_LEVELS=10;
    double TOLERANCE= 0.0001;
    for(int i=0; i<N; i++){
        x[i]=0;
    }
    
    //if(rank==0){
    initialize_system(A, b, prolongation, restriction, A_coarse, rank, size);
    //}

    multigrid_solver(A, b, x, A_coarse, prolongation, restriction, rank, size, TOLERANCE, MAX_LEVELS );

    if (rank==0){
      printf("Final x vector:\n");
       for(int i=0; i<N; i++)
       {
        printf("%.6f  ," , x[i]);
       }
    }
    MPI_Finalize();

  return 0;
}