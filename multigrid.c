#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define N 100
#define MAX_LEVELS 10
#define TOLERANCE 0.0001

double func(double x)
{
    double q= sin(5*x);
    return q;
}

void initialize_system(double A[N][N], double b[N] , double prolongation[N][N/2], double restriction[N/2][N],  double A_coarse[N/2][N/2]) {
    
    // double delta = (double)3/N;
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

    //              else                                //<-------This code snippet corresponds to the initialisation of the A matrix wrt
    //              A[i][j] = 1.0;                      //  the pade'sscheme initialised with 1's , 2's and 0's . The b matrix is initialised to the pade schem
    //         }                                        //  conditions. Hence the solution of this system of linear equation via the multigrid must give us the 
    //         else                                     // differentiation function.

    //             A[i][j] = 0.0;
    //     }
    // }
    
    // double f[N];
    // for(int i=0;i<N;i++)
    // {
    //     f[i]=func(i*delta);                           
    //     //printf("%lf , ",f[i]);
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
            else if (abs(i - j) == 1)       //<----- this code snippet corresponds to initialising the A matrix as a simple tridiagonal matrix of 
                A[i][j] = -1.0;               //      superdiagonal diagonal =2 and other two subdiagonals =-1
            else                               //      also b is initialised to 1 for all i's
           A[i][j] = 0.0;
        }
    }

    for (int i = 0; i < N; i++) {
        b[i] = 1.0;
    }

     for (int i=0; i<N; i++){
                                           // initialising the prolongation matrix 
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
        for (int j=0; j<N/2; j++){        // initialising the restriction matrix which comes as the transpose of the restriction matrix

           restriction[j][i]=prolongation[i][j];

        }
    }
    double M[N/2][N];
    for (int i=0; i<N/2; i++){              
        for (int j=0; j<N; j++){
            for (int k=0; k<N; k++) {
         
          M[i][j] += restriction[i][k]*A[k][j];

            }                                    // initialising A_course by the formula A_coarse= R*A*P
        }
    }

    for(int i=0; i<N/2; i++){
        for(int j=0; j<N/2; j++){
          for( int k=0; k<N; k++ ){
          A_coarse[i][j]+= M[i][k]*prolongation[k][j];
          }
        }
    }

}


void gauss_seidel(double A[N][N], double b[N], double x[N]) {
    for (int i = 0; i < N; i++) {
        double sum = b[i];
        for (int j = 0; j < N; j++) {
            if (j != i)                          // the gauss seidel algorithm on A, b and x matrix
                sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }
}


void gauss_seidel_coarse(double A_coarse[N/2][N/2], double residual_coarse[N/2], double correction_coarse [N/2]) {
    for (int i = 0; i < N/2; i++) {
        double sum = residual_coarse[i];
        for (int j = 0; j < N/2; j++) {         // the gauss seidel algorithm on A_coarse, residual_coarse , correction_coarse matrix
            if (j != i)
                sum -= A_coarse[i][j] * correction_coarse [j];
        }
        correction_coarse [i] = sum / A_coarse[i][i];
    }
}

void multigrid_solver(double A[N][N], double b[N], double x[N] , double A_coarse[N/2][N/2], double prolongation[N][N/2], double restriction[N/2][N]) {
    double residual[N], correction[N], residual_coarse[N/2], correction_coarse[N/2];
    double prev_x[N];

 
    gauss_seidel(A, b, x);  // initial smoothing

    double error;
    do {
       
        for (int i = 0; i < N; i++) {
            prev_x[i] = x[i];                  // storing the previous x value
        }

       
        for (int i = 0; i < N; i++) {
            double sum = b[i];
            for (int j = 0; j < N; j++) {       // iterating the residual 
                sum -= A[i][j] * x[j];
            }
            residual[i] = sum;
        }

       
        for (int level = 1; level < MAX_LEVELS; level++) {
            for (int i = 0; i < N/2; i++) {
                residual_coarse[i] = 0.0;
                for (int k=0; k<N; k++){            //  iterating residual coarse
                  residual_coarse[i] += restriction[i][k]*residual[k];
                }
                correction_coarse[i] = 0.0;
            }

        
         gauss_seidel_coarse(A_coarse, residual_coarse, correction_coarse);  // gauss seidel on coarse grids

           
            for (int i = 0; i < N; i++) {
                correction[i] = 0.0;
                for(int k=0; k<N/2; k++){           // iterating the correction  after performing gauss seidel on coarse grids
                    correction[i]+= prolongation[i][k]*correction_coarse[k];
                }
            }

            for (int i = 0; i < N; i++) {
                x[i] += correction[i];        // iterating new x
            }

            
            gauss_seidel(A, b, x);       // 

           
            for (int i = 0; i < N; i++) {
                double sum = b[i];
                for (int j = 0; j < N; j++) {  // 
                    sum -= A[i][j] * x[j];
                }
                residual[i] = sum;
            }
        }

        error = 0.0;
        for (int i = 0; i < N; i++) {
            error += fabs(x[i] - prev_x[i]);
        }

    } while (error > TOLERANCE);
}

int main() {
    double A[N][N];
    double b[N];
    double x[N] = {0};  
    double restriction[N/2][N];
    double prolongation [N][N/2];
    double A_coarse[N/2][N/2];

    
    initialize_system(A, b, prolongation, restriction, A_coarse);

    
    multigrid_solver(A, b, x, A_coarse , prolongation, restriction);

    printf("Final x vector:\n");
    for (int i = 0; i < N; i++) {
        printf("%.6f , ", x[i]);
    }

    return 0;
}