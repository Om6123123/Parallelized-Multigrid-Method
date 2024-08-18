# Parallelized-Multigrid-Method
The multi-grid method is a robust iterative approach utilized for efficiently solving large sparse systems
of linear equations, particularly in the context of solving partial differential equations. By employing
a hierarchy of grids, it accelerates convergence by integrating coarse and fine grids and exploiting the
varying length scales inherent in the problem. The method’s fundamental strategy involves approximating the solution on a coarse grid, then refining it on finer grids, and further interpolating back to
coarser grids to enhance the solution. This iterative process comprises several key steps, starting with
an initial guess and proceeding through pre-smoothing to reduce high-frequency error components,
restricting the residual to a coarser grid, solving the coarse grid problem, interpolating corrections to
finer grids, updating the solution with interpolated corrections, and post-smoothing to further refine
error components.

This paper explores parallelisation of the Multi-grid Method using OpenMP , OpenMPI , Open
ACC , CUDA and SYCL methods to meet the demand for faster computations in scientific and engineering domains.
Leveraging shared-memory parallelism with OpenMP ,distributed-memory parallelism with OpenMP
and GPU-CPU based distribution with OpenACC, we achieve notable speedup and scalability improvements. We evaluate the efficiency of our implementations, demonstrating their effectiveness in
addressing large-scale problems. Our findings provide valuable insights for optimizing complex numerical algorithms on modern parallel architectures

## Authors
- Om Raul - IIT Madras(Main contributor)
- Rudra Panch  IIT Madras
- Rachit Kumar - IIT Madras

## Parallelsim Strategy

As the name suggests , the V scheme is a multi-grid algorithm which involves iterating on the present
matrix in a V pattern in a up down format. It operates on a hierarchy of grids, moving between coarser
and finer levels to efficiently correct errors.
The entire notion of the Multi-grid notion is to achieve a faster convergence.Methods like Jacobi
iterative method, or Gauss-Seidel method tend to slow down as the frequency of error decreases.
Through interpolating finer matrices into coarser ones , we increase the frequency of errors and hence
speedup the convergence
1
The V-scheme would be involving the following steps:
- Firstly a pre-smoothing would be performed using gauss-seidel method.The main aim of the presmoothing operation in the V-cycle of the multi-grid method is to reduce high-frequency errors in
the solution on the finest grid level before proceeding to coarser levels for further correction. Presmoothing prepares the solution for the multi-grid correction process by damping out oscillations
and rapidly reducing local errors.
- Then the residual matrix and the correction matrix would be calculated. The residual matrix (r)
is nothing but the difference between the actual ’b’ matrix and the ’b’ matrix we get by using
the current ’x’. It is basically the error in ’b’ matrix.
- On the other hand the correction matrix (e) is the error in the ’x’ matrix. This correction matrix
is what we subject to the restriction process.
- Further the matrices A,e and r would be restricted into coarser matrices using the Restriction
matrix (R).
- iteration of gauss-seidel would be performed on these coarser matrices.
- further the matrices would be interpolated back into finer matrices using the Prolongation matrix
(I).
- This loop continues until the errors fall under a certain level of tolerance.
