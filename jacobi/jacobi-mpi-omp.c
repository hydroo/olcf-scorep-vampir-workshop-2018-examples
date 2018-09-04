/*
* Copyright 2012 NVIDIA Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef USE_MPI
#include <mpi.h>
#endif //USE_MPI
#include <omp.h>

int  handle_command_line_arguments(int argc, char** argv);
int  init_mpi(int argc, char** argv);
void init_host();
void init_cuda();

void finalize_mpi();
void finalize_host();
void finalize_cuda();

void start_timer();
void stop_timer();

void jacobi();

int n, m;
int n_global;

int rank=0;
int size=1;

int iter = 0;
int iter_max = 1000;

double starttime;
double runtime;

const float pi = 3.1415926535897932384626f;
const float tol = 1.0e-5f;
float residue = 1.0f;

float* A;
float* Anew;
float* y0_;

#ifdef USE_MPI
float* sendBuffer;
float* recvBuffer;
#endif //USE_MPI

/********************************/
/****         MAIN            ***/
/********************************/
int main(int argc, char** argv)
{
#ifdef USE_MPI
  if ( init_mpi(argc, argv) )
    {
      return 1;
    }
#endif //USE_MPI

  if ( handle_command_line_arguments(argc, argv) )
    {
      return -1;
    }

  init_host();

#ifdef USE_MPI
  /* This has do be done after handling command line arguments */
  sendBuffer = (float*) malloc ( (m-2) * sizeof(float) );
  recvBuffer = (float*) malloc ( (m-2) * sizeof(float) );

  MPI_Barrier(MPI_COMM_WORLD);
#endif //USE_MPI
    
  start_timer();

  // Main calculation
  jacobi();

  stop_timer();

  finalize_host();

#ifdef USE_MPI
  finalize_mpi();
#endif //USE_MPI
}

/********************************/
/****        JACOBI           ***/
/********************************/
void jacobi()
{
  while ( residue > tol && iter < iter_max )
    {
      residue = 0.0f;

#pragma omp parallel
      {
        float my_residue = 0.f;

#pragma omp for nowait
        for( int j = 1; j < n-1; j++)
          {
            for( int i = 1; i < m-1; i++ )
              {
                //Jacobi is Anew[j*m+i] = 1.0/1.0*(rhs[j*m+i] -
                //                        (                           -0.25f*A[(j-1) *m+ i]
                //                          -0.25f*A[j     *m+ (i+1)]                        -0.25f*A[j     *m+ (i-1)]
                //                                                    -0.25f*A[(j+1) *m+ i]));
                //rhs[j*m+i] == 0 for 0 <= j < n and 0 <= i < m
                // =>
                Anew[j *m+ i] = 0.25f * ( A[j     *m+ (i+1)] + A[j     *m+ (i-1)]
                                          +    A[(j-1) *m+ i]     + A[(j+1) *m+ i]);
                //Calculate residue of A
                //residue =
                //   rhs[j*m+i] -  (                           -0.25f*A[(j-1) *m+ i]
                //                   -0.25f*A[j     *m+ (i+1)] +1.00f*A[j     *m+ i]  -0.25f*A[j     *m+ (i-1)]
                //                                             -0.25f*A[(j+1) *m+ i]));
                //rhs[j*m+i] == 0 for 0 <= j < n and 0 <= i < m
                // =>
                //residue =  Anew[j *m+ i]-A[j *m + i]
                my_residue = fmaxf( my_residue, fabsf(Anew[j *m+ i]-A[j *m + i]));
              }
          }

#pragma omp critical
        {
          residue = fmaxf( my_residue, residue);
        }
      }

#ifdef USE_MPI
      float globalresidue = 0.0f;
      MPI_Allreduce( &residue, &globalresidue, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD );
      residue = globalresidue;
#endif //USE_MPI

#ifdef USE_MPI
      if ( size == 2 )
        {
          MPI_Status status;
          if ( rank == 0)
            {
              MPI_Sendrecv( Anew+(n-2)*m+1, m-2, MPI_FLOAT, 1, 0, A+(n-1)*m+1 , m-2, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &status );
            }
          else
            {
              MPI_Sendrecv( Anew + 1*m+1, m-2, MPI_FLOAT, 0, 0, A+0*m+1, m-2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status );
            }
        }
#endif //USE_MPI
		
#pragma omp parallel for
      for( int j = 0; j < n-1; j++)
        {
          for( int i = 1; i < m-1; i++ )
            {
              A[j *m+ i] = Anew[j *m+ i];
            }
        }

      if(rank == 0 && iter % 100 == 0)
        printf("%5d, %0.6f\n", iter, residue);

      iter++;
    }
}


/********************************/
/**** Initialization routines ***/
/********************************/

#ifdef USE_MPI
int init_mpi(int argc, char** argv)
{
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if ( size != 1 && size != 2 )
    {
      if ( rank == 0)
        printf("Error: %s can only run with 1 or 2 processes!\n",argv[0]);
      return 1;
    }

  return 0;
}
#endif //USE_MPI

void init_host()
{
  A	= (float*) malloc( n*m * sizeof(float) );
  Anew	= (float*) malloc( n*m * sizeof(float) );
  y0_	= (float*) malloc( n   * sizeof(float) );

#ifdef OMP_MEMLOCALTIY
#pragma omp parallel for shared(A,Anew,m,n)
  for( int j = 0; j < n; j++)
    {
      for( int i = 0; i < m; i++ )
        {
          Anew[j *m+ i] 	= 0.0f;
          A[j *m+ i] 		= 0.0f;
        }
    }
#endif //OMP_MEMLOCALTIY

  memset(A, 0, n * m * sizeof(float));
  memset(Anew, 0, n * m * sizeof(float));

  // set boundary conditions
#pragma omp parallel for
  for (int i = 0; i < m; i++)
    {
      //Set top boundary condition only for rank 0 (rank responsible of the upper halve of the domain)
      if ( rank == 0 )
        A[0	    *m+ i] = 0.f;
      //Set bottom boundary condition only for rank 1 (rank responsible of the lower halve of the domain)
      if ( rank == 0 || size == 1 )
        A[(n-1) *m+ i] = 0.f;
    }

  int j_offset = 0;
  if ( size == 2 && rank == 1 )
    {
      j_offset = n-2;
    }
  for (int j = 0; j < n; j++)
    {
      y0_[j] = sinf(pi * (j_offset + j) / (n-1));
      A[j *m+ 0] = y0_[j];
      A[j *m+ (m-1)] = y0_[j]*expf(-pi);
    }

#pragma omp parallel for
  for (int i = 1; i < m; i++)
    {
      if (rank == 0)
        Anew[0     *m+ i] = 0.f;
      if (rank == 1 || size == 1)
        Anew[(n-1) *m+ i] = 0.f;
    }
#pragma omp parallel for
  for (int j = 1; j < n; j++)
    {
      Anew[j *m+ 0] = y0_[j];
      Anew[j *m+ (m-1)] = y0_[j]*expf(-pi);
    }
}

int handle_command_line_arguments(int argc, char** argv)
{
  if ( argc > 3 )
    {
      if ( rank == 0)
        printf( "usage: %s [n] [m]\n", argv[0] );
      return -1;
    }

  n = 4096;
  if ( argc >= 2 )
    {
      n = atoi( argv[1] );
      if ( n <= 0 )
        {
          if ( rank == 0 )
            printf("Error: The number of rows (n=%i) needs to positive!\n",n);
          return 1;
        }
    }
  if ( size == 2 && n%2 != 0 )
    {
      if ( rank == 0)
        printf("Error: The number of rows (n=%i) needs to be devisible by 2 if two processes are used!\n",n);
      return 1;
    }
  m = n;
  if ( argc >= 3 )
    {
      m = atoi( argv[2] );
      if ( m <= 0 )
        {
          if ( rank == 0 )
            printf("Error: The number of columns (m=%i) needs to positive!\n",m);
          return 1;
        }
    }
  
  n_global = n;

  if ( size == 2 )
    {
      //Do a domain decomposition and add one row for halo cells
      n = n/2 + 1;
    }

  if ( rank == 0 )
    {
#pragma omp parallel
      {
#pragma omp master
        {
          printf("Jacobi relaxation Calculation: %d x %d mesh "
                 "with %d processes and %d threads.\n", 
                 n_global, m,size,omp_get_num_threads());
        }
      }
    }
  return 0;
}


/********************************/
/****  Finalization routines  ***/
/********************************/

#ifdef USE_MPI
void finalize_mpi()
{
  free( recvBuffer );
  free( sendBuffer );

  MPI_Finalize();
}
#endif //USE_MPI

void finalize_host()
{
  free(y0_);
  free(Anew);
  free(A);
}

/********************************/
/****    Timing functions     ***/
/********************************/
void start_timer()
{
#ifdef USE_MPI
  starttime = MPI_Wtime();
#else
  starttime = omp_get_wtime();
#endif //USE_MPI
}

void stop_timer()
{
#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  runtime = MPI_Wtime() - starttime;
#else
  runtime = omp_get_wtime() - starttime;
#endif //USE_MPI

  if (rank == 0)
    printf(" total: %f s\n", runtime);
}

