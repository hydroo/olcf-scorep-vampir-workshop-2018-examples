/*
 *   Heat conduction demo program
 *
 *  solves the heat equation on a 2D grid
 *
 *  March 2009
 *  Matthias.Lieber@tu-dresden.de
 *  Tobias.Hilbrich@zih.tu-dresden.de
 *
 *  Adapted: Jan 2013
 *
 *  Header for MPI version.
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <malloc.h>
#include <mpi.h>

/* This type represents the grid and its description */
typedef struct
{
    /* current theta array */
    double **theta;
    /* new theta array */
    double **thetanew;
    /* domain size (number of grid cells) in x and y */
    int xsize;
    int ysize;
    /* size of a grid cell */
    double dx;
    double dy;
    /* "heat equation constant" */
    double k;
} heatGrid;

/* This type defines everything thats needed for MPI */
typedef struct {
    /* Own rank, used to only let master do output*/
    int rank;
    /* Comm for a cartesian distribution of the grid*/
    MPI_Comm cart;
    /* Neighbors in communicator*/
    int up,down,left,right;
    /* Start of a processes sub-grid (x, y)*/
    int start_x;
    int start_y;
    /* Number of cells in x or y direction for this process*/
    int num_cells_x;
    int num_cells_y;
    /* Datatype used to transfer a data column*/
    MPI_Datatype rowtype;
} dataMPI;

void heatAllocate(heatGrid *grid, int xsize, int ysize);
void heatDeallocate(heatGrid *grid);
void heatInitialize(heatGrid* grid);
double heatInitFunc(double x);
void heatPrint(heatGrid* grid);
void heatTimestep(heatGrid* grid, dataMPI* mympi, double dt, double* dthetamax);
void heatBoundary(heatGrid* grid, dataMPI* mympi);
void heatTotalEnergy(heatGrid* grid, double *energy);
void heatMPISetup (heatGrid *grid, dataMPI* configMPI);
void heatMPIFree (dataMPI* configMPI);
void heatMPIGather (heatGrid *grid, dataMPI* configMPI);
