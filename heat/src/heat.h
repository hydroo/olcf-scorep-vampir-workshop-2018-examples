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
 *  Header for both OpenMP and serial version.
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <malloc.h>

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

void heatAllocate(heatGrid *grid, int xsize, int ysize);
void heatDeallocate(heatGrid *grid);
void heatInitialize(heatGrid* grid);
double heatInitFunc(double x);
void heatPrint(heatGrid* grid);
void heatTimestep(heatGrid* grid, double dt, double* dthetamax);
void heatBoundary(heatGrid* grid);
void heatTotalEnergy(heatGrid* grid, double *energy);
