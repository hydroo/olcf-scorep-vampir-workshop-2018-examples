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
 *  Version with OpenMP.
 */

#include "heat.h"
#include <omp.h>

/******************************************************
 * Allocate the heatGrid and initialize all variables.
 ******************************************************/
void heatAllocate(heatGrid *grid, int xsize, int ysize)
{
    int i,j;

    grid->xsize = xsize;
    grid->ysize = ysize;
    grid->theta    = (double**) malloc (sizeof(double*)*(xsize+2));
    grid->thetanew = (double**) malloc (sizeof(double*)*(xsize+2));
    grid->theta   [0] = (double*) malloc (sizeof(double)*(xsize+2)*(ysize+2));
    grid->thetanew[0] = (double*) malloc (sizeof(double)*(xsize+2)*(ysize+2));

    for (i = 0; i < xsize+2; i++)
    {
        grid->theta   [i] = grid->theta   [0]+i*(ysize+2);
        grid->thetanew[i] = grid->thetanew[0]+i*(ysize+2);

        for (j = 0; j < ysize+2; j++)
        {
            grid->theta   [i][j] = 0.0;
            grid->thetanew[i][j] = 0.0;
        }
    }

    grid->dx = 1.0;
    grid->dy = 1.0;
    grid->k = 1.0;
}

/******************************************************
 * Deallocate the heatGrid.
 ******************************************************/
void heatDeallocate(heatGrid *grid)
{
    free (grid->theta[0]);
    free (grid->theta);
    grid->theta = NULL;
    free (grid->thetanew[0]);
    free (grid->thetanew);
    grid->thetanew = NULL;
}

/******************************************************
 * Initialize the grid with some meaninful start values.
 ******************************************************/
void heatInitialize(heatGrid* grid)
{
    double xcenter, ycenter, radius, cr;
    int x, y, i , j;

    /* initialize with a circle */
    for (i = 0; i < grid->xsize+2; i++)
        for (j = 0; j < grid->ysize+2; j++)
            grid->theta[i][j] = 0.0;

    xcenter = (grid->xsize - 10) * grid->dx / 2.0;
    ycenter = (grid->ysize + 6)  * grid->dy / 2.0;
    radius = fmin(grid->xsize * grid->dx, grid->ysize * grid->dy) * 0.25;

    for (x=1; x <= grid->xsize; x++)
    {
        for (y=1; y <= grid->ysize; y++)
        {
            cr = sqrt( (x-xcenter)*(x-xcenter) + (y-ycenter)*(y-ycenter) );
            if ( cr < radius )
            {
                /* sharp boundary */
                /*grid->theta[x][y] = grid->theta[x][y] + 2.0;*/

                /* smooth boundary */
                grid->theta[x][y] = grid->theta[x][y] + 2.0 * heatInitFunc(cr/radius);
            }
        }
    }
}

/******************************************************
 * Polynomial function for the initialization a smooth heat profile, 0 <= x <= 1.
 ******************************************************/
double heatInitFunc(double x)
{
    return (-4.0*x*x*x + 4.0*x*x - 1.0*x + 1.0);
}

/******************************************************
 * Print grid to console.
 ******************************************************/
void heatPrint(heatGrid* grid)
{
    int x, y;

    /* print header */
    for (x=1; x <= grid->xsize; x++)
        printf ("====");
    printf("\n");

    /* print data */
    for (y=1; y <= grid->ysize; y++)
    {
        for (x=1; x <= grid->xsize; x++)
        {
            if(grid->theta[x][y] < 1.e-100)
                printf ("  . ");
            else
                printf("%4.1f", grid->theta[x][y]);
        }
        printf("\n");
    }
}

/******************************************************
 * Calculate one timestep of size dt on the grid.
 *
 * for each grid point:
 *
 *                                  ( d2T   d2T )
 *          thetanew =  theta + k * ( --- + --- ) * dt
 *                                  ( dx2   dy2 )
 *                                  _____________
 *                                        |
 *                                        |
 *  dthetamax returns the max. value of this term
 *  (useful as exit condition for the time stepping)
 ******************************************************/
void heatTimestep(heatGrid* grid, double dt, double* dthetamax)
{
    int x, y;
    double dtheta;

    *dthetamax = 0.0;

    /* calculate the time step: read from theta, write new timestep to thetanew */
#pragma omp parallel  private(dtheta, x, y)
    {
        double mymax = 0.0; /* For OpenMP (no max reduction) */
#pragma omp for
        for (x=1; x <= grid->xsize;x++)
        {
            for (y=1; y <= grid->ysize; y++)
            {
                dtheta = ( grid->theta[x-1][y] + grid->theta[x+1][y] - 2*grid->theta[x][y]) / (grid->dx * grid->dx)
                       + ( grid->theta[x][y-1] + grid->theta[x][y+1] - 2*grid->theta[x][y]) / (grid->dy * grid->dy);
                grid->thetanew[x][y] = grid->theta[x][y] + grid->k * dtheta * dt;

                mymax = fmax(fabs(dtheta), mymax); /* save max theta for the exit condition */
            }
        }

#pragma omp critical
        if (mymax > *dthetamax)
            *dthetamax = mymax;
    }

    /* update theta: copy thetanew to theta */
#pragma omp parallel for  private(x, y)
    for (x=1; x <= grid->xsize; x++)
        for (y=1; y <= grid->ysize; y++)
             grid->theta[x][y] = grid->thetanew[x][y];
}

/******************************************************
 * Set periodic boundary conditions.
 *
 * The grid arrays are allocated with additional "ghost cells"
 * in each spatial dimension. The lower boundary is copied to
 * the upper ghost cells (and vice versa) for each dimension:
 *
 *    ___________         ___________
 *   |  _______ |        |  stuvwx  |
 *   | |abcdef| |        |f|abcdef|a|
 *   | |ghijkl| |        |l|ghijkl|g|
 *   | |mnopqr| |   ->   |r|mnopqr|m|
 *   | |stuvwx| |        |x|stuvwx|s|
 *   |__________|        |__abcdef__|
 *
 ******************************************************/
void heatBoundary(heatGrid* grid)
{
    int i;

    /* y direction */
    for (i = 1; i <= grid->xsize; i++)
    {
        /* y-bottom */
        grid->theta[i][0] = grid->theta[i][grid->ysize];
        /* y-top */
        grid->theta[i][grid->ysize+1] = grid->theta[i][1];
    }

    /* x direction */
    for (i = 1; i <= grid->ysize; i++)
    {
        /* x-left */
        grid->theta[0][i] = grid->theta[grid->xsize][i];
        /* x-right */
        grid->theta[grid->xsize+1][i] = grid->theta[1][i];
    }
}


/******************************************************
 * Calculate the total energy (sum of theta in all grid cells)
 ******************************************************/
void heatTotalEnergy(heatGrid* grid, double *energy)
{
  int x, y;

  *energy = 0.0;
  for (x=1; x <= grid->xsize; x++)
      for (y=1; y <= grid->ysize; y++)
          *energy += grid->theta[x][y];
}

/******************************************************
 * Main program and time stepping loop.
 ******************************************************/
int main (int argc, char** argv)
{
    heatGrid mygrid;
    double dt, dthetamax, energyInitial, energyFinal, t0, t1;
    int step, nsteps=20;

    /* create heatGrid and initialize variables */
    heatAllocate(&mygrid, 8192, 8192);
    heatInitialize(&mygrid);
    dt = 0.05;
    dthetamax = 100.0;

    /* output of initial grid */
    // printf("initial grid:\n");
    // heatPrint(&mygrid);

    t0 = omp_get_wtime();

    /* energy of initial grid */
    heatTotalEnergy(&mygrid, &energyInitial);

    /* time stepping loop */
    for( step=1 ; step<=nsteps ; step++)
    {
        heatBoundary(&mygrid);
        heatTimestep(&mygrid, dt, &dthetamax );
    }

    t1 = omp_get_wtime();

    /* output of final grid */
    // printf ("\ngrid after %d iterations:\n",nsteps);
    // heatPrint(&mygrid);

    /* energy of final grid */
    heatTotalEnergy(&mygrid, &energyFinal);

    printf("\n= Energy Conservation Check =\n");
    printf(" initial Energy: %20.3f\n",energyInitial);
    printf("   final Energy: %20.3f\n",energyFinal);
    printf("     Difference: %20.3f\n",energyFinal-energyInitial);
    printf("           Time: %20.3f s\n",t1-t0);

    heatDeallocate(&mygrid);

    return 0;
}

/*EOF*/
