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
 *  Hybrid MPI/OpenMP version.
 */

#include "heat-mpi.h"
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
void heatTimestep(heatGrid* grid, dataMPI* mympi, double dt, double* dthetamax)
{
    int x, y;
    double dtheta;
    double mymax = 0.0;

    *dthetamax = 0.0;

    /* calculate the time step: read from theta, write new timestep to thetanew */
    /* Only calculate on a processes sub-grid */
#pragma omp parallel  private(dtheta, x, y)
    for (x=mympi->start_x; x < mympi->start_x + mympi->num_cells_x;x++)
    {
        double mymax = 0.0; /* For OpenMP (no max reduction) */
#pragma omp for
        for (y=mympi->start_y; y < mympi->start_y + mympi->num_cells_y; y++)
        {
            dtheta = ( grid->theta[x-1][y] + grid->theta[x+1][y] - 2*grid->theta[x][y]) / (grid->dx * grid->dx)
                   + ( grid->theta[x][y-1] + grid->theta[x][y+1] - 2*grid->theta[x][y]) / (grid->dy * grid->dy);
            grid->thetanew[x][y] = grid->theta[x][y] + grid->k * dtheta * dt;

            mymax = fmax(fabs(dtheta), mymax); /* save max theta for the exit condition */
        }

#pragma omp critical
        if (mymax > *dthetamax)
            *dthetamax = mymax;
    }

    /* Make MPI reduction to get maximum dtheta of all processes */
    MPI_Allreduce (&mymax, dthetamax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    /* update theta: copy thetanew to theta */
#pragma omp parallel for  private(x, y)
    for (x=mympi->start_x; x < mympi->start_x + mympi->num_cells_x;x++)
        for (y=mympi->start_y; y < mympi->start_y + mympi->num_cells_y; y++)
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
 * For MPI:
 * ========
 *
 * Each process has ghost cells around its sub-grid
 * e.g., Sub-grid & ghost cells for process "P_i":
 *
 *      __________________
 *     |Overall-grid      |
 *     |       ...        |
 *     |    __________    |
 *     |   |G ______ c|   |
 *     |   |h| P_i  |e|   |
 *     |   |o| Sub- |l|   |
 *     |...|s| Grid |l|...|
 *     |   |t|______|s|   |
 *     |   |__________|   |
 *     |       ...        |
 *     |__________________|
 *
 * Ghost cells are received from the neighbors.
 * Neighbors in turn receive border of the process.
 *
 * e.g., Border exchange for Process 0:
 *         _____
 *        |     |
 *        |  4  |
 *        |     |
 *        |FGHIJ|
 *  _____ ======= _____
 * |    v||abcde||q    |
 * |    w||f   g||r    |
 * |  2 x||h 0 i||s 1  |
 * |    y||j   k||t    |
 * |____z||lmnop||u____|
 *        =======
 *        |ABCDE|
 *        |     |
 *        |  3  |
 *        |_____|
 *
 * 0 sends (e,g,i,k,p) to 1
 * 0 receives (q,r,s,t,u) from 1 (put into its ghost cells)
 * 0 sends (a,f,h,j,l) to 2
 * 0 receives (v,w,x,y,z) from 2 (put into its ghost cells)
 * ....
 *
 ******************************************************/
void heatBoundary(heatGrid* grid, dataMPI* mympi)
{
    MPI_Status status;

    /*Send left column to left neighbor*/
    MPI_Bsend (&(grid->theta[mympi->start_x][mympi->start_y]),
            mympi->num_cells_y, MPI_DOUBLE, mympi->left, 123, mympi->cart);
    /*Receive Right border column from right neighbor*/
    MPI_Recv  (&(grid->theta[mympi->start_x+mympi->num_cells_x][mympi->start_y]),
            mympi->num_cells_y, MPI_DOUBLE, mympi->right, 123, mympi->cart, &status);

    /*Send right column to right neighbor*/
    MPI_Bsend (&(grid->theta[mympi->start_x+mympi->num_cells_x-1][mympi->start_y]),
            mympi->num_cells_y, MPI_DOUBLE, mympi->right, 123, mympi->cart);
    /*Receive Left border column from left neighbor*/
    MPI_Recv  (&(grid->theta[mympi->start_x-1][mympi->start_y]),
            mympi->num_cells_y, MPI_DOUBLE, mympi->left, 123, mympi->cart, &status);

    /*Send upper row to top neighbor*/
    MPI_Bsend (&(grid->theta[mympi->start_x][mympi->start_y]),
            1, mympi->rowtype, mympi->up, 123, mympi->cart);
    /*Receive lower border row from bottom neighbor*/
    MPI_Recv  (&(grid->theta[mympi->start_x][mympi->start_y+mympi->num_cells_y]),
            1, mympi->rowtype, mympi->down, 123, mympi->cart, &status);

    /*Send lower row to bottom neighbor*/
    MPI_Bsend (&(grid->theta[mympi->start_x][mympi->start_y+mympi->num_cells_y-1]),
            1, mympi->rowtype, mympi->down, 123, mympi->cart);
    /*Receive upper border row from top neighbor*/
    MPI_Recv  (&(grid->theta[mympi->start_x][mympi->start_y-1]),
            1, mympi->rowtype, mympi->up, 123, mympi->cart, &status);
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
 * Function to setup MPI data.
 *
 * (1) Initializes MPI
 * (2) Creates a cartesian communicator for border exchange
 * (3) Distributes the overall grid to the processes
 * (4) Sets up helpful data-type and MPI buffer
 *
 ******************************************************/
void heatMPISetup (heatGrid *grid, dataMPI* configMPI)
{
    int size,
        dims[2] = {0,0},
        periods[2] = {1,1},
        coords[2];
    int buf_size;
    char *buf;

    /* ==== (1) ==== */
    /* Base init*/
    MPI_Comm_rank (MPI_COMM_WORLD, &configMPI->rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    /* ==== (2) ==== */
    /* Create cartesian communicator*/
    MPI_Dims_create (size, 2, dims);
    MPI_Cart_create (MPI_COMM_WORLD, 2, dims, periods, 0, &configMPI->cart);

    /* Store neighbors in the grid */
    MPI_Cart_shift (configMPI->cart, 0, 1, &configMPI->left, &configMPI->right);
    MPI_Cart_shift (configMPI->cart, 1, 1, &configMPI->up,    &configMPI->down);

    /* ==== (3) ==== */
    /* Create partitioning of overall grid on processes */
    MPI_Cart_coords (configMPI->cart, configMPI->rank, 2, coords); /*My coordinate*/

    configMPI->start_x = 1 + (grid->xsize*coords[0])/dims[0];
    configMPI->num_cells_x = 1 + (grid->xsize*(coords[0]+1))/dims[0] - configMPI->start_x;
    configMPI->start_y = 1 + (grid->ysize*coords[1])/dims[1];
    configMPI->num_cells_y = 1 + (grid->ysize*(coords[1]+1))/dims[1] - configMPI->start_y;

    /* ==== (4) ==== */
    /* Create datatype to communicate one row */
    MPI_Type_vector (
            configMPI->num_cells_x, /* #blocks */
            1, /* #elements per block */
            grid->ysize+2, /* #stride */
            MPI_DOUBLE, /* old type */
            &configMPI->rowtype /* new type */ );
    MPI_Type_commit (&configMPI->rowtype);

    /* Create buffer for MPI */
    buf_size = sizeof(double) * grid->ysize * grid->ysize + MPI_BSEND_OVERHEAD;
    buf = (char*) malloc (buf_size);
    MPI_Buffer_attach (buf, buf_size);
}

/******************************************************
 * Function to free and finalize MPI.
 ******************************************************/
void heatMPIFree (dataMPI* configMPI)
{
    int buf_size;
    char *buf;

    MPI_Type_free (&configMPI->rowtype);
    MPI_Comm_free (&configMPI->cart);
    MPI_Buffer_detach (&buf, &buf_size); /*Free MPI buffer*/
    if (buf) free (buf);
}


/******************************************************
 * Gathers all data on process 0
 *
 * For output and total energy calculation it is
 * necessary to receive all sub-grids on process 0.
 *
 * It is a simple, but non-optimal implementation.
 ******************************************************/
void heatMPIGather (heatGrid *grid, dataMPI* mympi)
{
    int block_size[4]; /*stores: x_start,y_start, num_cells_x, num_cells_y*/
    MPI_Datatype blocktype;
    MPI_Status status;
    int i, size;

    /*Slaves send data*/
    if (mympi->rank != 0)
    {
        /*Prepare block info to be sent*/
        block_size[0] = mympi->start_x;
        block_size[1] = mympi->start_y;
        block_size[2] = mympi->num_cells_x;
        block_size[3] = mympi->num_cells_y;

        /* Create datatype to communicate one block*/
        MPI_Type_vector (
                mympi->num_cells_x, /* #blocks */
                mympi->num_cells_y, /* #elements per block */
                grid->ysize+2, /* #stride */
                MPI_DOUBLE, /* old type */
                &blocktype /* new type */ );
        MPI_Type_commit (&blocktype);

        MPI_Send (block_size, 4, MPI_INT, 0, 123, MPI_COMM_WORLD);
        MPI_Send (&grid->theta[mympi->start_x][mympi->start_y],1 ,blocktype, 0, 123, MPI_COMM_WORLD);

        MPI_Type_free (&blocktype);
    }
    else
    /*Master Receives data*/
    {
        MPI_Comm_size (MPI_COMM_WORLD, &size);
        for (i = 1; i < size; i++)
        {
            /*Receive Block Info*/
            MPI_Recv (block_size, 4, MPI_INT, i, 123, MPI_COMM_WORLD, &status);

            /* Create datatype to communicate one block*/
            MPI_Type_vector (
                    block_size[2], /* #blocks */
                    block_size[3], /* #elements per block */
                    grid->ysize+2, /* #stride */
                    MPI_DOUBLE, /* old type */
                    &blocktype /* new type */ );
            MPI_Type_commit (&blocktype);

            MPI_Recv (&grid->theta[block_size[0]][block_size[1]],1 ,blocktype, i, 123, MPI_COMM_WORLD, &status);

            MPI_Type_free (&blocktype);
        }
    }
}

/******************************************************
 * Main program and time stepping loop.
 ******************************************************/
int main (int argc, char** argv)
{
    heatGrid mygrid;
    double dt, dthetamax, energyInitial, energyFinal, t1, t2;
    int step, nsteps=20, required, provided;
    dataMPI mympi;

    /* initialize MPI */
    required = MPI_THREAD_FUNNELED;
    MPI_Init_thread (&argc, &argv, required, &provided);

    /* create heatGrid and initialize variables */
    heatAllocate(&mygrid, 8192, 8192);
    heatInitialize(&mygrid);
    dt = 0.05;
    dthetamax = 100.0;

    /* setup MPI */
    heatMPISetup (&mygrid, &mympi);

    /* Work only for master process
     * No Gather necessary here, all initialize equally*/
    if (mympi.rank == 0)
    {
        /* output of initial grid */
        // printf("initial grid:\n");
        // heatPrint(&mygrid);

        /* energy of initial grid */
        heatTotalEnergy(&mygrid, &energyInitial);

        t1 = MPI_Wtime();
    }

    /* time stepping loop */
    for( step=1 ; step<=nsteps ; step++)
    {
        heatBoundary(&mygrid, &mympi);
        heatTimestep(&mygrid, &mympi, dt, &dthetamax );
    }

    /* Gather data on process 0 for output*/
    heatMPIGather (&mygrid, &mympi);

    /* Work only for master process*/
    if (mympi.rank == 0)
    {
        t2 = MPI_Wtime();

        /* output of final grid */
        // printf ("\ngrid after %d iterations:\n",nsteps);
        // heatPrint(&mygrid);

        /* energy of final grid */
        heatTotalEnergy(&mygrid, &energyFinal);

        printf("\n= Energy Conservation Check =\n");
        printf(" initial Energy: %20.3f\n",energyInitial);
        printf("   final Energy: %20.3f\n",energyFinal);
        printf("     Difference: %20.3f\n",energyFinal-energyInitial);
        printf("           Time: %20.3f s\n", t2 - t1 );
    }

    heatDeallocate(&mygrid);

    /* Finalize MPI*/
    heatMPIFree (&mympi);
    MPI_Finalize ();

    return 0;
}

