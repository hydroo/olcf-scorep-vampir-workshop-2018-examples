!
!   Heat conduction demo program
!
!   solves the heat equation on a 2D grid
!
!   March 2009
!   Matthias.Lieber@tu-dresden.de
!   Tobias.Hilbrich@zih.tu-dresden.de
!
!   Adapted: Jan 2013
!
!   Hybrid MPI/OpenMP version.
!

module heatConduction

  implicit none
  include "mpif.h"

  ! this type represents the grid and its description
  type heatGrid
    ! current theta array
    double precision, pointer :: theta(:,:)
    ! new theta array
    double precision, pointer :: thetanew(:,:)
    ! domain size (number of grid cells) in x and y
    integer :: xsize
    integer :: ysize
    ! size of a grid cell
    double precision :: dx
    double precision :: dy
    ! "heat equation constant"
    double precision :: k
  end type heatGrid

  ! This type defines everything thats needed for MPI
  type dataMPI
    ! Own rank, used to only let master do output
    integer rank
    ! Comm for a cartesian distribution of the grid
    integer cart
    ! Neighbors in communicator
    integer up, down, left,right
    ! Start of a processes sub-grid (x, y)
    integer start_x
    integer start_y
    ! Number of cells in x or y direction for this process
    integer num_cells_x
    integer num_cells_y
    ! Datatype used to transfer a data column
    integer columntype
    ! Stores the adress of the MPI buffer passed to MPI for BSend
    character, pointer :: buf(:)
  end type dataMPI

contains

  ! Allocate the heatGrid and initialize all variables.
  !
  subroutine heatAllocate(grid, xsize, ysize)
    type(heatGrid), intent(inout) :: grid
    integer, intent(in) :: xsize
    integer, intent(in) :: ysize

    grid%xsize = xsize
    grid%ysize = ysize
    allocate( grid%theta   ( 0:xsize+1, 0:ysize+1 ) )
    allocate( grid%thetanew( 0:xsize+1, 0:ysize+1 ) )
    grid%theta    = 0.0d0
    grid%thetanew = 0.0d0
    grid%dx = 1.0d0
    grid%dy = 1.0d0
    grid%k = 1.0d0

  end subroutine heatAllocate


  ! Deallocate the heatGrid.
  !
  subroutine heatDeallocate(grid)
    type(heatGrid), intent(inout) :: grid

    deallocate( grid%theta )
    deallocate( grid%thetanew )
    nullify( grid%theta )
    nullify( grid%thetanew )

  end subroutine heatDeallocate


  ! Initialize the grid with some meaninful start values.
  !
  subroutine heatInitialize(grid)
    type(heatGrid), intent(inout) :: grid

    double precision :: xcenter, ycenter, radius, cr
    integer :: x, y

    ! initialize with a circle
    grid%theta = 0.0d0
    !xcenter = (grid%xsize + 1) * grid%dx / 2.0d0
    !ycenter = (grid%ysize + 1) * grid%dy / 2.0d0
    xcenter = (grid%xsize - 10) * grid%dx / 2.0d0
    ycenter = (grid%ysize + 6) * grid%dy / 2.0d0
    radius = min(grid%xsize * grid%dx, grid%ysize * grid%dy) * 0.25d0
    do x=1,grid%xsize
      do y=1,grid%ysize
        cr = sqrt( (x-xcenter)**2 + (y-ycenter)**2 )
        if( cr < radius ) then
          ! sharp boundary
          !grid%theta(x,y) = grid%theta(x,y) + 2.d0

          ! smooth boundary
          grid%theta(x,y) = grid%theta(x,y) + 2.d0 * heatInitFunc(cr/radius)

          ! very smooth boundary
          !grid%theta(x,y) = grid%theta(x,y) + 2.d0 * cos(3.14159*cr/(2*radius))
        end if
      end do
    end do

  end subroutine heatInitialize


  ! Polynomial function for the initialization of a smooth heat profile, 0 <= x <= 1.
  !
  double precision function heatInitFunc(x)
    double precision, intent(in) :: x
    heatInitFunc = -4.d0*x**3 + 4.d0*x**2 - 1.d0*x + 1.d0
  end function heatInitFunc


  ! Print grid to console.
  !
  subroutine heatPrint(grid)
    type(heatGrid), intent(in) :: grid
    integer :: x, y

    ! print header
    do x=1,grid%xsize
      write(*,'(A4)',advance='no') '===='
    end do
    write(*,*)

    ! print data
    do y=1,grid%ysize
      do x=1,grid%xsize
        if(grid%theta(x,y) < 1.d-100) then
          write(*,'(A4)',advance='no') '  . '
        else
          write(*,'(F4.1)',advance='no') grid%theta(x,y)
        end if
      end do
      write(*,*)
    end do

  end subroutine heatPrint


  ! Calculate one timestep of size dt on the grid.
  !
  ! for each grid point:
  !
  !                                  ( d2T   d2T )
  !          thetanew =  theta + k * ( --- + --- ) * dt
  !                                  ( dx2   dy2 )
  !                                  _____________
  !                                        |
  !                                        |
  !  dthetamax returns the max. value of this term
  !  (useful as exit condition for the time stepping)
  !
  subroutine heatTimestep(grid, mympi, dt, dthetamax)
    type(heatGrid), intent(inout) :: grid
    type(dataMPI),  intent(inout) :: mympi
    double precision, intent(in)  :: dt
    double precision, intent(out) :: dthetamax

    integer :: x, y, err
    double precision :: dtheta, mymax

    mymax = 0.0d0

    ! calculate the time step: read from theta, write new timestep to thetanew
    ! Only calculate on a processes sub-grid
    !$omp parallel do private(dtheta, x, y) reduction(max:dthetamax)
    do y=mympi%start_y,mympi%start_y + mympi%num_cells_y -1
      do x=mympi%start_x,mympi%start_x + mympi%num_cells_x -1

        dtheta = ( grid%theta(x-1,y) + grid%theta(x+1,y) - 2*grid%theta(x,y) ) / (grid%dx * grid%dx) &
               + ( grid%theta(x,y-1) + grid%theta(x,y+1) - 2*grid%theta(x,y) ) / (grid%dy * grid%dy)
        grid%thetanew(x,y) = grid%theta(x,y) + grid%k * dtheta * dt

        dthetamax = max(abs(dtheta), dthetamax) ! save max theta for the exit condition

      end do
    end do

    ! Make MPI reduction to get maximum dtheta of all processes
    CALL MPI_ALLREDUCE (mymax, dthetamax, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, err)

    ! update theta: copy thetanew to theta
    !$omp parallel do private(x, y)
    do y=mympi%start_y,mympi%start_y + mympi%num_cells_y -1
      do x=mympi%start_x,mympi%start_x + mympi%num_cells_x -1
        grid%theta(x,y) = grid%thetanew(x,y)
      end do
    end do

  end subroutine heatTimestep


  ! Set periodic boundary conditions.
  !
  ! The grid arrays are allocated with additional "ghost cells"
  ! in each spatial dimension. The lower boundary is copied to
  ! the upper ghost cells (and vice versa) for each dimension:
  !
  !    ___________         ___________
  !   |  _______ |        |  stuvwx  |
  !   | |abcdef| |        |f|abcdef|a|
  !   | |ghijkl| |        |l|ghijkl|g|
  !   | |mnopqr| |   ->   |r|mnopqr|m|
  !   | |stuvwx| |        |x|stuvwx|s|
  !   |__________|        |__abcdef__|
  !
  !
  ! For MPI:
  ! ========
  !
  ! Each process has ghost cells around its sub-grid
  ! e.g., Sub-grid & ghost cells for process "P_i":
  !
  !      __________________
  !     |Overall-grid      |
  !     |       ...        |
  !     |    __________    |
  !     |   |G ______ c|   |
  !     |   |h| P_i  |e|   |
  !     |   |o| Sub- |l|   |
  !     |...|s| Grid |l|...|
  !     |   |t|______|s|   |
  !     |   |__________|   |
  !     |       ...        |
  !     |__________________|
  !
  ! Ghost cells are received from the neighbors.
  ! Neighbors in turn receive border of the process.
  !
  ! e.g., Border exchange for Process 0:
  !         _____
  !        |     |
  !        |  4  |
  !        |     |
  !        |FGHIJ|
  !  _____ ======= _____
  ! |    v||abcde||q    |
  ! |    w||f   g||r    |
  ! |  2 x||h 0 i||s 1  |
  ! |    y||j   k||t    |
  ! |____z||lmnop||u____|
  !        =======
  !        |ABCDE|
  !        |     |
  !        |  3  |
  !        |_____|
  !
  ! O sends (e,g,i,k,p) to 1
  ! 0 receives (q,r,s,t,u) from 1 (put into its ghost cells)
  ! 0 sends (a,f,h,j,l) to 2
  ! 0 receives (v,w,x,y,z) from 2 (put into its ghost cells)
  ! ....
  !

  subroutine heatBoundary(grid, mympi)
    type(heatGrid), intent(inout) :: grid
    type(dataMPI) , intent(inout) :: mympi
    integer :: status(MPI_STATUS_SIZE)
    integer :: err

    ! Send left column to left neighbor
    CALL MPI_BSEND (grid%theta(mympi%start_x, mympi%start_y), &
            1, mympi%columntype, mympi%left, 123, mympi%cart, err)
    ! Receive Right border column from right neighbor
    CALL MPI_RECV  (grid%theta(mympi%start_x+mympi%num_cells_x, mympi%start_y), &
            1, mympi%columntype, mympi%right, 123, mympi%cart, status, err)

    ! Send right column to right neighbor
    CALL MPI_BSEND (grid%theta(mympi%start_x+mympi%num_cells_x-1, mympi%start_y), &
            1, mympi%columntype, mympi%right, 123, mympi%cart, err)
    ! Receive Left border column from left neighbor
    CALL MPI_RECV  (grid%theta(mympi%start_x-1, mympi%start_y), &
            1, mympi%columntype, mympi%left, 123, mympi%cart, status, err)

    ! Send upper row to top neighbor
    CALL MPI_BSEND (grid%theta(mympi%start_x,mympi%start_y), &
            mympi%num_cells_x, MPI_DOUBLE_PRECISION, mympi%up, 123, mympi%cart, err)
    ! Receive lower border row from bottom neighbor
    CALL MPI_RECV  (grid%theta(mympi%start_x, mympi%start_y+mympi%num_cells_y), &
            mympi%num_cells_x, MPI_DOUBLE_PRECISION, mympi%down, 123, mympi%cart, status, err)

    ! Send lower row to bottom neighbor
    CALL MPI_BSEND (grid%theta(mympi%start_x, mympi%start_y+mympi%num_cells_y-1), &
            mympi%num_cells_x, MPI_DOUBLE_PRECISION, mympi%down, 123, mympi%cart, err)
    ! Receive upper border row from top neighbor
    CALL MPI_RECV  (grid%theta(mympi%start_x, mympi%start_y-1), &
            mympi%num_cells_x, MPI_DOUBLE_PRECISION, mympi%up, 123, mympi%cart, status, err)

  end subroutine heatBoundary


  ! Calculate the total energy (sum of theta in all grid cells)
  !
  subroutine heatTotalEnergy(grid, energy)
    type(heatGrid), intent(in) :: grid
    double precision, intent(out) :: energy

    energy = SUM( grid%theta(1:grid%xsize,1:grid%ysize) )

  end subroutine heatTotalEnergy

  !
  ! Function to setup MPI data.
  !
  ! (1) Initializes MPI
  ! (2) Creates a cartesian communicator for border exchange
  ! (3) Distributes the overall grid to the processes
  ! (4) Sets up helpful data-type and MPI buffer
  !
  subroutine heatMPISetup (grid, configMPI)
    type(heatGrid), intent (inout) :: grid
    type(dataMPI), intent (inout)  :: configMPI
    integer :: size, buf_size, err, dims(2)=(/0,0/), coords(2)
    logical :: periods(2)=(/.true.,.true./)

    ! ==== (1) ====
    ! Base init
    CALL MPI_COMM_RANK (MPI_COMM_WORLD, configMPI%rank, err)
    CALL MPI_COMM_SIZE (MPI_COMM_WORLD, size, err)

    ! ==== (2) ====
    ! Create cartesian communicator
    CALL MPI_DIMS_CREATE (size, 2, dims, err)
    CALL MPI_CART_CREATE (MPI_COMM_WORLD, 2, dims, periods, .false., configMPI%cart, err)

    ! Store neighbors in the grid
    CALL MPI_CART_SHIFT (configMPI%cart, 0, 1, configMPI%left, configMPI%right, err)
    CALL MPI_CART_SHIFT (configMPI%cart, 1, 1, configMPI%up,   configMPI%down, err)

    ! ==== (3) ====
    ! Create partitioning of overall grid on processes
    CALL MPI_CART_COORDS (configMPI%cart, configMPI%rank, 2, coords, err) ! My coordinate

    configMPI%start_x = 1 + (grid%xsize*coords(1))/dims(1)
    configMPI%num_cells_x = 1 + (grid%xsize*(coords(1)+1))/dims(1) - configMPI%start_x
    configMPI%start_y = 1 + (grid%ysize*coords(2))/dims(2)
    configMPI%num_cells_y = 1 + (grid%ysize*(coords(2)+1))/dims(2) - configMPI%start_y

    ! ==== (4) ====
    ! Create datatype to communicate one row
    CALL MPI_TYPE_VECTOR (         &
            configMPI%num_cells_y, & ! #blocks
            1,                     & ! #elements per block
            grid%xsize+2,          & ! #stride
            MPI_DOUBLE_PRECISION,  & ! old type
            configMPI%columntype,  & ! new type
            err)
    CALL MPI_TYPE_COMMIT (configMPI%columntype, err)

    ! Create buffer for MPI
    buf_size = 8 * grid%ysize * grid%ysize + MPI_BSEND_OVERHEAD
    allocate( configMPI%buf(buf_size) )
    CALL MPI_Buffer_attach (configMPI%buf, buf_size, err)
  end subroutine heatMPISetup

  !
  ! Function to free and finalize MPI.
  !
  subroutine heatMPIFree (configMPI)
    type(dataMPI), intent(inout) :: configMPI
    integer :: buf_size, err
    character, pointer :: buf(:)

    CALL MPI_TYPE_FREE (configMPI%columntype, err)
    CALL MPI_COMM_FREE (configMPI%cart, err)
    !Do not detach the buffer, as there is a bug in Marmot
    !CALL MPI_BUFFER_DETACH (buf, buf_size, err) !Free MPI buffer
    deallocate (configMPI%buf)
  end subroutine heatMPIFree


  !
  ! Gathers all data on process 0
  !
  ! For output and total energy calculation it is
  ! necessary to receive all sub-grids on process 0.
  !
  ! It is a simple, but non-optimal implementation.
  !
  subroutine heatMPIGather (grid, mympi)
    type (heatGrid), intent(inout) :: grid
    type (dataMPI), intent(inout)  :: mympi

    integer :: block_size(4), & !stores: x_start,y_start, num_cells_x, num_cells_y
               blocktype, &
               status(MPI_STATUS_SIZE), &
               i, size, err

    ! Slaves send data
    if ( mympi%rank /= 0) then
      ! Prepare block info to be sent
      block_size(1) = mympi%start_x
      block_size(2) = mympi%start_y
      block_size(3) = mympi%num_cells_x
      block_size(4) = mympi%num_cells_y

      ! Create datatype to communicate one block
      CALL MPI_TYPE_VECTOR (        &
              mympi%num_cells_y,    & ! #blocks
              mympi%num_cells_x,    & ! #elements per block
              grid%xsize+2,         & ! #stride
              MPI_DOUBLE_PRECISION, & ! old type
              blocktype,            & ! new type
              err )
      CALL MPI_TYPE_COMMIT (blocktype, err)

      CALL MPI_SEND (block_size(1), 4, MPI_INTEGER, 0, 123, MPI_COMM_WORLD, err)
      CALL MPI_SEND (grid%theta(mympi%start_x, mympi%start_y),1 ,blocktype, 0, 123, MPI_COMM_WORLD, err)

      CALL MPI_TYPE_FREE (blocktype, err)
    else
    ! Master Receives data*/
      CALL MPI_COMM_SIZE (MPI_COMM_WORLD, size, err)
      do i=1, size-1
          ! Receive Block Info
          CALL MPI_RECV (block_size(1), 4, MPI_INTEGER, i, 123, MPI_COMM_WORLD, status, err)

          ! Create datatype to communicate one block
          CALL MPI_TYPE_VECTOR ( &
                  block_size(4), & ! #blocks
                  block_size(3), & ! #elements per block
                  grid%xsize+2, & ! #stride
                  MPI_DOUBLE_PRECISION, & ! old type
                  blocktype, & ! new type
                  err )
          CALL MPI_TYPE_COMMIT (blocktype, err)

          CALL MPI_RECV (grid%theta(block_size(1), block_size(2)),1 ,blocktype, i, 123, MPI_COMM_WORLD, status, err)

          CALL MPI_TYPE_FREE (blocktype, err)
      end do
    end if
  end subroutine heatMPIGather
end module heatConduction


! Main program and time stepping loop.
!
program heatExample

  use heatConduction
  implicit none

  type(heatGrid) :: mygrid
  type(dataMPI)  :: mympi
  double precision :: dt, dthetamax, energyInitial, energyFinal, &
                      t0, t1
  integer :: step, nsteps, required, provided, err

  ! initialize MPI
  required = MPI_THREAD_FUNNELED
  CALL MPI_INIT_THREAD ( required, provided, err)

  ! create heatGrid and initialize variables
  call heatAllocate(mygrid, 4096, 4096)
  call heatInitialize(mygrid)
  dt = 0.05d0
  dthetamax = 100.0d0
  nsteps = 20

  ! setup MPI
  call heatMPISetup (mygrid, mympi)

  ! Work only for master process
  ! No Gather necessary here, all initialize equally*/
  if (mympi%rank .EQ. 0) then
    ! output of initial grid
    ! write (*,*) 'initial grid:'
    ! call heatPrint(mygrid)

    ! energy of initial grid
    call heatTotalEnergy(mygrid, energyInitial)

    t0 = MPI_Wtime ()
  end if

  ! time stepping loop
  do step=1,nsteps
    call heatBoundary(mygrid, mympi)
    call heatTimestep(mygrid, mympi, dt, dthetamax )
  end do

  ! Gather data on process 0 for output
  call heatMPIGather (mygrid, mympi)

  ! Work only for master process
  if (mympi%rank .EQ. 0) then
    t1 = MPI_Wtime ()

    ! output of final grid
    ! write (*,*)
    ! write (*,*) 'grid after ',nsteps,' iterations:'
    ! call heatPrint(mygrid)

    ! energy of final grid
    call heatTotalEnergy(mygrid, energyFinal)

    write (*,*)
    write (*,'(A)')      '= Energy Conservation Check ='
    write (*,'(A,F20.3)') ' initial Energy: ',energyInitial
    write (*,'(A,F20.3)') '   final Energy: ',energyFinal
    write (*,'(A,F20.3)') '     Difference: ',energyFinal-energyInitial
    write (*,'(A,F20.3)') '       Time (s): ',t1-t0
  end if

  call heatDeallocate(mygrid)

  ! Finalize MPI
  call heatMPIFree (mympi)
  CALL MPI_FINALIZE (err)


end program heatExample
