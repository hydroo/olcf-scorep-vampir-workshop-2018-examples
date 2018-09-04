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
!   Serial version.
!



module heatConduction

  implicit none

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
  subroutine heatTimestep(grid, dt, dthetamax)
    type(heatGrid), intent(inout) :: grid
    double precision, intent(in)  :: dt
    double precision, intent(out) :: dthetamax

    integer :: x, y
    double precision :: dtheta

    dthetamax = 0.0d0

    ! calculate the time step: read from theta, write new timestep to thetanew
    do y=1,grid%ysize
      do x=1,grid%xsize

        dtheta = ( grid%theta(x-1,y) + grid%theta(x+1,y) - 2*grid%theta(x,y) ) / (grid%dx * grid%dx) &
               + ( grid%theta(x,y-1) + grid%theta(x,y+1) - 2*grid%theta(x,y) ) / (grid%dy * grid%dy)
        grid%thetanew(x,y) = grid%theta(x,y) + grid%k * dtheta * dt

        dthetamax = max(abs(dtheta), dthetamax) ! save max theta for the exit condition

      end do
    end do

    ! update theta: copy thetanew to theta
    do y=1,grid%ysize
      do x=1,grid%xsize
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
  subroutine heatBoundary(grid)
    type(heatGrid), intent(inout) :: grid

    ! y direction, bottom
    grid%theta(1:grid%xsize, 0           ) = grid%theta(1:grid%xsize, grid%ysize)
    ! y direction, top
    grid%theta(1:grid%xsize, grid%ysize+1) = grid%theta(1:grid%xsize, 1)

    ! x direction, left
    grid%theta(0           , 1:grid%ysize) = grid%theta(grid%xsize, 1:grid%ysize)
    ! x direction, right
    grid%theta(grid%xsize+1, 1:grid%ysize) = grid%theta(1         , 1:grid%ysize)

  end subroutine heatBoundary


  ! Calculate the total energy (sum of theta in all grid cells)
  !
  subroutine heatTotalEnergy(grid, energy)
    type(heatGrid), intent(in) :: grid
    double precision, intent(out) :: energy

    energy = SUM( grid%theta(1:grid%xsize,1:grid%ysize) )

  end subroutine heatTotalEnergy

end module heatConduction



! Main program and time stepping loop.
!
program heatExample

  use heatConduction
  implicit none

  type(heatGrid) :: mygrid
  double precision :: dt, dthetamax, energyInitial, energyFinal
  integer :: step, nsteps

  ! create heatGrid and initialize variables
  call heatAllocate(mygrid, 4096, 4096)
  call heatInitialize(mygrid)
  dt = 0.05d0
  dthetamax = 100.0d0
  nsteps = 20

  ! output of initial grid
  ! write (*,*) 'initial grid:'
  ! call heatPrint(mygrid)

  ! energy of initial grid
  call heatTotalEnergy(mygrid, energyInitial)

  ! time stepping loop
  do step=1,nsteps
    call heatBoundary(mygrid)
    call heatTimestep(mygrid, dt, dthetamax )
  end do

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

  call heatDeallocate(mygrid)

end program heatExample
