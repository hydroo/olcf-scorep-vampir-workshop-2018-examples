ifneq ($(findstring titan,${HOSTNAME}),)
   host=titan
endif
ifneq ($(findstring summitdev,${HOSTNAME}),)
   #host=summitdev
   host=summit
endif
ifneq ($(findstring peak,$(shell hostname -f)),)
   #host=peak
   host=summit
endif
ifeq ($(host),)
    ifneq ($(findstring summit,$(shell hostname -f)),)
       host=summit
    endif
endif

ifeq ($(host), titan)
    CC=cc
    FC=ftn
    MPICC=cc
    MPIFC=ftn
    LIBS=
endif
ifeq ($(host), summit)
    CC=gcc
    FC=gfortran
    MPICC=mpicc
    MPIFC=mpif90
    LIBS=-lm
endif

CFLAGS=-O2
FFLAGS=$(CFLAGS)
LFLAGS=

OMPFLAGS=-fopenmp

.PHONY: all
all: notinstrumented instrumented

.PHONY: notinstrumented
notinstrumented: bin/heat-c bin/heat-c-omp bin/heat-c-mpi bin/heat-c-mpi-omp bin/heat-f bin/heat-f-omp bin/heat-f-mpi bin/heat-f-mpi-omp
.PHONY: instrumented
instrumented: bin-scorep/heat-c bin-scorep/heat-c-omp bin-scorep/heat-c-mpi bin-scorep/heat-c-mpi-omp bin-scorep/heat-f bin-scorep/heat-f-omp bin-scorep/heat-f-mpi bin-scorep/heat-f-mpi-omp


# Not instrumented

bin/heat-c: src/heat.c
	$(CC) $^ $(CFLAGS) -o $@ $(LFLAGS) $(LIBS)

bin/heat-c-omp: src/heat-omp.c
	$(CC) $^ $(CFLAGS) $(OMPFLAGS) -o $@ $(LFLAGS) $(LIBS)

bin/heat-c-mpi: src/heat-mpi.c
	$(MPICC) $^ $(CFLAGS) -o $@ $(LFLAGS) $(LIBS)

bin/heat-c-mpi-omp: src/heat-mpi-omp.c
	$(MPICC) $^ $(CFLAGS) $(OMPFLAGS) -o $@ $(LFLAGS) $(LIBS)

bin/heat-f: src/heat.f90
	$(FC) $^ $(FFLAGS) -o $@ $(LFLAGS) $(LIBS)

bin/heat-f-omp: src/heat.f90
	$(FC) $^ $(FFLAGS) $(OMPFLAGS) -o $@ $(LFLAGS) $(LIBS)

bin/heat-f-mpi: src/heat-mpi.f90
	$(MPIFC) $^ $(FFLAGS) -o $@ $(LFLAGS) $(LIBS)

bin/heat-f-mpi-omp: src/heat-mpi-omp.f90
	$(MPIFC) $^ $(FFLAGS) $(OMPFLAGS) -o $@ $(LFLAGS) $(LIBS)

# Instrumented

bin-scorep/heat-c: src/heat.c
	scorep $(CC) $^ $(CFLAGS) -o $@ $(LFLAGS) $(LIBS)

bin-scorep/heat-c-omp: src/heat-omp.c
	scorep $(CC) $^ $(CFLAGS) $(OMPFLAGS) -o $@ $(LFLAGS) $(LIBS)

bin-scorep/heat-c-mpi: src/heat-mpi.c
	scorep $(MPICC) $^ $(CFLAGS) -o $@ $(LFLAGS) $(LIBS)

bin-scorep/heat-c-mpi-omp: src/heat-mpi-omp.c
	scorep $(MPICC) $^ $(CFLAGS) $(OMPFLAGS) -o $@ $(LFLAGS) $(LIBS)

bin-scorep/heat-f: src/heat.f90
	scorep $(FC) $^ $(FFLAGS) -o $@ $(LFLAGS) $(LIBS)

bin-scorep/heat-f-omp: src/heat.f90
	scorep $(FC) $^ $(FFLAGS) $(OMPFLAGS) -o $@ $(LFLAGS) $(LIBS)

bin-scorep/heat-f-mpi: src/heat-mpi.f90
	scorep $(MPIFC) $^ $(FFLAGS) -o $@ $(LFLAGS) $(LIBS)

bin-scorep/heat-f-mpi-omp: src/heat-mpi-omp.f90
	scorep $(MPIFC) $^ $(FFLAGS) $(OMPFLAGS) -o $@ $(LFLAGS) $(LIBS)


$(shell mkdir -p bin)
$(shell mkdir -p bin-scorep)

.PHONY: clean
clean:
	rm -rf *.mod bin bin-scorep *.err *.out _batch-job.lsf _batch-job.pbs
