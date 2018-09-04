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
    MPICC=cc
    LIBS=
    CUDALFLAGS=
    CUDALIBS=
endif
ifeq ($(host), summit)
    CC=gcc
    MPICC=mpicc
    LIBS=-lm
    CUDALFLAGS=-L${OLCF_CUDA_ROOT}/lib64
    CUDALIBS=-lcudart
endif

NVCC=nvcc

CFLAGS=-O2
LFLAGS=
NVCCFLAGS=-O2 -arch=sm_35

OMPFLAGS=-fopenmp
MPIFLAGS=-DUSE_MPI


.PHONY: all
all: notinstrumented instrumented 

.PHONY: notinstrumented
notinstrumented: bin/jacobi-omp bin/jacobi-cuda bin/jacobi-mpi-omp bin/jacobi-mpi-cuda
.PHONY: instrumented
instrumented: bin-scorep/jacobi-omp bin-scorep/jacobi-cuda bin-scorep/jacobi-mpi-omp bin-scorep/jacobi-mpi-cuda

# Not instrumented

jacobi-cuda-kernel.o: jacobi-cuda-kernel.cu
	$(NVCC) -c $(NVCCFLAGS) $^ -o $@ $(LFLAGS) $(LIBS)

bin/jacobi-omp: jacobi-mpi-omp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@ $(LFLAGS) $(LIBS)

bin/jacobi-cuda: jacobi-mpi-omp-cuda.c jacobi-cuda-kernel.o
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@ $(LFLAGS) $(CUDALFLAGS) $(LIBS) $(CUDALIBS)

bin/jacobi-mpi-omp: jacobi-mpi-omp.c
	$(MPICC) $(CFLAGS) $(OMPFLAGS) $(MPIFLAGS) $^ -o $@ $(LFLAGS) $(LIBS)

bin/jacobi-mpi-cuda: jacobi-mpi-omp-cuda.c jacobi-cuda-kernel.o
	$(MPICC) $(CFLAGS) $(OMPFLAGS) $(MPIFLAGS) $^ -o $@ $(LFLAGS) $(CUDALFLAGS) $(LIBS) $(CUDALIBS)

# Instrumented

jacobi-cuda-kernel-scorep.o: jacobi-cuda-kernel.cu
	scorep        $(NVCC) -c $(NVCCFLAGS) $^ -o $@ $(LFLAGS) $(LIBS)

bin-scorep/jacobi-omp: jacobi-mpi-omp.c
	scorep        $(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@ $(LFLAGS) $(LIBS)

bin-scorep/jacobi-cuda: jacobi-mpi-omp-cuda.c jacobi-cuda-kernel-scorep.o
	scorep --cuda $(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@ $(LFLAGS) $(CUDALFLAGS) $(LIBS) $(CUDALIBS)

bin-scorep/jacobi-mpi-omp: jacobi-mpi-omp.c
	scorep        $(MPICC) $(CFLAGS) $(OMPFLAGS) $(MPIFLAGS) $^ -o $@ $(LFLAGS) $(LIBS)

bin-scorep/jacobi-mpi-cuda: jacobi-mpi-omp-cuda.c jacobi-cuda-kernel.o
	scorep --cuda $(MPICC) $(CFLAGS) $(OMPFLAGS) $(MPIFLAGS) $^ -o $@ $(LFLAGS) $(CUDALFLAGS) $(LIBS) $(CUDALIBS)


$(shell mkdir -p bin)
$(shell mkdir -p bin-scorep)

.PHONY: clean
clean:
	rm -rf *.o bin bin-scorep *.err *.out _batch-job.lsf _batch-job.pbs *.input.c *.input.prep.c *.input.prep.c.opari.inc *.input.prep.opari.c
