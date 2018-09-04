#! /bin/sh

host="${HOSTNAME%\-*}"
if [ "$host" = "$HOSTNAME" ]; then # there was no -* suffix, happens for summit and peak
    if [ -n "$LMOD_SYSTEM_NAME" ]; then
        host="$LMOD_SYSTEM_NAME"
    else
        error_and_exit 'Unexpected host: Neither hostname is system-* nor $LMOD_SYSTEM_NAME is set. Aborting'
        exit -1
    fi
fi
is_titan()     { return $(test "$host" = "titan");     }
is_summitdev() { return $(test "$host" = "summitdev"); }
is_peak()      { return $(test "$host" = "peak");      }
is_summit()    { return $(test "$host" = "summit");    }

wd=$(pwd)

if [ $# -lt 1 ]; then
	echo >&2 "Usage:"
	echo >&2 "  $0 <executable>"
	exit 1
fi


executable="$1"
project="<projid>"
project="stf010"

is_scorep() { return $(echo $executable | grep -q scorep); }
project_lower="$(echo $project | tr '[:upper:]' '[:lower:]')"

if is_scorep; then
	jobname="$(basename $executable)-scorep"
	is_scorep_var=true
else
	jobname="$(basename $executable)"
	is_scorep_var=false
fi
queue_titan=debug
wt=5

if $(echo $executable | grep -q "mpi-omp"); then
	processes=2
	threads=2
elif $(echo $executable | grep -q "mpi"); then
	processes=4
	threads=1
elif $(echo $executable | grep -q "omp"); then
	processes=1
	threads=4
else
	processes=1
	threads=1
fi

# PBS job script
#---------------------------------------
cat > _batch-job.pbs <<EOF
#!/bin/sh

#PBS -N $jobname
#PBS -q $queue_titan
#PBS -l walltime=$wt:00,nodes=1,feature=gpudefault
#PBS -o \$PBS_JOBID-$jobname.out
#PBS -e \$PBS_JOBID-$jobname.err
#PBS -A $project_lower

source \$MODULESHOME/init/bash

module switch PrgEnv-pgi PrgEnv-gnu
module load scorep

cd \$PBS_O_WORKDIR
wd=\$PBS_O_WORKDIR

sources="$executable scorep.filter"
targetdir="\$MEMBERWORK/$project_lower/heat/\$PBS_JOBID-$jobname"

mkdir -p \$targetdir
cp -r \$sources \$targetdir

cd \$targetdir

export OMP_NUM_THREADS=$threads
# export SCOREP_ENABLE_TRACING=yes
# export SCOREP_TOTAL_MEMORY=300M
# export SCOREP_FILTERING_FILE=scorep.filter
# export SCOREP_METRIC_PAPI=PAPI_TOT_INS,PAPI_TOT_CYC

aprun -n$processes  -N$processes -d$threads -j1 ./$(basename $executable)

if $is_scorep_var; then
    cp -r \$targetdir/scorep-* \$wd
fi

EOF
#---------------------------------------

if is_summitdev; then
	gccmodule=gcc/5.4.0
	targetdir="/lustre/atlas/scratch/$USER/$project_lower/heat/\$LSB_JOBID-$jobname"
elif is_peak || is_summit; then
	gccmodule=gcc
	targetdir="/gpfs/alpinetds/$project_lower/scratch/$USER/heat/\$LSB_JOBID-$jobname"
fi

# LSF job script
#--------------------------------------
cat > _batch-job.lsf <<EOF
#BSUB -J $jobname
#BSUB -P $project_lower
#BSUB -o %J-$jobname.out
#BSUB -e %J-$jobname.err
#BSUB -nnodes 1
#BSUB -W $wt

\$MODULESHOME/init/bash
module load $gccmodule scorep &> /dev/null

cd \$LS_SUBCWD
wd=\$LS_SUBCWD

sources="$executable scorep.filter"
targetdir="$targetdir"

mkdir -p \$targetdir
cp -r \$sources \$targetdir

cd \$targetdir

export OMP_NUM_THREADS=$threads
# export SCOREP_ENABLE_TRACING=yes
# export SCOREP_TOTAL_MEMORY=300M
# export SCOREP_FILTERING_FILE=scorep.filter
# export SCOREP_METRIC_PAPI=PAPI_TOT_INS,PAPI_TOT_CYC

jsrun -n1 -a$processes -c$(($processes*$threads)) -r1 ./$(basename $executable)

if $is_scorep_var; then
    cp -r \$targetdir/scorep-* \$wd
fi

EOF
#---------------------------------------

if is_titan; then
	qsub _batch-job.pbs
elif is_summitdev || is_peak || is_summit; then
	bsub _batch-job.lsf
fi
