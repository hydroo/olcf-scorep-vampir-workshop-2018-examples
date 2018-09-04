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

if is_titan; then
	module switch PrgEnv-pgi PrgEnv-gnu
	module load scorep
elif is_summitdev; then
	# gcc/6.3.1 throws the following error: heat-c-omp-mpi: /sw/summitdev/gcc/6.2.1-20170301/src/gcc/libgomp/target.c:2965: GOMP_set_offload_targets: Assertion `!gomp_offload_targets_init' failed.
	module load gcc/5.4.0 scorep &> /dev/null
elif is_peak || is_summit; then
	module load gcc scorep &> /dev/null
fi
