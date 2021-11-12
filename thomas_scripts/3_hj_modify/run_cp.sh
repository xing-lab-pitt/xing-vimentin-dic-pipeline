#!/bin/bash

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     Slurm Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#SBATCH --account=tch42

#SBATCH --job-name=CP_PIPELINE

#SBATCH --cpus-per-task=16

#SBATCH --ntasks=1

#SBATCH --mail-user=tch42@pitt.edu
#SBATCH --mail-type=ALL

#SBATCH --output=${1}/cp_slurm_out.o

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     User Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#!/bin/bash

job_outdir_path=$1
script_dir=$2
cellprof=$3

source activate $cellprof

${cellprof}/bin/python \
	${script_dir}/CellProfiler.py -c -r -i \
	${job_outdir_path}/seg/ -o \
	$job_outdir_path -p \
	${script_dir}/cell_track_HK2_5min_interval.cpproj
