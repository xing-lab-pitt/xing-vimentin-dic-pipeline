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
tools_dir=$2

eval "$(conda shell.bash hook)"
module load anaconda/3-cluster
source activate cp4

cellprofiler -c -r -i \
	${job_outdir_path}/seg/ -o \
	$job_outdir_path -p \
	${tools_dir}/cell_track_HK2_5min_interval_temporal_gap5.cppipe
