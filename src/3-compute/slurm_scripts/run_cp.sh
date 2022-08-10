#!/bin/bash

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     Slurm Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#SBATCH --account=yak53

#SBATCH --job-name=CP_PIPELINE

#SBATCH --cpus-per-task=16

#SBATCH --ntasks=1

#SBATCH --mail-user=yak53@pitt.edu
#SBATCH --mail-type=ALL

#SBATCH --output=${1}/cp_slurm_out.o

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#                     User Construction Section
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#!/bin/bash
output_path=$1
tools_dir=$2

eval "$(conda shell.bash hook)"
module load anaconda/3-cluster
source activate cp4

cellprofiler -c -r -i \
	${output_path}/seg/ -o \
	$output_path -p \
	cell_track_HK2_5min_interval.cppipe\
	# cell_track_HK2_5min_interval_temporal_gap5.cppipe
	
