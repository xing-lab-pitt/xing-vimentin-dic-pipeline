#!/bin/bash

#makes the slurm output folder and slurm array submission file 
### THIS VERSION ONLY CAN MAKE ONE SLURM SUBMISSION FILE SO IF DIFFERENT RUNS OF THE SAME DATA WANTS TO BE RUN, THIS FILE NEEDS TO BE CHANGED.
### ALL SLURM FILES WILL BE OVERWRITTEN EACH RUN

### THIS VERSION ONLY CAN MAKE ONE SLURM SUBMISSION FILE SO IF DIFFERENT RUNS OF THE SAME DATA WANTS TO BE RUN, THIS FILE NEEDS TO BE CHANGED.
### THIS ALSO ASSUMES THAT ALL POSITIONS ARE DONE WITH 0-2 BEFORE RUNNING THE STEPS 3-9. I BELIEVE IT WOULD START ANALZYING IMCOMPLETE SET OF AN XY POSITION
	### IF RUN BEFORE PREVIOUS STEPS ARE COMPLETE

source src/3-compute/slurm_scripts/slurm_config.sh

####################
#  SLURM SECTION   #
####################

#get name of experiment for naming slurm files
desc=${data_dir##*/}

#defining slurm output folder and slurm array submission file
slurm_dir=${data_dir}/slurm 
slurm_array_file=slurm_array_file_${pipeline_steps}_${desc}.txt
slurm_array_file_path=${slurm_dir}/submission_files/${slurm_array_file}

#finds the length of the array 
num_XY_pos=$(ls $data_dir | awk "/.*XY.*/" | wc -l ) 

#array_length=1-${num_XY_pos}
array_length=1-12
max_jobs_at_once=%6

mkdir -p ${slurm_dir}/stdout
mkdir -p ${slurm_dir}/stderr
mkdir -p ${slurm_dir}/submission_files

if [[ -e $slurm_array_file_path ]]
then
	rm $slurm_array_file_path 
fi

touch $slurm_array_file_path

#####################
#GENERATE JOB SCRIPT#
#####################

for XY_pos in $(ls $data_dir | awk "/.*XY.*/"); do

	for tile_num in $(ls ${data_dir}/${XY_pos}); do

		#img_path=${data_dir}/${XY_pos}
		#output_path=${out_dir}/${XY_pos}

		img_path=${data_dir}/${XY_pos}/${tile_num}
		output_path=${out_dir}/${XY_pos}/${tile_num}
		position_label=$XY_pos

		if [[ $pipeline_steps == 0-2 ]]
		then

			mkdir -p $output_path

			echo bash ${script_dir}/slurm_scripts/run_pipe_0-2_single_position.sh $script_dir $tools_dir $data_dir $img_path $output_path $data_dir $position_label\
						$reg_wts_file $icnn_am_wts_file $icnn_seg_wts_file >> $slurm_array_file_path

		elif [[ $pipeline_steps == 3-9 ]]
		then

			echo bash ${script_dir}/slurm_scripts/run_pipe_3-9_single_position.sh $script_dir $tools_dir $data_dir $img_path $output_path $icnn_seg_wts_file >> $slurm_array_file_path

		fi

	done

done

#####################
# SUBMIT JOB SCRIPT #
#####################

echo ${slurm_dir}/out/${desc}_${pipeline_steps}_XY%a.out 

sbatch --output=${slurm_dir}/stdout/${desc}_${pipeline_steps}_XY%a.out \
       --error=${slurm_dir}/stderr/${desc}_${pipeline_steps}_XY%a.err \
       --array=${array_length}${max_jobs_at_once} \
         ${script_dir}/slurm_scripts/submit_array_job_${pipeline_steps}.sh $slurm_array_file_path

