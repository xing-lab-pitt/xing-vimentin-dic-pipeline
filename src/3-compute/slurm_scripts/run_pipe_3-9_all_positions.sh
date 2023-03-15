# main output directory name
date=2022-10-28
desc=CFP_A549_VIM_168hr_NoTreat_NA_YL_Ti2e_2022-10-28
job_outdir=${date}_${desc}

# define paths & files
script_dir=./src/3-compute
tools_dir=./src/3-compute/tools
# dat_dir=/net/capricorn/home/xing/data/sorted_data/EBSS_Starvation/tif_STAV-A549_VIM_24hours_NoTreat_NA_YL_Ti2e_2022-12-21
dat_dir=/net/capricorn/home/xing/yak53/research/image_analysis/sample_tutorial2
out_dir=${dat_dir}/out/
ori_dir=${dat_dir}

icnn_seg_wts_file=${dat_dir}/wts/icnn_seg/icnn_seg_dc_comb_wk.hdf5

### mkdir and resize ###
for i in {4..4} ; do
#for i in 0{1..3} ; do
	img_path=${ori_dir}/XY${i}
	output_path=${out_dir}/XY${i}
	sbatch ${script_dir}/slurm_scripts/run_pipe_3-9_single_position.sh $script_dir $tools_dir $dat_dir $img_path $output_path $icnn_seg_wts_file
done


