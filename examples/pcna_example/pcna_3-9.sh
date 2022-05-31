# main output directory name
date=12-19-21
desc=72hr_no-treat 
job_outdir=${date}_${desc}

# define paths & files
script_dir=./src/3-compute
tools_dir=./src/memes
dat_dir=/net/capricorn/home/xing/ken67/xing-vimentin-dic-pipeline/test_datasets/sample_test_data
out_dir=${dat_dir}/out/
ori_dir=${dat_dir}

icnn_seg_wts_file=${dat_dir}/wts/icnn_seg/icnn_seg_dc_comb_wk.hdf5

### mkdir and resize ###
for i in {1..1} ; do
#for i in 0{1..3} ; do
	img_path=${ori_dir}/XY${i}
	output_path=${out_dir}/XY${i}
	sbatch ${script_dir}/slurm_scripts/run_pipe_3-9_single_position.sh $script_dir $tools_dir $dat_dir $img_path $output_path $icnn_seg_wts_file
done


