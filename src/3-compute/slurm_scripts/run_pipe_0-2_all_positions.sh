# main output directory name
date=01-13-22
desc=72hr_no-treat 
job_outdir=${date}_${desc}

# define paths & files
script_dir=./src/3-compute
tools_dir=./src/3-compute/tools
dat_dir=/net/capricorn/home/xing/ken67/xing-vimentin-dic-pipeline/test_datasets/thomas_test_data_gap1
out_dir=${dat_dir}/out/
ori_dir=${dat_dir}

reg_wts_file=${dat_dir}/wts/reg/a549_reg_pt25_no-treat.hdf5 
icnn_am_wts_file=${dat_dir}/wts/icnn_am/icnn_am_dc_comb_wk.hdf5
icnn_seg_wts_file=${dat_dir}/wts/icnn_seg/icnn_seg_dc_comb_wk.hdf5

### mkdir and resize ###
for i in {1..1} ; do
#for i in 0{1..3} ; do
	img_path=${ori_dir}/XY${i}
	output_path=${out_dir}/XY${i}
	position_label=XY${i}
	mkdir -p $output_path
	sbatch ${script_dir}/run_pipe_0-2_single_position.sh $script_dir $tools_dir $dat_dir $img_path $output_path $ori_dir $position_label\
                $reg_wts_file $icnn_am_wts_file $icnn_seg_wts_file
done

