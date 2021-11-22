# main output directory name
date=11-2-21
desc=mt_20E-3_lt_35E-2 
job_outdir=${date}_${desc}

# define paths & files
script_dir=/net/capricorn/home/xing/tch42/Projects/A549_144hr/scripts/3_hj_modify
tools_dir=/net/capricorn/home/xing/tch42/Projects/A549_144hr/scripts/memes
dat_dir=/net/capricorn/home/xing/tch42/Projects/A549_144hr/data
out_dir=${dat_dir}/out
ori_dir=${dat_dir}/ori/10-21_pcna_no-treatment

reg_wts_file=${dat_dir}/wts/reg/a549_40x_weights_v3.hdf5
icnn_am_wts_file=${dat_dir}/wts/icnn_am/icnn_am_dc_comb_wk.hdf5
icnn_seg_wts_file=${dat_dir}/wts/icnn_seg/icnn_seg_dc_comb_wk.hdf5

### mkdir and resize ###
for i in {1..3} ; do
#for i in 0{1..3} ; do
	img_path=${ori_dir}/XY${i}
	output_path=${out_dir}/${job_outdir}/XY${i}
	position_label=XY${i}
	mkdir -p $output_path
	sbatch run_pipe_0-9_receiver.sh $script_dir $tools_dir $dat_dir $img_path $output_path $ori_dir $position_label\
		$reg_wts_file $icnn_am_wts_file $icnn_seg_wts_file
done


