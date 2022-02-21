# main output directory name
date=01-13-22
desc=72hr_no-treat 
job_outdir=${date}_${desc}

# define paths & files
script_dir=/net/capricorn/home/xing/tch42/projects/a549_40x/scripts/3-compute
tools_dir=/net/capricorn/home/xing/tch42/projects/a549_40x/scripts/memes
dat_dir=/net/capricorn/home/xing/tch42/projects/a549_40x/data
out_dir=${dat_dir}/out/
ori_dir=${dat_dir}/ori/pcna/01-13-22_72hr_no-treat

icnn_seg_wts_file=${dat_dir}/wts/icnn_seg/icnn_seg_dc_comb_wk.hdf5

### mkdir and resize ###
for i in {1..1} ; do
#for i in 0{1..3} ; do
	img_path=${ori_dir}/XY${i}_test
	output_path=${out_dir}/${job_outdir}/XY${i}_test
	sbatch run_pipe_3-9_receiver.sh $script_dir $tools_dir $dat_dir $img_path $output_path $icnn_seg_wts_file
done


