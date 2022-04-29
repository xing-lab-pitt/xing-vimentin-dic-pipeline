# main output directory name
date=12-19-21
desc=72hr_no-treat 
job_outdir=${date}_${desc}

# define paths & files
script_dir=/net/capricorn/home/xing/tch42/projects/emt/scripts/3-compute
tools_dir=/net/capricorn/home/xing/tch42/projects/emt/scripts/memes
dat_dir=/net/capricorn/home/xing/tch42/projects/emt/data
out_dir=${dat_dir}/out/
ori_dir=${dat_dir}/ori/pcna/${job_outdir}

icnn_seg_wts_file=${dat_dir}/wts/icnn_seg/icnn_seg_dc_comb_wk.hdf5

### mkdir and resize ###
for i in {1..3} ; do
#for i in 0{1..3} ; do
	img_path=${ori_dir}/XY${i}
	output_path=${out_dir}/${job_outdir}/XY${i}
	sbatch run_pipe_3-9_receiver.sh $script_dir $tools_dir $dat_dir $img_path $output_path $icnn_seg_wts_file
done


