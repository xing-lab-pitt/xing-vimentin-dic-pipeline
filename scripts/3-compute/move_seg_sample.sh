#!/bin/bash
sample_per_dir=50
pos_start=1
pos_end=$3
# not out dest, but existing segmentation output folder 
output_dir=$1
total_time=$2
interval=$((total_time/sample_per_dir))
seg_sample_path=$output_dir/seg_sample
#seg_sample_path=/mnt/data0/Ke/weikang_exp_process/2019-05-05_A549_vim/output/seg_sample

rm -rf $seg_sample_path
mkdir $seg_sample_path
echo $interval
for  ((i=pos_start;i<pos_end+1;i++))
do
    echo $i;
    seg_folder=$output_dir/$i/seg;
    echo $seg_folder;
    for ((j=1;j<=total_time;j=j+interval))
    do
	echo $j
	echo $path
	# NOTE: path need to handle padded zeros of file names
	# change according to file name format, i is xy_pos and j is time
	path=$(printf $seg_folder/"seg_a549_vim_tgf4ngxy%02dt%03dc2.png" $i $j)
	# path=$(printf $seg_folder/"seg_A549_2d_T%03d_XY%02d_Mono1.png" $j $i)
	cp $path $seg_sample_path/
    done
done

    
