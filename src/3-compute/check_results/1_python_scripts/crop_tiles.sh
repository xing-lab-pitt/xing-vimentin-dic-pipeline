#!/bin/bash

#sorted_dir=/net/capricorn/home/xing/dap182/xing/image_analysis/sample-tutorial/
sorted_dir=/net/capricorn/home/xing/data/sorted_A549-VIM-PCNA_144hr_4ng-mL_TGFB_YY_Ti2e_2022-5-6/
crop_script_path=/net/capricorn/home/xing/dap182/xing/git/xing-vimentin-dic-pipeline/src/3-compute/check_results/1_python_scripts/crop_tiles.py 
#this is the total number of tiles you want to be created
num_tiles=4

for pos_number in $(ls $sorted_dir | awk "/.*XY.*/")
do
    echo
    echo $pos_number
    echo

    output_dir=${sorted_dir}cropped/${pos_number}/
    img_dir=${sorted_dir}${pos_number}/

    python $crop_script_path --img_dir $img_dir --tiles $num_tiles --output_dir $output_dir

    for i in $(seq 1 $num_tiles)
    do
        mkdir -p ${output_dir}${pos_number}_tile${i} 
        mv ${output_dir}*tile${i}* ${output_dir}${pos_number}_tile${i} >/dev/null 2>&1
    done


done