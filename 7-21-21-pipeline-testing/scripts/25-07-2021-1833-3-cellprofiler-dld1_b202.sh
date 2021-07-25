#!/bin/bash
#SBATCH --account=huijing
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=32
#SBATCH --mail-user=651441679@qq.com
#SBATCH --mail-type=ALL
#SBATCH --output=/net/capricorn/home/xing/huijing/Segmentation/scripts/Image_analysis_pipeline_vim_dic/7-21-21-pipeline-testing/scripts/25-07-2021-1833-3-cellprofiler-dld1_b202.o
#SBATCH --job-name=CP_PIPELINE

eval "$(conda shell.bash hook)"
conda activate /net/capricorn/home/xing/weikang/anaconda3/envs/cellprofiler

/net/capricorn/home/xing/weikang/anaconda3/envs/cellprofiler/bin/python /net/capricorn/home/xing/huijing/Segmentation/scripts/Image_analysis_pipeline_vim_dic/CellProfiler.py -c -r -i /net/capricorn/home/xing/huijing/Segmentation/scripts/Image_analysis_pipeline_vim_dic/7-21-21-pipeline-testing/sample_data/06-21-21-B1_02_crop_part_30_output/seg/  -o /net/capricorn/home/xing/huijing/Segmentation/scripts/Image_analysis_pipeline_vim_dic/7-21-21-pipeline-testing/sample_data/06-21-21-B1_02_crop_part_30_output/ -p /net/capricorn/home/xing/huijing/Segmentation/scripts/Image_analysis_pipeline_vim_dic/cell_track_HK2_5min_interval.cpproj