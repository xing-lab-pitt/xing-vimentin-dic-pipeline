#!/bin/bash
#SBATCH --account=huijing
#SBATCH -p dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --mail-user=651441679@qq.com
#SBATCH --mail-type=ALL
#SBATCH --output /net/capricorn/home/xing/huijing/Segmentation/scripts/Image_analysis_pipeline_vim_dic/7-21-21-pipeline-testing/scripts/25-07-2021-1805-1-edt.o
#SBATCH --job-name=edt

eval "$(conda shell.bash hook)"
module load anaconda
source activate tf1
echo 'env activated'

python /net/capricorn/home/xing/huijing/Segmentation/scripts/Image_analysis_pipeline_vim_dic/pipe_1_img_edt.py /net/capricorn/home/xing/huijing/Segmentation/scripts/Image_analysis_pipeline_vim_dic/7-21-21-pipeline-testing/sample_data/06-21-21-B1_02_crop_part_30/ /net/capricorn/home/xing/huijing/Segmentation/scripts/Image_analysis_pipeline_vim_dic/7-21-21-pipeline-testing/sample_data/06-21-21-B1_02_crop_part_30_output/ /net/capricorn/home/xing/huijing/Segmentation/scripts/Image_analysis_pipeline_vim_dic/7-21-21-pipeline-testing/sample_data/models/imgseg_weights/7-21-21-dld1_edt_fcn_add1_ep100.hdf5 reg_seg
    