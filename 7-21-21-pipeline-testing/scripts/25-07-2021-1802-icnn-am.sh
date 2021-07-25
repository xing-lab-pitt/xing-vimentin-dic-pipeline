#!/bin/bash
#SBATCH --account=huijing
#SBATCH -p dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --mail-user=651441679@qq.com
#SBATCH --mail-type=ALL
#SBATCH --output /net/capricorn/home/xing/huijing/Segmentation/scripts/Image_analysis_pipeline_vim_dic/7-21-21-pipeline-testing/scripts/25-07-2021-1802-icnn-am.o
#SBATCH --job-name=icnn_am

eval "$(conda shell.bash hook)"
module load anaconda
source activate tf1
echo 'env activated'

python /net/capricorn/home/xing/huijing/Segmentation/scripts/vimentin_DIC_segmentation_pipeline/hj_modify_pipe/train_icnn_am.py ./icnn_am_testing/ /net/capricorn/home/xing/huijing/Segmentation/scripts/Image_analysis_pipeline_vim_dic/7-21-21-pipeline-testing/sample_data/models/am_icnn_weights/062121_dld1B102_am.hdf5
    