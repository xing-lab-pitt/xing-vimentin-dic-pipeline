import numpy as np
import sys
from datetime import datetime
import subprocess
import os

import pipe_util2


def sbatch_file_icnn_am(scr_folder, train_folder, weight, name=None):
    """
    output_folder = string, folder for storing the .o file. Usually the same as script folder
    """
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H%M")
    scr_folder = pipe_util2.folder_verify(scr_folder)

    if name:
        ofile_name = scr_folder + dt_string + "-icnn-am-" + name + ".o"
        sfile_name = scr_folder + dt_string + "-icnn-am-" + name + ".sh"

    else:
        ofile_name = scr_folder + dt_string + "-icnn-am" + ".o"
        sfile_name = scr_folder + dt_string + "-icnn-am" + ".sh"

    ofile_name = os.path.abspath(ofile_name)
    sfile_name = os.path.abspath(sfile_name)

    contents = """#!/bin/bash
#SBATCH --account=huijing
#SBATCH -p dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --mail-user=651441679@qq.com
#SBATCH --mail-type=ALL
#SBATCH --output %s
#SBATCH --job-name=icnn_am

eval "$(conda shell.bash hook)"
module load anaconda
source activate tf1
echo 'env activated'

python /net/capricorn/home/xing/huijing/Segmentation/scripts/Image_analysis_pipeline_vim_dic/train_icnn_am.py %s %s
    """ % (
        ofile_name,
        train_folder,
        weight,
    )
    s = open(sfile_name, "w")
    s.write(contents)
    s.close()

    return sfile_name


def sbatch_file_1_edt(output_folder, edt_input_path, edt_output_path, edt_weight_file, edt_model_mode, name=None):
    """
    output_folder = string, folder for storing the .o file. Usually the same as script folder
    edt_input_path
    """
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H%M")
    output_folder = pipe_util2.folder_verify(output_folder)
    edt_input_path = pipe_util2.folder_verify(edt_input_path)
    edt_output_path = pipe_util2.folder_verify(edt_output_path)

    if name:
        ofile_name = output_folder + dt_string + "-1-edt-" + name + ".o"
        sfile_name = output_folder + dt_string + "-1-edt-" + name + ".sh"

    else:
        ofile_name = output_folder + dt_string + "-1-edt" + ".o"
        sfile_name = output_folder + dt_string + "-1-edt" + ".sh"

    ofile_name = os.path.abspath(ofile_name)
    sfile_name = os.path.abspath(sfile_name)

    contents = """#!/bin/bash
#SBATCH --account=huijing
#SBATCH -p dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --mail-user=651441679@qq.com
#SBATCH --mail-type=ALL
#SBATCH --output %s
#SBATCH --job-name=edt

eval "$(conda shell.bash hook)"
module load anaconda
source activate tf1
echo 'env activated'

python /net/capricorn/home/xing/huijing/Segmentation/scripts/Image_analysis_pipeline_vim_dic/pipe_1_img_edt.py %s %s %s %s
    """ % (
        ofile_name,
        edt_input_path,
        edt_output_path,
        edt_weight_file,
        edt_model_mode,
    )
    s = open(sfile_name, "w")
    s.write(contents)
    s.close()

    return sfile_name


def sbatch_file_2_watershed(
    script_folder, watershed_img_path, watershed_output_path, icnn_am_weights, icnn_seg_weights, name=None
):

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H%M")
    script_folder = pipe_util2.folder_verify(script_folder)
    watershed_img_path = pipe_util2.folder_verify(watershed_img_path)
    watershed_output_path = pipe_util2.folder_verify(watershed_output_path)

    if name:
        ofile_name = script_folder + dt_string + "-2-watershed-" + name + ".o"
        sfile_name = script_folder + dt_string + "-2-watershed-" + name + ".sh"

    else:
        ofile_name = script_folder + dt_string + "-2-watershed" + ".o"
        sfile_name = script_folder + dt_string + "-2-watershed" + ".sh"

    ofile_name = os.path.abspath(ofile_name)
    sfile_name = os.path.abspath(sfile_name)
    contents = """#!/bin/bash
#SBATCH --account=huijing
#SBATCH -p dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --mail-user=651441679@qq.com
#SBATCH --mail-type=ALL
#SBATCH --output %s
#SBATCH --job-name=watershed

eval "$(conda shell.bash hook)"
module load anaconda
source activate tf1
echo 'env activated'

python /net/capricorn/home/xing/huijing/Segmentation/scripts/Image_analysis_pipeline_vim_dic/pipe_2_edt_watershed.py %s %s %s %s
    """ % (
        ofile_name,
        watershed_img_path,
        watershed_output_path,
        icnn_am_weights,
        icnn_seg_weights,
    )
    s = open(sfile_name, "w")
    s.write(contents)
    s.close()

    return sfile_name


# Check last error msg
# subprocess.run("exit 1", shell=True, check=True)


def sbatch_file_3_cellprofiler(script_folder, cp_output_path, name=None):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H%M")
    script_folder = pipe_util2.folder_verify(script_folder)
    cp_output_path = pipe_util2.folder_verify(cp_output_path)

    if name:
        ofile_name = script_folder + dt_string + "-3-cellprofiler-" + name + ".o"
        sfile_name = script_folder + dt_string + "-3-cellprofiler-" + name + ".sh"

    else:
        ofile_name = script_folder + dt_string + "-3-cellprofiler" + ".o"
        sfile_name = script_folder + dt_string + "-3-cellprofiler" + ".sh"

    ofile_name = os.path.abspath(ofile_name)
    sfile_name = os.path.abspath(sfile_name)
    contents = """#!/bin/bash
#SBATCH --account=huijing
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=32
#SBATCH --mail-user=651441679@qq.com
#SBATCH --mail-type=ALL
#SBATCH --output=%s
#SBATCH --job-name=CP_PIPELINE

eval "$(conda shell.bash hook)"
conda activate /net/capricorn/home/xing/weikang/anaconda3/envs/cellprofiler

/net/capricorn/home/xing/weikang/anaconda3/envs/cellprofiler/bin/python /net/capricorn/home/xing/huijing/Segmentation/scripts/Image_analysis_pipeline_vim_dic/CellProfiler.py -c -r -i %s  -o %s -p /net/capricorn/home/xing/huijing/Segmentation/scripts/Image_analysis_pipeline_vim_dic/cell_track_HK2_5min_interval.cpproj""" % (
        ofile_name,
        cp_output_path + "seg/",
        cp_output_path,
    )
    s = open(sfile_name, "w")
    s.write(contents)
    s.close()

    return sfile_name


def sbatch_job_icnn_am(scr_folder, train_folder, weight, name=None):
    """Generating the sbatch code and then submit the job."""
    scr_path = sbatch_file_icnn_am(scr_folder, train_folder, weight, name)
    cmd = "sbatch %s" % scr_path
    subprocess.call(cmd, shell=True)


def sbatch_job_1_edt(output_folder, edt_input_path, edt_output_path, edt_weight_file, edt_model_mode, name=None):
    """Generating the sbatch code and then submit the job."""
    scr_path = sbatch_file_1_edt(output_folder, edt_input_path, edt_output_path, edt_weight_file, edt_model_mode, name)
    cmd = "sbatch %s" % scr_path
    subprocess.call(cmd, shell=True)


def sbatch_job_2_watershed(
    script_folder, watershed_img_path, watershed_output_path, icnn_am_weights, icnn_seg_weights, name=None
):
    scr_path = sbatch_file_2_watershed(
        script_folder, watershed_img_path, watershed_output_path, icnn_am_weights, icnn_seg_weights, name
    )
    cmd = "sbatch %s" % scr_path
    subprocess.call(cmd, shell=True)


def sbatch_job_3_cellprofiler(script_folder, cp_output_path, name=None):

    # if file exists, then remove it.
    cp_output_path = pipe_util2.folder_verify(cp_output_path)
    if os.path.exists(cp_output_path + "cell_track.db"):
        p1 = cp_output_path + "cell_track.db"
        p2 = cp_output_path + "cell_track.properties"
        os.remove(p1)
        os.remove(p2)

    scr_path = sbatch_file_3_cellprofiler(script_folder, cp_output_path, name)
    cmd = "sbatch %s" % scr_path
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    pass
