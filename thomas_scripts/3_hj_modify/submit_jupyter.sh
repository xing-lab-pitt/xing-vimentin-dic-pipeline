#!/bin/bash

#SBATCH --job-name=jupyter_try

#SBATCH -p dept_cpu

#SBATCH -t 8:00:00

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=1

#SBATCH --output sbatch.stdout

## get tunneling info

XDG_RUNTIME_DIR=""

ipnport=$(shuf -i8000-9999 -n1)

ipnip=$(hostname -i)

## print tunneling instructions to sbtach.stdout

echo -e "

Copy/Paste this in your local terminal to ssh tunnel with remote

-----------------------------------------------------------------

ssh -N -L $ipnport:$ipnip:$ipnport huijing@cluster.csb.pitt.edu

-----------------------------------------------------------------

Then open a browser on your local machine to the following address

------------------------------------------------------------------

localhost:$ipnport  (prefix w/ https:// if using password)

------------------------------------------------------------------

"

## start an ipcluster instance and launch jupyter server

module load anaconda/3-cluster
conda info --envs
conda activate segmentation

jupyter-notebook --NotebookApp.iopub_data_rate_limit=100000000000000 \
                 --port=$ipnport --ip=$ipnip --NotebookApp.password='' --NotebookApp.token='' --no-browser
