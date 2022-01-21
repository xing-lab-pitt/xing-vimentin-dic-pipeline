# xing-vimentin-dic-pipeline

This repository is a Xing lab tool at PITT for analyzing imaging data. [adding citation?]
Hello

## Dev Codebase Contribution Process
- Follow feature-staging-main review process
    - create a specific branch for new feature
    - implement and test on your own branch
    - create pull request
    - discuss with lab members and merge into the main branch
- Follow python [google code style](https://google.github.io/styleguide/pyguide.html)


## Install Precommit Hook  
**You would like to ensure that the code changes you push have good code style**  
**This step enables pre-commit to check and auto-format your style before every git commit locally via commandline.**
### Install (once)  
`pip install pre-commit`  
`pre-commit install`  
### format all code (rare usage)  
`pre-commit run --all`  
Now everytime you run `git commit -m msg`, pre-commit will check your changes.


# Module structure
## Directories


### Notebooks
- **0_Enviroment_setup**: Step 1, setting up pipeline environment
- **1_preprocessing**: Step 2, files for generating training data and train for segmentation edt weight. 
- **2_ICNN_interactive_training**: Notebooks with interactive cell picker. (only have the apoptosis/mitosis version for now.)
- **2_ICNN_training_notebooks**: Notebooks with code for icnn_am and icnn_seg training. (heritage.)
- **Other1_Image_folders_processing**: includes notebook assisting migrate imgs with a pattern, and interactively look at images as video for each folder.
    - 1_img_file_to_folders.ipynb: assisting migrate imgs with a pattern
    - 2-img-folder-remove.ipynb: for a given folder list (each folder contains images from one time series), browse the video for the folder.  
- **Other2_Various**:
    - cell_track_HK2_5min_interval.cpproj: the pipeline file for cellprofiler
    - pipe_meancontour_and_pca_modes.ipynb: calculating mean contour for single cells, this is needed for step 7 (pipe_7_cell_contours_calculation.py)
- **7-21-21-pipeline-testing**: example folder, contains necessary files for go through the pipeline. 

# Usage  
## Image Data Annotation  
Use image-annotation-tools to label images as required. Detailed descriptions are in the corresponding subfolders.  
  
## Create conda virtual environment for python

First you may want to set up a python virtual environment for the pipeline.

Go to the 0_Environment_setup folder, and type following code:

`conda env create -f tf1.yaml`

This create an environment from an environment.yml file. The first line of the yml file sets the new environment's name, here we call it 'tf1'.


This repository includes all needed files for segment and tracing single cell. 
- To work with a example data set, you can start from the `/7-21-21-pipeline-testing` folder. 
- To use this script as module, you can first add the folder path to your script, and then call each step as follows:

#### Note:
- The key package for this environment is to have TF version = 1.5, with curresponding Keras version. 
- TF 1.5 works only with lower h5py version <3. Thus if you have a higher version of h5py, you may want to downgrade it to 2.10 use following `conda install -c conda-forge h5py=2.10`.

## Huijing's Notebook Example
```python
import sys
sys.path.insert(1, 'path/to/xing-vimentin-dic-pipeline/')

import pipe_util2
import pipe_1_img_edt as edt1
import pipe_2_edt_watershed as ws2
import pipe_3_traj_reorganize_1st as tr1
import pipe_4_traj_reorganize_2nd as tr2
import pipe_5_traj_reorganize_3rd as tr3
import pipe_6_build_single_cell as p6
import pipe_7_cell_contours_calculation as p7
import pipe_8_haralick_calculation as p8
import pipe_9_morph_pca as p9
import pipe_10_haralick_pca as p10
import sbatch_jobs

edt1.folder_edt_predict(img_path, output_path, weight, model_mode)
ws2.simple_edt_watershed(img_path, output_path, icnn_am_weights, icnn_seg_weights, small_obj_thres = 10)
tr1.traj_reconganize1(img_path, output_path, icnn_seg_weights)
tr2.traj_reconganize2(output_path)
tr3.traj_reconganize3(output_path)
p6.build_single_cell(output_path)
p7.cell_contours_calculation(output_path, mean_contour_path)
p8.single_folder_run(img_path, output_path)
p9.morph_pca(top_path)
p10.haralick_pca(top_path)
```


## 1-Preprocessing

The preprocessing step creates a trained neural network for recognizing a pixel that belong to cell segments, and generate a weight file (.hdf5) that is reusable for further image analysis steps.

### 0.1-Prepare training date
- Export image data into single .tif files.
- Base on availability of the fuorescent channel, use matlab scripts in the /matlab_scripts folder. This step we manually generate cell segment images.
- Randomely crop the cell segment images to a predefined shape (320*320 as default) with modifying generate_reg_cla_patch.py. This step generating training images.
- Training CNN to generate weight file with train_reg_seg_augment.ipynb.

## Image Analysis pipeline
All steps are integrated in "run_all.sh", please read the instructions in this file first. Several model files including reg_seg model for step1, icnn models for step2 and step3 are not uploaded due to github file size limitation.
