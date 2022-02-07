# Download weight file
The weight file for edt prediction exceeds the file limit for github, thus can't be uploaded here. 

You can download the weight file here:

[weight_file](https://pitt-my.sharepoint.com/:u:/g/personal/huijing_pitt_edu/EQP32o-W8q5Kktpi47OebvoB3dAFw_ws3UeZkPH5qBfteA?e=dqZp4B)

After downloading, put this file into the folder /7-21-21-pipeline-testing/sample_data/models/imgseg_weights

# Try out pipeline

This example comtains 4 notebooks

- 1-Edt prediction.ipynb: predict the edt images based on the weight file. 
- 2-Prepare-single-cell-images-for-am-training.ipynb: interactively picking single cells for traing icnn apoptosis weight. (Hint: you can modify this file for interactive picking cells for icnn seg weight file.)
- 3-Retrain-apoptosis-icnn.ipynb: Training for the icnn_am weight.
- 4-Pipeline run through.ipynb: a sample script for running all steps in the pipeline. 

# Folder structure

```
A_A1_1/    # The /img folder, usually the image input, contains a single image time series.
|   +-- xx.tif
|   +-- xx.tif
|
A_A1_1_*_output/ #    The output/ folder.
+-- reg/    # edt prediction folder
|   +-- xx.tif
|   +-- xx.tif
|  
+-- seg/    # segmentation mask folder
|   +-- xx.png
|   +-- xx.png
|
+-- rgb_num/    # color coded image folder
|   +-- xx.png
|   +-- xx.png
|  
+-- celltrack.db 
+-- am_record.csv
+-- mitosis_record.csv
+-- Per_Object_relink.csv
+-- traj_object_num.csv
```
