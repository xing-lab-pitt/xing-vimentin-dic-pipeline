## Xinglab vim-dic pipeline documentation 3-compute
---
### Overview
#### This step of the pipeline focuses on extracting the information from the datasets. Parts of this step include cell segmentation, trajectory tracking, and dimensional reduction through PCA.
---
### 1-img-edt
#### Predicts Euclidean Distance Transform(EDT) map for the cell locations of the DIC channel images. This prepares for the watershed segmentation that follow. 