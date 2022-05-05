# Xing-lab imaging analysis pipeline


## User guide

### Products after each step
#### step1: Segmentation 
```
/net/capricorn/home/xing/ken67/xing-vimentin-dic-pipeline/test_datasets/sample_test_data/out//XY1
└── edt
    ├── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif
    ├── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif
    └── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif
```

#### step2: edt transform with watershed  
```
/net/capricorn/home/xing/ken67/xing-vimentin-dic-pipeline/test_datasets/sample_test_data/out//XY1
├── am_record.csv
├── edt
│   ├── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif
│   ├── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif
│   └── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif
├── rgb_num
│   ├── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif.png
│   ├── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif.png
│   └── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif.png
└── seg
    ├── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif.png
    ├── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif.png
    └── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif.png
```

#### Cellprofiler step
```
step cell profiler complete
/net/capricorn/home/xing/ken67/xing-vimentin-dic-pipeline/test_datasets/sample_test_data/out//XY1
├── am_record.csv
├── cell_track.db
├── cell_track.properties
├── edt
│   ├── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif
│   ├── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif
│   └── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif
├── rgb_num
│   ├── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif.png
│   ├── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif.png
│   └── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif.png
└── seg
    ├── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif.png
    ├── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif.png
    └── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif.png
```

#### Step3: trajectory
```
step3 complete
/net/capricorn/home/xing/ken67/xing-vimentin-dic-pipeline/test_datasets/sample_test_data/out//XY1
├── am_record.csv
├── border_obj.npy
├── candi_mitosis_label
├── cell_track.db
├── cell_track.properties
├── edt
│   ├── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif
│   ├── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif
│   └── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif
├── false_link.npy
├── false_mitosis_obj
├── false_traj_label
├── fuse_pairs.npy
├── fuse_pairs_to_break.npy
├── Per_Object_break.csv
├── Per_Object_modify.csv
├── Per_Relationships_break.csv
├── postfuse_cells.npy
├── postsplit_cells.npy
├── postsplit_group
├── prefuse_cells.npy
├── prefuse_group
├── presplit_cells.npy
├── rgb_num
│   ├── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif.png
│   ├── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif.png
│   └── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif.png
├── seg
│   ├── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif.png
│   ├── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif.png
│   └── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif.png
├── split_pairs.npy
└── split_pairs_to_break.npy
```


#### Step4: trajectory
```
step4 complete
/net/capricorn/home/xing/ken67/xing-vimentin-dic-pipeline/test_datasets/sample_test_data/out//XY1
├── am_record.csv
├── border_obj.npy
├── candi_mitosis_label
├── cell_track.db
├── cell_track.properties
├── daughter_cells
├── edt
│   ├── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif
│   ├── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif
│   └── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif
├── false_link.npy
├── false_mitosis_obj
├── false_traj_label
├── fuse_pairs.npy
├── fuse_pairs_to_break.npy
├── mitoses.npy
├── mitosis_labels
├── mother_cells
├── Per_Object_break.csv
├── Per_Object_mitosis.csv
├── Per_Object_modify.csv
├── Per_Relationships_break.csv
├── postfuse_cells.npy
├── postsplit_cells.npy
├── postsplit_group
├── prefuse_cells.npy
├── prefuse_group
├── presplit_cells.npy
├── rgb_num
│   ├── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif.png
│   ├── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif.png
│   └── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif.png
├── seg
│   ├── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif.png
│   ├── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif.png
│   └── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif.png
├── split_pairs.npy
└── split_pairs_to_break.npy
```

#### Step5: trajectory
```
step5 complete
/net/capricorn/home/xing/ken67/xing-vimentin-dic-pipeline/test_datasets/sample_test_data/out//XY1
├── am_record.csv
├── border_obj.npy
├── candi_mitosis_label
├── cell_track.db
├── cell_track.properties
├── daughter_cells
├── edt
│   ├── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif
│   ├── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif
│   └── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif
├── false_link.npy
├── false_mitosis_obj
├── false_traj_label
├── fuse_pairs.npy
├── fuse_pairs_to_break.npy
├── link_obj_pairs.npy
├── mitoses.npy
├── mitosis_labels
├── mitosis_record.csv
├── mother_cells
├── Per_Object_break.csv
├── Per_Object_mitosis.csv
├── Per_Object_modify.csv
├── Per_Object_relink.csv
├── Per_Relationships_break.csv
├── postfuse_cells.npy
├── postsplit_cells.npy
├── postsplit_group
├── prefuse_cells.npy
├── prefuse_group
├── presplit_cells.npy
├── rgb_num
│   ├── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif.png
│   ├── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif.png
│   └── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif.png
├── seg
│   ├── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif.png
│   ├── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif.png
│   └── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif.png
├── split_pairs.npy
├── split_pairs_to_break.npy
└── traj_object_num.csv
```


#### Step6: build single cell
```
step6 complete
/net/capricorn/home/xing/ken67/xing-vimentin-dic-pipeline/test_datasets/sample_test_data/out//XY1
├── am_record.csv
├── border_obj.npy
├── candi_mitosis_label
├── cells
│   └── cells
├── cell_track.db
├── cell_track.properties
├── daughter_cells
├── edt
│   ├── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif
│   ├── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif
│   └── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif
├── false_link.npy
├── false_mitosis_obj
├── false_traj_label
├── fuse_pairs.npy
├── fuse_pairs_to_break.npy
├── link_obj_pairs.npy
├── mitoses.npy
├── mitosis_labels
├── mitosis_record.csv
├── mother_cells
├── Per_Object_break.csv
├── Per_Object_mitosis.csv
├── Per_Object_modify.csv
├── Per_Object_relink.csv
├── Per_Relationships_break.csv
├── postfuse_cells.npy
├── postsplit_cells.npy
├── postsplit_group
├── prefuse_cells.npy
├── prefuse_group
├── presplit_cells.npy
├── rgb_num
│   ├── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif.png
│   ├── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif.png
│   └── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif.png
├── seg
│   ├── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif.png
│   ├── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif.png
│   └── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif.png
├── split_pairs.npy
├── split_pairs_to_break.npy
└── traj_object_num.csv
```


#### Step7: calculate cell contour
```
step7 complete
/net/capricorn/home/xing/ken67/xing-vimentin-dic-pipeline/test_datasets/sample_test_data/out//XY1
├── am_record.csv
├── border_obj.npy
├── candi_mitosis_label
├── cells
│   └── cells
├── cell_track.db
├── cell_track.properties
├── daughter_cells
├── edt
│   ├── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif
│   ├── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif
│   └── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif
├── false_link.npy
├── false_mitosis_obj
├── false_traj_label
├── fuse_pairs.npy
├── fuse_pairs_to_break.npy
├── link_obj_pairs.npy
├── mitoses.npy
├── mitosis_labels
├── mitosis_record.csv
├── mother_cells
├── Per_Object_break.csv
├── Per_Object_mitosis.csv
├── Per_Object_modify.csv
├── Per_Object_relink.csv
├── Per_Relationships_break.csv
├── postfuse_cells.npy
├── postsplit_cells.npy
├── postsplit_group
├── prefuse_cells.npy
├── prefuse_group
├── presplit_cells.npy
├── rgb_num
│   ├── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif.png
│   ├── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif.png
│   └── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif.png
├── seg
│   ├── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif.png
│   ├── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif.png
│   └── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif.png
├── split_pairs.npy
├── split_pairs_to_break.npy
└── traj_object_num.csv
```

#### Step8: calculate haralick features
```
├── am_record.csv
├── border_obj.npy
├── candi_mitosis_label
├── cells
│   ├── cells
│   └── fluor_cells
├── cell_track.db
├── cell_track.properties
├── daughter_cells
├── edt
│   ├── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif
│   ├── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif
│   └── reg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif
├── false_link.npy
├── false_mitosis_obj
├── false_traj_label
├── fuse_pairs.npy
├── fuse_pairs_to_break.npy
├── link_obj_pairs.npy
├── mitoses.npy
├── mitosis_labels
├── mitosis_record.csv
├── mother_cells
├── Per_Object_break.csv
├── Per_Object_mitosis.csv
├── Per_Object_modify.csv
├── Per_Object_relink.csv
├── Per_Relationships_break.csv
├── postfuse_cells.npy
├── postsplit_cells.npy
├── postsplit_group
├── prefuse_cells.npy
├── prefuse_group
├── presplit_cells.npy
├── rgb_num
│   ├── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif.png
│   ├── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif.png
│   └── rgb_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif.png
├── seg
│   ├── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T001_XY1_C1.tif.png
│   ├── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T002_XY1_C1.tif.png
│   └── seg_a549_vim_rfp_pcna_gfp_time_lapse_72hr_no_treatment_011322_T003_XY1_C1.tif.png
├── split_pairs.npy
├── split_pairs_to_break.npy
└── traj_object_num.csv
```



## Environment installation guide
### 
- envrionment files are stored in envs/

```
conda create -n yourEnv python=3.7
pip install -U -r envs/tf1_requirements.txt
```
Note that python=3.7 is necesssary to support tensorflow 1.15 version. Higer versions of python do not exist when tensorflow==1.15.0 was released.

Create conda environment for cellprofiler4. Note that currently cellprofiler4 only supports 3.8, so it cannot be added to the tf1 environment we just created directly.
```
conda env create -f envs/cp4.yml 
```

## Install Precommit Hook  
**You would like to ensure that the code you push has good code style**  
**This step enables pre-commit to check and auto-format your style before every git commit.**
### Install (once)  
`pip install pre-commit`  
`pre-commit install`  
### format all code (rare usage)  
`pre-commit run --all`


## Development guide