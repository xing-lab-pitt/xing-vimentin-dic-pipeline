## User guide

### Copy required files into your data folder  
Note: this is a recommended procedure. Alternatively, you can specify the files used by the pipeline in corresponding bash scripts and skip this section.  
Assume you have a dataset folder `RPE1_exp0`, which contains XY1, XY2, XY3, etc. We recommend copy resource files into your dataset folder so that you can retrospect and know what resource files you used for a particular pipeline run.
 - copy `wts` containing weights file for machine learning models into `RPE1_exp0`
```bash
test_datasets/sample_test_data/wts/
├── icnn_am
│   └── icnn_am_dc_comb_wk.hdf5
├── icnn_seg
│   └── icnn_seg_dc_comb_wk.hdf5
└── reg
    └── a549_reg_pt25_no-treat.hdf5
```
   - You can download [our pretrained A549 model weights here](https://pitt-my.sharepoint.com/:u:/g/personal/ken67_pitt_edu/Ea2BU-tOkaxPntMDsRNMEu4BqBJyVKXM-14M3zUOf_WZaA?e=8mWnV9)
 - copy `stats` containing `mean_cell_contour` into `RPE1_exp0`
   - there are two versions of mean_cell_contour in our repo: py2 and py3. In newest version, please always use the py3 version, or pipeline mean contour loading is going to compain about an issue mentioned in https://github.com/xing-lab-pitt/xing-vimentin-dic-pipeline/issues/19. 
   - WARNING: this `stats` part will probably be refactored later because `stats` naming is not appropriate here. Meanwhile it contains some trajectory analysis from Thomas that we may need to understand and refactor later if needed
```bash
test_datasets/sample_test_data/stats/
└──  mean_cell_contour
```

### Required script modifications
- paths of your datasets
- paths of resource files

### Sample Usage

On cluster:
```
sbatch ./src/3-compute/run_pipe_0-2_all_positions.sh
```

After step0-2 completes, run

```
sbatch ./src/3-compute/run_pipe_0-2_all_positions.sh
```


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
