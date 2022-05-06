## User guide
### Required script modifications


### Usage

On cluster:
```
sbatch ./scripts/3-compute/run_pipe_0-2_all_positions.sh
```

After step0-2 completes, run

```
sbatch ./scripts/3-compute/run_pipe_0-2_all_positions.sh
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
