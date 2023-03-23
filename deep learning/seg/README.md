2023.3.22 15:57
Modify /home/orfu/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py,  add "mirror_axes = None" in line 397 (configure_rotation_dummyDA_mirroring_and_inital_patch_size method) to close MirrorTransform during training and val

2023.3.23 15:01
Modify /home/orfu/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py,  modify "KFold=3" in line 512 and line 515 (do_split method) to run model in 3fold
Modify /home/orfu/nnUNet/nnunetv2/evaluation/find_best_configuration.py,  modify "param folds default to (0,1,2)" in line 81 (find_best_configuration function) to run model in 3fold

