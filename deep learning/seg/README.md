2023.3.22 15:57
Modify /home/orfu/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py,  add "mirror_axes = None" in line 397 (configure_rotation_dummyDA_mirroring_and_inital_patch_size method) to close MirrorTransform during training and val

2023.3.23 15:01
Modify /home/orfu/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py,  modify "KFold=3" in line 512 and line 515 (do_split method) to run model in 3fold
Modify /home/orfu/nnUNet/nnunetv2/evaluation/find_best_configuration.py,  modify "param folds default to (0,1,2)" in line 81 (find_best_configuration function) to run model in 3fold

2023.3.23 17:45
Modify /home/orfu/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py,  modify "g.save(join(self.output_folder, "network_architecture.svg"), format="svg")
" in line 477 (plot_network_architecture method) to save svg (save pdf cause error)
Modify /home/orfu/anaconda3/envs/pytorch/lib/python3.7/site-packages/hiddenlayer/graph.py,  modify "dot.render(file_name, directory=directory, cleanup=True, format=format)" in line 368 (save method) to save svg (save pdf cause error)