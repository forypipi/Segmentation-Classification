nnUNetv2_ensemble -i /data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed/nnUNet_Dataset/nnUNet_results/Dataset111_WholeBodyTumor/2d_testset_inference /data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed/nnUNet_Dataset/nnUNet_results/Dataset111_WholeBodyTumor/3d_testset_inference -o /data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed/nnUNet_Dataset/nnUNet_results/Dataset111_WholeBodyTumor/ensemble_testset_inference -np 8
nnUNetv2_ensemble -i /data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed/nnUNet_Dataset/nnUNet_results/Dataset111_WholeBodyTumor/2d_trainset_inference /data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed/nnUNet_Dataset/nnUNet_results/Dataset111_WholeBodyTumor/3d_trainset_inference -o /data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed/nnUNet_Dataset/nnUNet_results/Dataset111_WholeBodyTumor/ensemble_trainset_inference -np 8