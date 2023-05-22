import pickle
import time
import itertools
import argparse


import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn, Tensor, optim, tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, dataset
from torchcam.methods import CAM

import SimpleITK as sitk
import nibabel as nib
import pandas as pd
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

import loss
from models.ViT import ViT
from models.resnet import ResNet18
from models.densenet import DenseNet121_3d
from models.models import UNet
from models.swinTransformer import SwinTransformer

from batchgenerators.utilities.file_and_folder_operations import load_json, join
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
import nnunetv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cls_model_name', type=str, default="ResNet3D_PETCT", help='A REQUIRED PARAMETER! input "model name"+"_"+"modality", like ResNet3D_PETCT')
    parser.add_argument('-seg_model_name', type=str, default="VNet3D_PETCT", help='A REQUIRED PARAMETER! input "model name"+"_"+"modality", like VNet3D_PETCT')
    parser.add_argument('-train_best', type=bool, default=False, help='A parameter for grid search or directly train best model according to the grid search result in "Performance" folder.')
    args = parser.parse_args()
    return args

def split_train_val_test(clinical: pd.DataFrame, test_radio=0.1, kFold=3):
    train_set, _ = train_test_split(clinical, test_size=test_radio, stratify=clinical['class'], random_state=42)
    train_set.insert(clinical.shape[1], 'training_label', np.nan)
    X, y = train_set['name'], train_set['class']
    skf = StratifiedKFold(n_splits=kFold, shuffle=True, random_state=42)
    for fold, (_, val_idx) in enumerate(skf.split(X, y)):
        train_set.iloc[val_idx, 2] = [fold] * len(val_idx)
    train_set = train_set.drop("class", axis=1)
    clinical = clinical.merge(train_set, how='left', left_on='name', right_on='name')
    clinical['training_label'].fillna(-1, inplace=True)
    clinical['training_label'] = clinical['training_label'].astype('int')
    clinical.to_csv("./pipeline/data/training label.csv")
    return clinical

def acc(pred, true, writer: SummaryWriter, epoch):
    accuracy = accuracy_score(true, pred)
    if writer != None:
        writer.add_scalar("accuracy", accuracy, epoch)
    return accuracy

def normalized(img, range1=(0, 1), device="cuda"):
    img = torch.as_tensor(img, dtype=torch.float32, device=device)      # int32 to float32
    if img.max() != img.min():
        img = (img - img.min()) / (img.max() - img.min())    # to 0--1
    img = img * (range1[1] - range1[0]) + range1[0]
    return img

def compute_loss(seg_loss, weights: list, outputs: list, mask, device='cuda'):
    total_loss = torch.as_tensor(0., dtype=torch.float32, device=device)
    if len(weights) != len(outputs):
        raise Exception("The len of weights should equal to the len of outputs")
    for weight, output in zip(weights, outputs):
        # print(output.shape, mask.shape)
        total_loss = total_loss + weight * seg_loss(output, mask)
        mask = F.interpolate(mask, scale_factor=0.5)
    return total_loss

def val(
        writer,
        cls_info: dict,
        seg_info: dict,
        model_name="ResNet3D_VNet3D",
        EPOCHS=50,
        tag="val0",
        device="cuda",
        using_checkpoint=False
        ):
    cls_model_name, seg_model_name = cls_info['model'], seg_info['model']

    with torch.no_grad():
        early_stop, max_dice = 0, 0

        classes, root_dir, modality, cls_batch = cls_info['classes'], cls_info['root_dir'], cls_info['modality'], cls_info['cls_batch']
        cls_nums_worker, cls_persistent_workers = cls_info['cls_nums_worker'], cls_info['cls_persistent_workers']
        cls_mode = cls_info['cls_mode']
        cls_loss = cls_info['cls_loss']

        for epoch in range(int(EPOCHS)):

            if using_checkpoint:
                # if epoch % 5 == 0:
                    # cls_model, seg_model = get_models(cls_model=cls_model_name, seg_model=seg_model_name, best=False, epoch=epoch, device=device)
                cls_model, seg_model = get_models(cls_model=cls_model_name, seg_model=seg_model_name, best=False, epoch=epoch, device=device)
            else:
                # if not use checkpoint: 
                cls_model, seg_model = get_models(cls_model=cls_model_name, seg_model=seg_model_name, best=True, epoch="val0", device=device)

            cls_model.eval()
            seg_model.eval()
            
            # classification
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            if epoch == 0:
                cls_data = MyData(classes, [cls_mode], 'cls', first_epoch=True, model_name=model_name, root_dir=root_dir, device=device, modality=modality)
                cls_loader = DataLoader(dataset=cls_data, batch_size=cls_batch, shuffle=False, drop_last=False, num_workers=cls_nums_worker, persistent_workers=cls_persistent_workers)
            else:
                cls_data = MyData(classes, [cls_mode], 'cls', first_epoch=False, model_name=model_name, root_dir=root_dir, device=device, modality=modality)
                cls_loader = DataLoader(dataset=cls_data, batch_size=cls_batch, shuffle=False, drop_last=False, num_workers=cls_nums_worker, persistent_workers=cls_persistent_workers)
            

            accuracy = cls_val(                              # save validation CAM
                cls_model, 
                cls_loader, 
                cls_loss,
                f"CLS [{epoch}/{EPOCHS}]",
                writer, 
                epoch,
                tag=tag,
                device=device
                ) 

            # segmentation
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            classes, root_dir, modality, seg_batch = seg_info['classes'], seg_info['root_dir'], seg_info['modality'], seg_info['seg_batch']
            seg_nums_worker, seg_persistent_workers = seg_info['seg_nums_worker'], seg_info['seg_persistent_workers']
            seg_mode = seg_info['seg_mode']
            seg_data = MyData(classes, [seg_mode], 'seg', model_name, root_dir=root_dir, device=device, modality=modality)
            seg_loader = DataLoader(dataset=seg_data, batch_size=seg_batch, shuffle=False, drop_last=False, num_workers=seg_nums_worker, persistent_workers=seg_persistent_workers)
            seg_loss = seg_info['seg_loss']

            dice = seg_val(
                seg_model, 
                seg_loader, 
                seg_loss, 
                f"SEG [{epoch}/{EPOCHS}]", 
                model_name, 
                writer,
                epoch, 
                tag=tag, 
                device=device
                )
            

            if max_dice < dice:
                max_dice = dice
                early_stop = epoch
            
    return early_stop, accuracy, max_dice

def cls_val(model, 
        data_loader: DataLoader, 
        cls_loss,
        tqdm_description,
        writer=None, 
        epoch=50, 
        tag="val0", 
        device="cuda",
        ):
    model.eval()
    with torch.no_grad():
        total_loss = torch.as_tensor(0., device=device)
        loop = tqdm(data_loader, total=len(data_loader))
        loop.set_description(tqdm_description)
        pred_result, true_result, pred_prob = torch.as_tensor([], device=device), torch.as_tensor([], device=device), torch.as_tensor([], device=device)
        for data in loop:
            images, labels, paths = data

            cam = CAM(model, input_shape=(3, 128, 128, 128), target_layer='layer2', fc_layer='fc')
            # Retrieve the CAM by passing the class index and the model output
            classification = model(images)
            clsloss = cls_loss(classification, labels)
            activation_map = cam(classification.argmax(dim=1).tolist(), classification)

            activation_map = F.interpolate(activation_map[0].unsqueeze(dim=1), scale_factor=(8, 8, 8), mode="trilinear")

            classification_softmax = nn.Softmax(dim=1)(classification)
            pred_result = torch.concat((pred_result, torch.argmax(classification_softmax, dim=1)))
            pred_prob = torch.concat((pred_prob, classification_softmax))
            true_result = torch.concat((true_result, labels))
            total_loss += clsloss

            for cam, path in zip(activation_map, paths):
                path = os.path.join(path, 'cam')
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(cam, os.path.join(path, 'inference_cam.pt'))

            for result, path in zip(classification_softmax.argmax(dim=1).tolist(), paths):
                if not os.path.exists(path):
                    os.makedirs(path)
                with open(os.path.join(path, 'predict_label.txt'), 'w+') as f:
                    f.write(str(result))

        pred_result = list(map(int, pred_result.cpu()))
        pred_prob = pred_prob.cpu().tolist()
        true_result = list(map(int, true_result.cpu()))

        accuracy = acc(pred_result, true_result, writer=writer, epoch=epoch)
        if writer:
            writer.add_scalar("classification loss", total_loss.cpu() / len(data_loader), epoch)

    return accuracy

def seg_dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = m1 * m2
    dice = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    return dice.sum() / num

def seg_compute_loss(seg_loss, weights: list, outputs: list, mask, device='cuda'):
    total_loss = torch.as_tensor(0., dtype=torch.float32, device=device)
    if len(weights) != len(outputs):
        raise Exception("The len of weights should equal to the len of outputs")
    for weight, output in zip(weights, outputs):
        # print(output.shape, mask.shape)
        total_loss = total_loss + weight * seg_loss(output, mask)
        mask = F.interpolate(mask, scale_factor=0.5)
    return total_loss

def seg_val(
        model, 
        data_loader: DataLoader, 
        seg_loss, 
        tqdm_description, 
        model_name="ResNet3D_VNet3D", 
        writer=None, 
        epoch=50, 
        device="cuda", 
        save=True, 
        tag="val0"
        ):
    model.eval()
    with torch.no_grad():
        total_loss = torch.as_tensor(0., device=device)
        loop = tqdm(data_loader, total=len(data_loader))
        loop.set_description(tqdm_description)
        dice = torch.as_tensor(0., device=device)
        for data in loop:
            images, masks, paths = data
            segmentation = model(images)

            segmentation = [i[:, 0].unsqueeze(dim=1) for i in segmentation]

            weights = [2**(-i) for i in range(len(segmentation))]
            weights_one = [i/sum(weights) for i in weights]
            segloss = seg_compute_loss(seg_loss, weights_one, segmentation, masks, device=device)

            sigmoid = nn.Sigmoid()
            segmentation = sigmoid(segmentation[0])
            dice += seg_dice_coeff(segmentation, masks)
            if save:
                segmentation_01 = segmentation.ge(0.5)
                for seg, path in zip(segmentation_01, paths):
                    seg = torch.squeeze(seg)
                    if not os.path.exists(os.path.join(path, 'predict_mask_pt')):
                        os.makedirs(os.path.join(path, 'predict_mask_pt'))
                    if os.path.exists(os.path.join(path, 'predict_mask_pt')):
                        torch.save(seg, os.path.join(path, 'predict_mask_pt', model_name + '_inference_01_mask.pt'))

                for seg, path in zip(segmentation, paths):
                    seg = torch.squeeze(seg)
                    if not os.path.exists(os.path.join(path, 'predict_mask_pt')):
                        os.makedirs(os.path.join(path, 'predict_mask_pt'))
                    if os.path.exists(os.path.join(path, 'predict_mask_pt')):
                        torch.save(seg, os.path.join(path, 'predict_mask_pt', model_name + '_inference_sigmoid_mask.pt'))

            total_loss += segloss

        if writer != None:
            writer.add_scalar("segmentation loss", total_loss.cpu() / len(data_loader), epoch)
            writer.add_scalar("dice", dice.cpu() / len(data_loader), epoch)

    return dice.cpu() / len(data_loader)

def get_models(
        cls_model,
        seg_model,
        best=True,
        epoch='val0',
        device="cuda",
):
    # seg_model = "VNet3D"
    if best:
        cls_model = torch.load(f"/data/orfu/DeepLearning/Segmentation-Classification/deep learning/pipeline/model/stage2/pipeline/cls/{cls_model}_{epoch}_best.pt").to(device)
        seg_model = torch.load(f"/data/orfu/DeepLearning/Segmentation-Classification/deep learning/pipeline/model/stage2/pipeline/seg/{seg_model}_{epoch}_best.pt").to(device)
    else:
        cls_model = torch.load(f"/data/orfu/DeepLearning/Segmentation-Classification/deep learning/pipeline/model/stage2/pipeline/checkpoint/cls/{cls_model}_{epoch}_01.pt").to(device)
        seg_model = torch.load(f"/data/orfu/DeepLearning/Segmentation-Classification/deep learning/pipeline/model/stage2/pipeline/checkpoint/seg/{seg_model}_{epoch}_01.pt").to(device)

    return cls_model, seg_model

class MyData(Dataset):
    def __init__(self, 
                 data_label: pd.DataFrame, 
                 label_list: list,
                 mode,
                 first_epoch=True,
                 model_name="ResNet3D_VNet3D",
                 device='cuda', 
                 modality="PETCT", 
                 root_dir=r'/data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed'
                 ):
        
        self.first_epoch = first_epoch

        if mode not in ['cls', 'seg']:
            raise Exception(f"Wrong input: mode={mode}, must be one of 'cls'/'seg'")
        self.mode = mode

        if self.mode == 'cls':
            mode_list = ['NEGATIVE', 'LYMPHOMA', 'MELANOMA', 'LUNG_CANCER']
        else:
            mode_list = ['LYMPHOMA', 'MELANOMA', 'LUNG_CANCER']
        # mode_list = ['NEGATIVE', 'LYMPHOMA', 'MELANOMA', 'LUNG_CANCER']

        self.data = data_label[data_label['training_label'].isin(label_list) & data_label['class'].isin(mode_list)]
        # if self.mode == 'seg':
        #     data_label.apply()
        self.data_label = data_label
        self.root_dir = root_dir
        self.device = device
        self.model_name = model_name

        if modality not in ['PETCT', 'CT', 'PET']:
            raise Exception(f"Wrong input: modality={modality}, must be one of 'PETCT'/'CT'/'PET'")
        self.modality = modality

       

    def __getitem__(self, item):

        label_dict = {'NEGATIVE': 0, 'LYMPHOMA': 1, 'MELANOMA': 2, 'LUNG_CANCER': 3}
        patient_id = self.data.iloc[item, 1]
        diagnose = self.data.iloc[item, 0]
        patient_dir = os.path.join(self.root_dir, diagnose, patient_id)
        img_method = "tensor"
        if img_method == "sitk":
            ct_path = os.path.join(patient_dir, 'CT_nii', '1.nii')
            pet_path = os.path.join(patient_dir, 'PET_nii', '1.nii')
            ct_image = normalized(sitk.GetArrayFromImage(sitk.ReadImage(ct_path)), device=self.device)
            pet_image = normalized(sitk.GetArrayFromImage(sitk.ReadImage(pet_path)), device=self.device)
        elif img_method == 'tensor':
            ct_path = os.path.join(patient_dir, 'CT_pt', '1.pt')
            pet_path = os.path.join(patient_dir, 'PET_pt', '1.pt')
            ct_image = normalized(torch.load(ct_path), device=self.device)
            pet_image = normalized(torch.load(pet_path), device=self.device)
        elif img_method == "pickle":
            ct_path = os.path.join(patient_dir, 'ct_pickle', '1.pickle')
            pet_path = os.path.join(patient_dir, 'pet_pickle', '1.pickle')
            ct_f = open(ct_path, 'rb')
            pet_f = open(pet_path, 'rb')
            ct_image = normalized(pickle.load(ct_f), device=self.device)
            pet_image = normalized(pickle.load(pet_f), device=self.device)


        clinical = pd.read_csv(os.path.join(patient_dir, "clinical.csv"))

        label = tensor(label_dict[clinical.loc[0, "label"]]).to(self.device)
        if self.modality == "PETCT":
            image = torch.stack((ct_image, pet_image))
        elif self.modality == "PET":
            image = pet_image.unsqueeze(dim=0)
        else:
            image = ct_image.unsqueeze(dim=0)
        
        if self.mode == "cls":
            mask_path = os.path.join(patient_dir, 'predict_mask_pt', self.model_name + '_inference_sigmoid_mask.pt')
            # mask_path = os.path.join(patient_dir, 'predict_mask_pt', self.model_name + '_inference_01_mask.pt')
            # mask_path = os.path.join(patient_dir, 'seg_pt', '1.pt')
            
            if (not self.first_epoch) and os.path.exists(mask_path):
                mask = torch.load(mask_path, map_location=self.device)
                if  len(mask.shape) == 3:
                    mask = mask.unsqueeze(dim=0)
            else:
                mask = torch.full((1, 128, 128, 128), fill_value=0., dtype=torch.float32, device=self.device)
            image = torch.cat([image, mask], dim=0)

            return image, label, patient_dir
        
        else:
            cam_path = os.path.join(patient_dir, 'cam', 'inference_cam.pt')
            if os.path.exists(cam_path):
                cam = torch.load(cam_path, map_location=torch.device(self.device))
            else:
                cam = torch.full((1, 128, 128, 128), fill_value=0., dtype=torch.float32, device=self.device)
            image = torch.cat([image, cam], dim=0)

            seg_path = os.path.join(patient_dir, 'seg_pt', '1.pt')
            seg = torch.as_tensor(torch.load(seg_path), dtype=torch.float32, device=self.device)
            seg = torch.where(seg==255., 1., 0.)
            seg = torch.unsqueeze(seg, dim=0)

            return image, seg, patient_dir

    def __len__(self):
        return len(self.data)

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("running on GPU")
    else:
        device = torch.device('cpu')
        print("running on CPU")

    start = time.time()
    torch.multiprocessing.set_start_method(method='forkserver', force=True)
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True
    cls_persistent_workers = False
    cls_nums_worker = 4
    seg_persistent_workers = False
    seg_nums_worker = 0
    root_dir = r'/data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed'
    # root_dir = r'/data/zhxie/oufu_data_400G/preprocessed'
    # root_dir = r'E:\dataset\preprocessed'

    kFold = 5
    labels = ["NEGATIVE", "LUNG_CANCER", "LYMPHOMA", "MELANOMA"]
    dfs = [pd.DataFrame({"class": [label] * len(os.listdir(os.path.join(root_dir, label))), "name": os.listdir(os.path.join(root_dir, label))}) for label in labels]
    Whole_df = pd.concat(dfs)
    classes = split_train_val_test(clinical=Whole_df, kFold=kFold, test_radio=0.1)
    torch.set_default_dtype(torch.float32)    

    args = parse_args()
    cls_model_modality = args.cls_model_name
    seg_model_modality = args.seg_model_name

    cls_model_name = cls_model_modality.split("_")[0]
    seg_model_name = seg_model_modality.split("_")[0]
    model_name = f"{cls_model_name}_{seg_model_name}"

    if cls_model_modality.split("_")[1] != seg_model_modality.split("_")[1]:
        raise Exception("cls and seg models should have same modality!")
    
    modality = cls_model_modality.split("_")[1]
    print(cls_model_name, seg_model_name, modality)

    cls_batch = 4
    seg_batch = 2
    epoch = 75
    using_checkpoint = True

    cls_loss = nn.CrossEntropyLoss().to(device)
    seg_loss = loss.HybridLoss()

    paths = [
        os.path.join('./pipeline/log/inference/val', f"{cls_model_name}_{seg_model_name}_{modality}"),
        os.path.join('./pipeline/log/inference/test', f"{cls_model_name}_{seg_model_name}_{modality}"),
    ]
    for dir in paths:
        if not os.path.exists(dir):
            os.makedirs(dir)
    print(paths)

    val_writer = SummaryWriter(paths[0])
    test_writer = SummaryWriter(paths[1])

    seg_info = {
        'model': seg_model_name,
        'classes': classes, 
        'root_dir': root_dir,
        'seg_loss': seg_loss,
        'modality': modality,
        'seg_batch': seg_batch, 
        'seg_nums_worker': seg_nums_worker, 
        'seg_persistent_workers': seg_persistent_workers,
        'seg_mode': 0,
    }
    cls_info = {
        'model': cls_model_name,
        'classes': classes, 
        'root_dir': root_dir,
        'cls_loss': cls_loss,
        'modality': modality,
        'cls_batch': cls_batch, 
        'cls_nums_worker': cls_nums_worker, 
        'cls_persistent_workers': cls_persistent_workers,
        'cls_mode': 0,
    }

    early_stop, _, _ = val(
            writer=val_writer, 
            cls_info=cls_info,
            seg_info=seg_info,
            EPOCHS=epoch, 
            device=device,
            using_checkpoint=using_checkpoint,
        )
    

    cls_info['cls_mode'] = -1
    seg_info['seg_mode'] = -1
    _, acc, max_dice = val(
            writer=test_writer, 
            cls_info=cls_info,
            seg_info=seg_info,
            EPOCHS=early_stop+1,
            device=device,
            using_checkpoint=using_checkpoint,
        )
    
    print("-"*20)
    print(f"accuracy: {acc}")
    print(f"max dice: {max_dice}")
    print(f"total time: {(time.time() - start) / 60: .2f}min")


if __name__ == '__main__':
    main()
    