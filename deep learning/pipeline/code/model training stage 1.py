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
    parser.add_argument('-seg_model_name', type=str, default="nnUNet3D_PETCT", help='A REQUIRED PARAMETER! input "model name"+"_"+"modality", like VNet3D_PETCT')
    parser.add_argument('-train_best', type=bool, default=True, help='A parameter for grid search or directly train best model according to the grid search result in "Performance" folder.')
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
    accuracy = np.round(accuracy_score(true, pred), 4)
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

def training(
        cls_model: nn.modules, 
        seg_model: nn.modules,
        cls_optimizer,
        seg_optimizer,
        cls_loss,
        seg_loss,
        cls_train_loader: DataLoader,
        cls_test_loader: DataLoader,
        seg_train_loader: DataLoader,
        seg_test_loader: DataLoader,
        writer: dict,
        model_name="ResNet3D_nnUNet3D",
        EPOCHS=50,
        cls_threshold=0.1,
        seg_threshold=0.,
        cls_model_name="ResNet3D",
        seg_model_name="nnUNet3D",
        tag="val0",
        device="cuda"
        ):
    
    cls_model.train()
    with torch.enable_grad():
        early_stop, cls_step, seg_step, max_acc, max_dice = 0, 0, 0, 0, 0

        for epoch in range(int(EPOCHS)):
            cls_loss_list, seg_loss_list = [], []

            # classification
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            loop = tqdm(cls_train_loader, total=len(cls_train_loader))
            for data in loop:
                cls_step += 1
                images, labels, _ = data
                # image: [从头向下, 从前向后, 从左向右]
                cls_optimizer.zero_grad()

                classification = cls_model(images)
                cls_train_loss = torch.abs(cls_loss(classification, labels) - cls_threshold) + cls_threshold
                cls_loss_list.append(float(cls_train_loss.cpu()))

                cls_train_loss.backward()
                cls_optimizer.step()

                if cls_step % 5 == 0 and len(cls_loss_list) >= 5:
                    writer["train"].add_scalar("classification loss", sum(cls_loss_list[-5:]) / 5, cls_step)
                # writer.add_scalar("train loss", train_loss, cls_step)

                loop.set_description(f'CLS Epoch [{epoch + 1}/{EPOCHS}]')
                # loop.set_postfix(loss=float(cls_train_loss.cpu()))

            _, accuracy = cls_val(                              # save validation CAM
                cls_model, 
                cls_test_loader, 
                cls_loss, 
                "validation_CAM",
                writer["test"], 
                (epoch+1)*len(loop), 
                tag=tag,
                save=True, 
                device=device
                )
            cls_val(cls_model, cls_train_loader, cls_loss, tqdm_description="train_CAM", device=device)      # save training CAM

            if max_acc < accuracy:
                max_acc = accuracy
                early_stop = epoch
                torch.save(cls_model, os.path.join('/data/orfu/DeepLearning/Segmentation-Classification/deep learning/pipeline/model/cls', f"{cls_model_name}_{tag}_best.pt"))
    

            # segmentation
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            loop = tqdm(seg_train_loader, total=len(seg_train_loader))
            for data in loop:
                seg_step += 1
                images, masks, _ = data
                # image: [从头向下, 从前向后, 从左向右]
                seg_optimizer.zero_grad()
                segmentation = seg_model(images)
                segmentation = [i[:, 0].unsqueeze(dim=1) for i in segmentation]

                # train_loss = torch.abs(seg_loss(segmentation, masks) - threshold) + threshold   # weightfocal loss
                weights = [2**(-i) for i in range(len(segmentation))]
                weights_one = torch.as_tensor([i/sum(weights) for i in weights], dtype=torch.float32, device=device)
                seg_train_loss = compute_loss(seg_loss, weights_one, segmentation, masks, device=device)
                seg_train_loss = torch.abs(seg_train_loss - seg_threshold) + seg_threshold

                seg_train_loss.backward()
                seg_optimizer.step()

                seg_loss_list.append(float(seg_train_loss.cpu()))
                if seg_step % 5 == 0 and len(seg_loss_list) >= 5:
                    writer["train"].add_scalar("segmentation loss", sum(seg_loss_list[-5:]) / 5, seg_step)

                loop.set_description(f'SEG Epoch [{epoch + 1}/{EPOCHS}]')
            
            dice = seg_val(
                seg_model, 
                seg_test_loader, 
                seg_loss, 
                "validation_Seg", 
                model_name, 
                writer["test"], 
                (epoch+1)*len(loop), 
                tag=tag, 
                device=device, 
                save=False
                )
            seg_val(seg_model, seg_train_loader, seg_loss, "train_Seg", model_name, device=device)
            
            if max_dice < dice:
                max_dice = dice
                early_stop = epoch
                torch.save(seg_model, os.path.join('/data/orfu/DeepLearning/Segmentation-Classification/deep learning/pipeline/model/seg', f"{seg_model_name}_{tag}_best.pt"))
            
    return early_stop, cls_model, seg_model, cls_train_loss, seg_train_loss, max_acc, max_dice

def cls_val(model, 
        data_loader: DataLoader, 
        cls_loss,
        tqdm_description,
        writer=None, 
        epoch=50, 
        tag="val0", 
        device="cuda",
        save=True
        ):
    model.eval()
    with torch.no_grad():
        total_loss = torch.as_tensor(0., device=device)
        loop = tqdm(data_loader, total=len(data_loader))
        loop.set_description(tqdm_description)
        pred_result, true_result, pred_prob = torch.as_tensor([], device=device), torch.as_tensor([], device=device), torch.as_tensor([], device=device)
        for data in loop:
            images, labels, paths = data
            classification = model(images)
            clsloss = cls_loss(classification, labels)
            classification = nn.Softmax(dim=1)(classification)
            pred_result = torch.concat((pred_result, torch.argmax(classification, dim=1)))
            pred_prob = torch.concat((pred_prob, classification))
            true_result = torch.concat((true_result, labels))
            total_loss += clsloss

            if save:
                cam = CAM(model, input_shape=(3, 128, 128, 128), target_layer='layer2', fc_layer='fc')
                out = model(images)
                # Retrieve the CAM by passing the class index and the model output
                activation_map = cam(out.argmax(dim=1).tolist(), out)
                activation_map = F.interpolate(activation_map[0].unsqueeze(dim=1), scale_factor=(8, 8, 8), mode="trilinear")

                for cam, path in zip(activation_map, paths):
                    if not os.path.exists(path):
                        os.makedirs(path)
                    torch.save(cam, os.path.join(path, 'cam.pt'))

        pred_result = list(map(int, pred_result.cpu()))
        pred_prob = pred_prob.cpu().tolist()
        true_result = list(map(int, true_result.cpu()))

        accuracy = acc(pred_result, true_result, writer=writer, epoch=epoch)
        if writer != None:
            writer.add_scalar("classification loss", total_loss.cpu() / len(data_loader), epoch)

    result = pd.DataFrame({"pred": pred_result, "true": true_result, "pred prob": pred_prob})
    return result, accuracy

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
        model_name="ResNet3D_nnUNet3D", 
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
                    if not os.path.exists(path):
                        os.makedirs(path)
                    torch.save(seg, os.path.join(path, 'predict_mask_pt', model_name + '_01_mask.pt'))
                for seg, path in zip(segmentation, paths):
                    seg = torch.squeeze(seg)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    torch.save(seg, os.path.join(path, 'predict_mask_pt', model_name + '_sigmoid_mask.pt'))
            total_loss += segloss

        if writer != None:
            writer.add_scalar("segmentation loss", total_loss.cpu() / len(data_loader), epoch)
            writer.add_scalar("dice", dice.cpu() / len(data_loader), epoch)

    return dice

def load_what_we_need(model_training_output_dir, checkpoint_name, device='cuda'):
    # we could also load plans and dataset_json from the init arguments in the checkpoint. Not quite sure what is the
    # best method so we leave things as they are for the moment.
    dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
    plans = load_json(join(model_training_output_dir, 'plans.json'))
    plans_manager = PlansManager(plans)

    checkpoint = torch.load(join(model_training_output_dir, checkpoint_name), map_location=torch.device(device))

    trainer_name = checkpoint['trainer_name']
    configuration_name = checkpoint['init_args']['configuration']
    inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
        'inference_allowed_mirroring_axes' in checkpoint.keys() else None

    parameters = checkpoint['network_weights']

    configuration_manager = plans_manager.get_configuration(configuration_name)
    # restore network
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')
    network = trainer_class.build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                                       num_input_channels, enable_deep_supervision=True)
    network.load_state_dict(parameters)
    network.to(device)
    return configuration_manager, inference_allowed_mirroring_axes, plans_manager, dataset_json, network, trainer_name


def get_models(
        cls_model,
        seg_model,
        modality,
        device="cuda",
):
    if modality not in ["PET", "CT", "PETCT"]:
        raise Exception(f"Wrong input: modality={modality}, must be one of 'PETCT'/'CT'/'PET'")

    cls_model = torch.load(f"/data/orfu/DeepLearning/Segmentation-Classification/deep learning/pipeline/model/pretrain/cls/{cls_model}_{modality}.pt").to(device)
    _, _, _, _, seg_model, _ = load_what_we_need(
        model_training_output_dir=f"/data/orfu/DeepLearning/Segmentation-Classification/deep learning/pipeline/model/pretrain/seg/{seg_model}",
        checkpoint_name=f"{seg_model}.pth",
        device=device,
    )

    if modality == "PETCT":
        # cls_model.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # cls_model.fc = nn.Linear(64, 4, device=device)
        cls_model.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False, device=device), 
            cls_model.conv1,
            )

        # seg_model.in_tr.conv = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        seg_model.encoder.stages[0][0].convs[0].conv = nn.Sequential(
            nn.Conv3d(3, 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), device=device),
            seg_model.encoder.stages[0][0].convs[0].conv,
            )
        seg_model.encoder.stages[0][0].convs[0].all_modules[0] = nn.Sequential(
            nn.Conv3d(3, 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), device=device),
            seg_model.encoder.stages[0][0].convs[0].all_modules[0],
            )
        # seg_model.decoder.seg_layers = nn.ModuleList(
        #     [
        #         nn.Conv3d(320, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), device=device),
        #         nn.Conv3d(256, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), device=device),
        #         nn.Conv3d(128, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), device=device),
        #         nn.Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), device=device),
        #         nn.Conv3d(32, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), device=device),
        #     ]
        # )

    return cls_model, seg_model

class MyData(Dataset):
    def __init__(self, 
                 data_label: pd.DataFrame, 
                 label_list: list,
                 mode,
                 model_name="ResNet3D_nnUNet3D",
                 device='cuda', 
                 modality="PETCT", 
                 root_dir=r'/data/orfu/DeepLearning/Segmentation-Classification/oufu_data_400G/preprocessed'
                 ):
        
        if mode not in ['cls', 'seg']:
            raise Exception(f"Wrong input: mode={mode}, must be one of 'cls'/'seg'")
        self.mode = mode

        if self.mode == 'cls':
            mode_list = ['NEGATIVE', 'LYMPHOMA', 'MELANOMA', 'LUNG_CANCER']
        else:
            mode_list = ['LYMPHOMA', 'MELANOMA', 'LUNG_CANCER']

        self.data = data_label[data_label['training_label'].isin(label_list) & data_label['class'].isin(mode_list)]
        self.data_label = data_label
        self.root_dir = root_dir
        self.device = device
        self.model_name = model_name

        if modality not in ['PETCT', 'CT', 'PET']:
            raise Exception(f"Wrong input: modality={modality}, must be one of 'PETCT'/'CT'/'PET'")
        self.modality = modality

       

    def __getitem__(self, item):
        # sex_dict = {'F': 0, 'M': 1}
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
            # mask_path = os.path.join(patient_dir, 'predict_mask_pt', self.model_name + '_sigmoid_mask.pt')
            mask_path = os.path.join(patient_dir, 'seg_pt', '1.pt')
            if os.path.exists(mask_path):
                mask = torch.load(mask_path, map_location=self.device)
                if  len(mask.shape) == 3:
                    mask = mask.unsqueeze(dim=0)
            else:
                mask = torch.full((1, 128, 128, 128), fill_value=0.5, dtype=torch.float32, device=self.device)
            image = torch.cat([image, mask], dim=0)

            return image, label, patient_dir
        
        else:
            cam_path = os.path.join(patient_dir, 'cam.pt')
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
    cls_nums_worker = 7
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

    grid_search = {
        'cls_lr': [1e-6,],
        'seg_lr': [0.01],
        'cls_WeightDecay': [2e-4],
        'seg_WeightDecay': [3e-5],
        'cls_threshold': [0.4],
        'seg_threshold': [0.],
        'epoch': [100],
    }

    cls_batch = 4
    seg_batch = 2
    cls_loss = nn.CrossEntropyLoss().to(device)
    seg_loss = loss.HybridLoss()

    train_for_best = args.train_best
    if not train_for_best:
        grid_search_path = os.path.join("./pipeline/Performance", f"{cls_model_name}_{seg_model_name}_{modality}")

        if os.path.exists(os.path.join(grid_search_path, "grid search.csv")):
            validation_df = pd.read_csv(os.path.join(grid_search_path, "grid search.csv"), index_col=0)
        else:
            validation_df = pd.DataFrame()

        combinations = list(itertools.product(*grid_search.values()))
        for combination in combinations:
            params = dict(zip(tuple(grid_search.keys()), combination))

            acc_list, dice_list = [], []

            validation = 0      # only validation on the first fold

            print(f"start training on valid {validation}")

            param_path = "_".join([str(item) for key_value in params.items() for item in key_value])
            paths = [
                os.path.join('./pipeline/log/train', f"{cls_model_name}_{seg_model_name}_{modality}", param_path, "val"+str(validation)),
                os.path.join('./pipeline/log/test', f"{cls_model_name}_{seg_model_name}_{modality}", param_path, "val"+str(validation)),
            ]

            print(paths)
            for dir in paths:
                if not os.path.exists(dir):
                    os.makedirs(dir)

            train_writer = SummaryWriter(paths[0])
            test_writer = SummaryWriter(paths[1])

            trainlist = list(range(validation))
            trainlist.extend(list(range(validation+1, kFold)))

            cls_train_data = MyData(classes, trainlist, 'cls', model_name, device, root_dir=root_dir, modality=modality)
            cls_train_dataloader = DataLoader(dataset=cls_train_data, batch_size=cls_batch, shuffle=True, drop_last=False, num_workers=cls_nums_worker, persistent_workers=cls_persistent_workers)
            seg_train_data = MyData(classes, trainlist, 'seg', model_name, device, root_dir=root_dir, modality=modality)
            seg_train_dataloader = DataLoader(dataset=seg_train_data, batch_size=seg_batch, shuffle=True, drop_last=False, num_workers=seg_nums_worker, persistent_workers=seg_persistent_workers)
            
            cls_valid_data = MyData(classes, [validation], 'cls', model_name, device, root_dir=root_dir, modality=modality)
            cls_valid_dataloader = DataLoader(dataset=cls_valid_data, batch_size=cls_batch, shuffle=False, drop_last=False, num_workers=cls_nums_worker, persistent_workers=cls_persistent_workers)
            seg_valid_data = MyData(classes, [validation], 'seg', model_name, device, root_dir=root_dir, modality=modality)
            seg_valid_dataloader = DataLoader(dataset=seg_valid_data, batch_size=seg_batch, shuffle=False, drop_last=False, num_workers=seg_nums_worker, persistent_workers=seg_persistent_workers)

            cls_model, seg_model = get_models(cls_model=cls_model_name, seg_model=seg_model_name, modality=modality, device=device)
            cls_optimizer = optim.AdamW(cls_model.parameters(), lr=params['cls_lr'], weight_decay=params['cls_WeightDecay'])    # L2 loss
            seg_optimizer = optim.AdamW(seg_model.parameters(), lr=params['seg_lr'], weight_decay=params['seg_WeightDecay'])

            early_stop, cls_model, seg_model, _, _, max_acc, max_dice = training(
                cls_model,
                seg_model,
                cls_optimizer,
                seg_optimizer,
                cls_loss,
                seg_loss,
                cls_train_dataloader,
                cls_valid_dataloader,
                seg_train_dataloader,
                seg_valid_dataloader,
                writer={"train": train_writer, "test": test_writer},
                model_name=model_name, 
                tag="val"+str(validation),
                EPOCHS=params['epoch'],
                cls_threshold=params['cls_threshold'],
                seg_threshold=params['seg_threshold'],
                device=device
                )

            acc_list.append(max_acc)
            accuracy = np.mean(np.array(acc_list))

            dice_list.append(max_dice.cpu())
            dice = np.mean(np.array(dice_list))

            params['dice'] = dice
            params['accuracy'] = accuracy
            params['epoch'] = early_stop
            validation_df = validation_df.append(params, ignore_index=True)

            if not os.path.exists(grid_search_path):
                os.makedirs(grid_search_path)
            validation_df.to_csv(os.path.join(grid_search_path, "grid search.csv"))

    # retrain for best
    try:
        grid_search_path = os.path.join("./pipeline/Performance", f"{cls_model_name}_{seg_model_name}_{modality}")
        validation_df = pd.read_csv(os.path.join(grid_search_path, "grid search.csv"), index_col=0)

        print(validation_df)
        params = validation_df.loc[validation_df['dice'].idxmax()].to_dict()
        params.pop('accuracy')
        params.pop('dice')

    except:
        combination = list(itertools.product(*grid_search.values()))[0]
        params = dict(zip(tuple(grid_search.keys()), combination))

    print(f"best param: {params}")
    param_path = "_".join([str(item) for key_value in params.items() for item in key_value])
    paths = [
        os.path.join('./pipeline/log/train', f"{cls_model_name}_{seg_model_name}_{modality}", param_path, "best"),
        os.path.join('./pipeline/log/test', f"{cls_model_name}_{seg_model_name}_{modality}", param_path, "best"),
        os.path.join("./pipeline/Performance", f"{cls_model_name}_{seg_model_name}_{modality}", param_path, "best"),
        os.path.join("./pipeline/data/output", f"{cls_model_name}_{seg_model_name}_{modality}", param_path, "best")
    ]
    for dir in paths:
        if not os.path.exists(dir):
            os.makedirs(dir)
    print(paths)

    train_writer = SummaryWriter(paths[0])
    test_writer = SummaryWriter(paths[1])
    cls_train_data = MyData(classes, list(range(1, kFold)), 'cls', model_name, device=device, root_dir=root_dir, modality=modality)
    cls_train_dataloader = DataLoader(dataset=cls_train_data, batch_size=cls_batch, shuffle=True, drop_last=False, num_workers=cls_nums_worker, persistent_workers=cls_persistent_workers)
    seg_train_data = MyData(classes, list(range(1, kFold)), 'seg', model_name, device=device, root_dir=root_dir, modality=modality)
    seg_train_dataloader = DataLoader(dataset=seg_train_data, batch_size=seg_batch, shuffle=True, drop_last=False, num_workers=seg_nums_worker, persistent_workers=seg_persistent_workers)

    cls_test_data = MyData(classes, [0], 'cls', model_name, root_dir=root_dir, device=device, modality=modality)
    cls_test_dataloader = DataLoader(dataset=cls_test_data, batch_size=cls_batch, shuffle=False, drop_last=False, num_workers=cls_nums_worker, persistent_workers=cls_persistent_workers)
    seg_test_data = MyData(classes, [0], 'seg', model_name, root_dir=root_dir, device=device, modality=modality)
    seg_test_dataloader = DataLoader(dataset=seg_test_data, batch_size=cls_batch, shuffle=False, drop_last=False, num_workers=seg_nums_worker, persistent_workers=seg_persistent_workers)

    cls_model, seg_model = get_models(cls_model=cls_model_name, seg_model=seg_model_name, modality="PETCT", device=device)
    cls_optimizer = optim.AdamW(cls_model.parameters(), lr=params['cls_lr'], weight_decay=params['cls_WeightDecay'])    # L2 loss
    seg_optimizer = optim.AdamW(seg_model.parameters(), lr=params['seg_lr'], weight_decay=params['seg_WeightDecay'])

    _, cls_model, seg_model, _, max_acc, max_dice = training(
            cls_model, 
            seg_model, 
            cls_optimizer, 
            seg_optimizer,
            cls_loss, 
            seg_loss, 
            cls_train_dataloader,
            cls_test_dataloader,
            seg_train_dataloader,
            seg_test_dataloader,
            model_name=model_name,
            writer={"train": train_writer, "test": test_writer}, 
            EPOCHS=params['epoch'], 
            cls_threshold=params['cls_threshold'],
            seg_threshold=params['seg_threshold'],
            device=device,
        )

    print("get cls test result")
    test_result, test_accuracy = cls_val(cls_model, cls_test_dataloader, "test_CAM", cls_loss)
    print("get cls train result")
    train_result, train_accuracy = cls_val(cls_model, cls_train_dataloader, "train_CAM", cls_loss)

    print("get seg test result")
    test_dice = seg_val(seg_model, seg_test_dataloader, "test_Seg", seg_loss, save=True)
    print("get seg train result")
    train_dice = seg_val(seg_model, seg_train_dataloader, "train_Seg", seg_loss, save=True)


    torch.save(cls_model, os.path.join('./pipeline/model/pipeline/cls', f"{cls_model_name}_{seg_model_name}_{modality}"+'.pt'))
    torch.save(seg_model, os.path.join('./pipeline/model/pipeline/seg', f"{cls_model_name}_{seg_model_name}_{modality}"+'.pt'))

    test_result.to_csv(os.path.join(paths[3], "test_result.csv"))
    train_result.to_csv(os.path.join(paths[3], "train_result.csv"))
    print("-"*20)
    print(f"Test accuracy: {test_accuracy}")
    print(f"Train accuracy: {train_accuracy}")
    print(f"Test dice: {test_dice}")
    print(f"Train dice: {train_dice}")
    print(f"total time: {(time.time() - start) / 60: .2f}min")


if __name__ == '__main__':
    main()
    