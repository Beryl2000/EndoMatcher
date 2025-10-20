import torch
import numpy as np
from torch.utils.data import DataLoader
import dataset  
import utils


def collate_fn(data):
    data.sort(key=lambda x: len(x[2]), reverse=False)
    colors_12=[]
    feature_1D_locations_1=[]
    feature_1D_locations_2=[]
    feature_2D_locations_1=[]
    feature_2D_locations_2=[]
    gt_heatmaps_1=[]
    gt_heatmaps_2=[]
    boundaries=[]
    boundaries_right=[]
    pair_intrinsic_matrices=[]
    real_flags = []  

    min_len = len(data[0][2]) 
    for unit in data:
        colors_12.append(unit[0])

        feature_1D_locations_1.append(unit[1][:min_len])
        feature_1D_locations_2.append(unit[2][:min_len])
        feature_2D_locations_1.append(unit[3][:min_len])
        feature_2D_locations_2.append(unit[4][:min_len])
        gt_heatmaps_1.append(unit[5][:min_len])
        gt_heatmaps_2.append(unit[6][:min_len])

        boundaries.append(unit[7])
        boundaries_right.append(unit[8])
        pair_intrinsic_matrices.append(unit[9])
        real_flags.append(unit[10])  

    colors_12 = np.array(colors_12)
    feature_1D_locations_1 = np.array(feature_1D_locations_1)
    feature_1D_locations_2 = np.array(feature_1D_locations_2)
    feature_2D_locations_1 = np.array(feature_2D_locations_1)
    feature_2D_locations_2 = np.array(feature_2D_locations_2)
    gt_heatmaps_1 = np.array(gt_heatmaps_1)
    gt_heatmaps_2 = np.array(gt_heatmaps_2)
    boundaries = np.array(boundaries)
    boundaries_right = np.array(boundaries_right)
    pair_intrinsic_matrices = np.array(pair_intrinsic_matrices)
    real_flags = np.array(real_flags) 

    return torch.tensor(colors_12), torch.tensor(feature_1D_locations_1), torch.tensor(feature_1D_locations_2), \
           torch.tensor(feature_2D_locations_1),torch.tensor(feature_2D_locations_2), torch.tensor(gt_heatmaps_1), torch.tensor(gt_heatmaps_2), \
           torch.tensor(boundaries), torch.tensor(boundaries_right), torch.tensor(pair_intrinsic_matrices),torch.tensor(real_flags)


def create_dataloaders(cfg, split_list):
    train_filenames, val_filenames, test_filenames = utils.get_color_file_names_by_bag(
        root=cfg.training_data_root,
        training_patient_id=cfg.training_patient_id,
        validation_patient_id=cfg.validation_patient_id,
        testing_patient_id=cfg.testing_patient_id
    )

    sequence_path_list = utils.get_parent_folder_names(cfg.training_data_root, id_range=cfg.id_range)

    train_dataset = dataset.MixDataset(
        image_file_names=train_filenames,
        folder_list=sequence_path_list,
        adjacent_range=cfg.adjacent_range,
        out_size=cfg.netinput_size,
        inlier_percentage=cfg.inlier_percentage,
        reprojection_error_threshold=cfg.reprojection_error_threshold,
        network_downsampling=cfg.network_downsampling,
        load_intermediate_data=cfg.load_intermediate_data,
        intermediate_data_root=cfg.precompute_root,
        sampling_size=cfg.sampling_size,
        phase="train",
        train_phase=cfg.phase,
        heatmap_sigma=cfg.heatmap_sigma,
        pre_workers=cfg.num_pre_workers,
        visible_interval=cfg.visibility_overlap,
        num_iter=cfg.num_iter,
        split_list=split_list
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_dataset = dataset.MixDataset(
        image_file_names=val_filenames,
        folder_list=sequence_path_list,
        adjacent_range=cfg.adjacent_range,
        out_size=cfg.netinput_size,
        inlier_percentage=cfg.inlier_percentage,
        reprojection_error_threshold=cfg.reprojection_error_threshold,
        network_downsampling=cfg.network_downsampling,
        load_intermediate_data=cfg.load_intermediate_data,
        intermediate_data_root=cfg.precompute_root,
        sampling_size=cfg.sampling_size,
        phase="validation",
        train_phase=cfg.phase,
        heatmap_sigma=cfg.heatmap_sigma,
        pre_workers=cfg.num_pre_workers,
        visible_interval=cfg.visibility_overlap,
        num_iter=cfg.num_iter,
        split_list=split_list
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, val_loader