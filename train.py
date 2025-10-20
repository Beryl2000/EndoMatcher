'''
Author: Xingtong Liu, Yiping Zheng, Benjamin Killeen, Masaru Ishii, Gregory D. Hager, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2020 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or unberath@jhu.edu
'''


from pathlib import Path
import torch
import numpy as np
import random
import datetime
import tqdm
import math
from omegaconf import OmegaConf
import argparse


# local import
import utils
from losses import RMSRLoss, MatchingAccuracyMetric
from models import maybe_load_pretrained, setup_model_and_optimizer,FeatureResponseGenerator
from data import create_dataloaders
from min_norm_solvers import MinNormSolver
from mgda import Variable, gradient_normalizers, compute_rmsr_loss_for_task

def load_cfg(config_file):
    cfg = OmegaConf.load(config_file)
    cfg.num_epoch = 30 if cfg.phase == "train_real" else 10
    cfg.training_data_root = Path(cfg.training_data_root)
    cfg.log_root = Path(cfg.log_root)
    now = datetime.datetime.now()
    cfg.log_root = cfg.log_root / f"log_train_{now.month}_{now.day}_{now.hour}_{now.minute}"
    cfg.log_root.mkdir(parents=True, exist_ok=True)
    cfg.precompute_root = cfg.training_data_root / "precompute_mix"
    cfg.precompute_root.mkdir(mode=0o777, parents=True, exist_ok=True)
    if cfg.trained_model_path is not None:
        cfg.trained_model_path = Path(cfg.trained_model_path)
    return cfg

def main(cfg):
    print("Training with config:", cfg)

    utils.set_random_seed(10085)
    device_ids = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_root, training_data_root, precompute_root = utils.prepare_paths(cfg)
    writer = utils.init_tensorboard(log_root)
    split_list = utils.load_dataset_split()

    train_loader, val_loader = create_dataloaders(cfg, split_list)
    model, optimizer, lr_scheduler = setup_model_and_optimizer(cfg, device_ids, train_loader)
    response_map_generator = FeatureResponseGenerator(scale=cfg.matching_scale, threshold=cfg.matching_threshold)
    RMSR_loss = RMSRLoss()
    matching_accuracy_metric = MatchingAccuracyMetric(threshold=5)

    model, epoch, step = maybe_load_pretrained(model, cfg, device_ids)
    validation_step = step
    num_epoch = cfg.num_epoch
    batch_size = cfg.batch_size

    for cur_epoch in range(epoch, num_epoch + 1):
        torch.manual_seed(10086 + cur_epoch)
        np.random.seed(10086 + cur_epoch)
        random.seed(10086 + cur_epoch)

        model.train()
        tq = tqdm.tqdm(total=len(train_loader) * batch_size,dynamic_ncols=True,ncols=40)
        
        for batch, (
                colors_12,
                feature_1D_locations_1,
                feature_1D_locations_2,
                feature_2D_locations_1,
                feature_2D_locations_2,
                gt_heatmaps_1,
                gt_heatmaps_2,
                boundaries,
                boundaries_right,
                pair_intrinsic_matrices,real_flag
        ) in \
                enumerate(train_loader):

            tq.set_description('Epoch {}, lr {:.2e}'.format(cur_epoch,lr_scheduler.get_last_lr()[0]))

            colors_12, feature_1D_locations_1, feature_1D_locations_2, \
            feature_2D_locations_1, feature_2D_locations_2, boundaries, \
            boundaries_right, pair_intrinsic_matrices, real_flag = [
                d.cuda(device=device_ids[0]) for d in (
                    colors_12, feature_1D_locations_1, feature_1D_locations_2,
                    feature_2D_locations_1, feature_2D_locations_2, boundaries,
                    boundaries_right, pair_intrinsic_matrices, real_flag
                )
            ]

            dataset_ids = [str(f) for f in real_flag.cpu().numpy()]
            tasks = list(set(dataset_ids))

            rmsr_loss_list = {}
            grads = {}

            for task_id in tasks:
                optimizer.zero_grad()
                task_mask = [
                    i for i, flag in enumerate(dataset_ids) if flag == task_id
                ]
                if not task_mask:
                    continue

                idx = torch.tensor(task_mask).cuda(device=device_ids[0])

                c12 = colors_12[idx]
                f1d_locs1 = feature_1D_locations_1[idx]
                f1d_locs2 = feature_1D_locations_2[idx]
                bd = boundaries[idx]
                bd_r = boundaries_right[idx]

                rmsrl=compute_rmsr_loss_for_task(model, c12, f1d_locs1, f1d_locs2, bd, bd_r, response_map_generator, RMSR_loss)

                rmsr_loss_list[task_id] = rmsrl.data

                rmsrl.backward()

                grads[task_id] = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads[task_id].append(
                            Variable(param.grad.data.clone(),
                                     requires_grad=False))

            gn = gradient_normalizers(grads,
                                      rmsr_loss_list,
                                      normalization_type='loss+')

            for t in grads:
                grads[t] = [g / (gn[t] + 1e-8) for g in grads[t]]

            # Compute optimal scale via MGDA
            if len(grads) == 1:
                sol = [1]
            else:
                sol, _ = MinNormSolver.find_min_norm_element([
                    grads[t] for t in tasks
                ])  

            optimizer.zero_grad()
            rmsr_loss_list2 = {}
            for i, task_id in enumerate(tasks):
                task_mask = [
                    i for i, flag in enumerate(dataset_ids) if flag == task_id
                ]
                if not task_mask:
                    continue

                idx = torch.tensor(task_mask).cuda(device=device_ids[0])

                c12 = colors_12[idx]
                f1d_locs1 = feature_1D_locations_1[idx]
                f1d_locs2 = feature_1D_locations_2[idx]
                bd = boundaries[idx]
                bd_r = boundaries_right[idx]


                rmsrl=compute_rmsr_loss_for_task(model, c12, f1d_locs1, f1d_locs2, bd, bd_r, response_map_generator, RMSR_loss)


                rmsr_loss_list2[task_id] = rmsrl.data

                if i > 0:
                    total_loss = total_loss + sol[i] * rmsrl
                else:
                    total_loss = sol[i] * rmsrl


            if math.isnan(total_loss.item()) or math.isinf(total_loss.item()):
                tq.update(batch_size)
                continue
            total_loss.backward()
            trainable_params = filter(
                lambda p: p.requires_grad and p.grad is not None,
                model.parameters())
            torch.nn.utils.clip_grad_norm_(trainable_params, 10.0)
            optimizer.step()

            mean_rmsrl = total_loss.item() if batch == 0 else (
                mean_rmsrl * batch + total_loss.item()) / (batch + 1.0)
            step += 1

            tq.update(batch_size)
            tq.set_postfix(loss='average: {:.5f}, current: {:.5f}'.format(
                mean_rmsrl, total_loss.item()))
            writer.add_scalars('Train', {'loss': mean_rmsrl}, step)
            lr_scheduler.step()

        torch.cuda.empty_cache()
        tq.close()

        # Validation
        model.eval()
        # Update progress bar
        tq = tqdm.tqdm(total=len(val_loader) * batch_size,
                       dynamic_ncols=True,
                       ncols=40)
        torch.manual_seed(10086)
        np.random.seed(10086)
        random.seed(10086)
        with torch.no_grad():
            for batch, (colors_12, feature_1D_locations_1,
                        feature_1D_locations_2, feature_2D_locations_1,
                        feature_2D_locations_2, gt_heatmaps_1, gt_heatmaps_2,
                        boundaries, boundaries_right, pair_intrinsic_matrices,
                        _) in enumerate(val_loader):
                tq.set_description('Validation Epoch {}'.format(cur_epoch))

                colors_12 = colors_12.cuda(device=device_ids[0])
                feature_1D_locations_1 = feature_1D_locations_1.cuda(
                    device=device_ids[0])
                feature_1D_locations_2 = feature_1D_locations_2.cuda(
                    device=device_ids[0])
                feature_2D_locations_1 = feature_2D_locations_1.cuda(
                    device=device_ids[0])
                feature_2D_locations_2 = feature_2D_locations_2.cuda(
                    device=device_ids[0])
                boundaries = boundaries.cuda(device=device_ids[0])
                boundaries_right = boundaries_right.cuda(device=device_ids[0])
                pair_intrinsic_matrices = pair_intrinsic_matrices.cuda(
                    device=device_ids[0])

                _, _, _, _, _, _, feature_maps_1, feature_maps_2 = model.forward(
                    colors_12)

                response_map_2 = response_map_generator([
                    feature_maps_1, feature_maps_2, feature_1D_locations_1,
                    boundaries_right
                ]) 
                response_map_1 = response_map_generator([
                    feature_maps_2, feature_maps_1, feature_1D_locations_2,
                    boundaries
                ])

                ratio_1, ratio_2, ratio_3 = matching_accuracy_metric(
                    [response_map_1, feature_2D_locations_1,
                     boundaries]) 
                ratio_4, ratio_5, ratio_6 = matching_accuracy_metric(
                    [response_map_2, feature_2D_locations_2,
                     boundaries_right])  

                accuracy_1 = 0.5 * ratio_1 + 0.5 * ratio_4
                accuracy_2 = 0.5 * ratio_2 + 0.5 * ratio_5
                accuracy_3 = 0.5 * ratio_3 + 0.5 * ratio_6

                if batch == 0:
                    mean_accuracy_1 = np.mean(accuracy_1.item())
                    mean_accuracy_2 = np.mean(accuracy_2.item())
                    mean_accuracy_3 = np.mean(accuracy_3.item())

                else:
                    mean_accuracy_1 = (mean_accuracy_1 * batch +
                                       accuracy_1.item()) / (batch + 1.0)
                    mean_accuracy_2 = (mean_accuracy_2 * batch +
                                       accuracy_2.item()) / (batch + 1.0)
                    mean_accuracy_3 = (mean_accuracy_3 * batch +
                                       accuracy_3.item()) / (batch + 1.0)

                validation_step += 1
                tq.update(batch_size)
                tq.set_postfix(
                    accuracy_1='average: {:.5f}, current: {:.5f}'.format(
                        mean_accuracy_1, accuracy_1.item()),
                    accuracy_2='average: {:.5f}, current: {:.5f}'.format(
                        mean_accuracy_2, accuracy_2.item()),
                    accuracy_3='average: {:.5f}, current: {:.5f}'.format(
                        mean_accuracy_3, accuracy_3.item()))
                writer.add_scalars(
                    'val', {
                        'accuracy_1': mean_accuracy_1,
                        'accuracy_2': mean_accuracy_2,
                        'accuracy_3': mean_accuracy_3
                    }, validation_step)
            tq.close()

            model_path_epoch = log_root / \
                               'checkpoint_model_epoch_{}_{}_{}_{}.pt'.format(cur_epoch, mean_accuracy_1, mean_accuracy_2, mean_accuracy_3)
            utils.save_model(model=model,
                             optimizer=optimizer,
                             epoch=cur_epoch + 1,
                             step=step,
                             model_path=model_path_epoch,
                             validation_loss=mean_accuracy_1)
        torch.cuda.empty_cache()
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EndoMatcher")
    parser.add_argument("--config", type=str, default="train_config.yaml", help="Path to config YAML file")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    main(cfg)