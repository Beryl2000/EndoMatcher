import torch.nn as nn
import torch
from transformers import get_cosine_schedule_with_warmup

from dpt.models import EndoMacher


def maybe_load_pretrained(model, cfg, device_ids):
    if cfg.load_trained_model:
        if cfg.trained_model_path.exists():
            print(f"Loading {cfg.trained_model_path} ...")
            pre_trained_state = torch.load(str(cfg.trained_model_path),
                                           map_location=f'cuda:{device_ids[0]}')
            step = pre_trained_state['step']
            epoch = pre_trained_state['epoch']
            model_state = model.state_dict()
            trained_model_state = {
                k: v for k, v in pre_trained_state["model"].items() if k in model_state
            }
            model_state.update(trained_model_state)
            model.load_state_dict(model_state)
            print(f"Restored model, epoch {epoch}, step {step}")
        else:
            raise OSError("No trained model detected")
    else:
        epoch, step = 0, 0
    return model, epoch, step




def setup_model_and_optimizer(cfg, device_ids, train_loader):
    model = EndoMacher(
        backbone="vitb_rn50_384",
        non_negative=True,
        pretrainedif=False,
        enable_attention_hooks=False
    )
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda(device=device_ids[0])

    if cfg.phase == "train_real":
        for param in model.module.pretrained.model.patch_embed.backbone.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=0.01
    )

    # 选择不同训练阶段的学习率调度策略
    if cfg.phase == "train_synthetic":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[], gamma=0.1
        )
    else:
        warmup_epochs = 5
        steps_per_epoch = len(train_loader)
        total_steps = cfg.num_epoch * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    return model, optimizer, lr_scheduler


class FeatureResponseGenerator(nn.Module):
    def __init__(self, scale=20.0, threshold=0.9995):
        super(FeatureResponseGenerator, self).__init__()
        self.scale = scale
        self.threshold = threshold
    def forward(self, x):
        if len(x) == 4:
            source_feature_map, target_feature_map, source_feature_1D_locations, boundaries =x
            mask = None 
        elif len(x) == 5:
            source_feature_map, target_feature_map, source_feature_1D_locations, boundaries,mask = x  
        if len(x) == 5:
            batch_size, channel, height, width = source_feature_map.shape
            _, sampling_size, _ = source_feature_1D_locations.shape
            source_feature_1D_locations = source_feature_1D_locations.view(batch_size, 1,
                                                                        sampling_size).expand(-1, channel, -1)

            sampled_feature_vectors = torch.gather(source_feature_map.view(batch_size, channel, height * width), 2,
                                                source_feature_1D_locations.long())
            sampled_feature_vectors = sampled_feature_vectors.view(batch_size, channel, sampling_size, 1,
                                                                1).permute(0, 2, 1, 3, 4).view(batch_size,
                                                                                                sampling_size,
                                                                                                channel,
                                                                                                1, 1)

            temp = [None for _ in range(batch_size)]
            for i in range(batch_size):
                temp[i] = torch.nn.functional.conv2d(input=target_feature_map[i].view(1, channel, height, width),
                                                    weight=sampled_feature_vectors[i].view(sampling_size, channel,
                                                                                            1,
                                                                                            1),
                                                    padding=0)   
            cosine_distance_map = 0.5 * torch.cat(temp, dim=0) + 0.5
            cosine_distance_map = torch.exp(self.scale * (cosine_distance_map - self.threshold)) 
            cosine_distance_map = cosine_distance_map *mask
            cosine_distance_map = cosine_distance_map / torch.sum(cosine_distance_map, dim=(2, 3), keepdim=True) 
        elif len(x) == 4:
            batch_size, channel, height, width = source_feature_map.shape
            _, sampling_size, _ = source_feature_1D_locations.shape
            source_feature_1D_locations = source_feature_1D_locations.view(batch_size, 1,
                                                                        sampling_size).expand(-1, channel, -1)

            sampled_feature_vectors = torch.gather(source_feature_map.view(batch_size, channel, height * width), 2,
                                                source_feature_1D_locations.long())
            sampled_feature_vectors = sampled_feature_vectors.view(batch_size, channel, sampling_size, 1,
                                                                1).permute(0, 2, 1, 3, 4).view(batch_size,
                                                                                                sampling_size,
                                                                                                channel,
                                                                                                1, 1)
            temp = [None for _ in range(batch_size)]
            for i in range(batch_size):
                temp[i] = torch.nn.functional.conv2d(input=target_feature_map[i].view(1, channel, height, width),
                                                    weight=sampled_feature_vectors[i].view(sampling_size, channel,
                                                                                            1,
                                                                                            1),
                                                    padding=0)  
            cosine_distance_map = 0.5 * torch.cat(temp, dim=0) + 0.5 
            cosine_distance_map = torch.exp(self.scale * (cosine_distance_map - self.threshold)) 
            cosine_distance_map = cosine_distance_map / torch.sum(cosine_distance_map, dim=(2, 3), keepdim=True)
        return cosine_distance_map

