'''
Author: Xingtong Liu, Yiping Zheng, Benjamin Killeen, Masaru Ishii, Gregory D. Hager, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2020 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or unberath@jhu.edu
'''

import torch.nn as nn
import torch




class RMSRLoss(nn.Module):
    def __init__(self, eps=1.0e-10,cutper=0.2):
        super(RMSRLoss, self).__init__()
        self.eps = eps
        self.cutper=cutper

    def forward(self, x): #       
        response_map, source_feature_1d_locations, boundaries = x

        batch_size, sampling_size, height, width = response_map.shape

        response_map = response_map / torch.sum(response_map, dim=(2, 3), keepdim=True) 

        sampled_cosine_distance = torch.gather(response_map.view(batch_size, sampling_size, height * width), 2,
                                            source_feature_1d_locations.view(batch_size, sampling_size,
                                                                                1).long())  

        sampled_boundaries = torch.gather(
            boundaries.view(batch_size, 1, height * width).expand(-1, sampling_size, -1), 2,
            source_feature_1d_locations.view(batch_size, sampling_size,
                                            1).long())

        list_loss= sampled_boundaries * -torch.log(self.eps + sampled_cosine_distance)

        k = int(list_loss.numel() * 0.2) 
        sorted_loss, _ = torch.sort(list_loss.flatten())  
        threshold = sorted_loss[k]  

        mask = (list_loss > threshold).float()  
        sampled_boundaries_sum = 1.0 + torch.sum(mask * sampled_boundaries)

        loss = torch.sum(mask * sampled_boundaries * -torch.log(self.eps + sampled_cosine_distance)) /sampled_boundaries_sum

        return loss
  
class MatchingAccuracyMetric(nn.Module):
    def __init__(self, threshold):
        super(MatchingAccuracyMetric, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        response_map, source_feature_2d_locations, boundaries = x
        batch_size, sampling_size, height, width = response_map.shape

        _, detected_target_1d_locations = \
            torch.max(response_map.view(batch_size, sampling_size, height * width), dim=2, keepdim=True)

        detected_target_1d_locations = detected_target_1d_locations.float()
        detected_target_2d_locations = torch.cat(
            [torch.fmod(detected_target_1d_locations, width),
             torch.floor(detected_target_1d_locations / width)],
            dim=2).view(batch_size, sampling_size, 2).float()

        distance = torch.norm(detected_target_2d_locations - source_feature_2d_locations,
                              dim=2, keepdim=False)
        ratio_1 = torch.sum((distance < self.threshold).float()) / (batch_size * sampling_size)
        ratio_2 = torch.sum((distance < 2.0 * self.threshold).float()) / (batch_size * sampling_size)
        ratio_3 = torch.sum((distance < 4.0 * self.threshold).float()) / (batch_size * sampling_size)
        return ratio_1, ratio_2, ratio_3
    