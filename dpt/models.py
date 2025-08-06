import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class DPT(BaseModel):

    def __init__(
        self,
        features=256,
        backbone="vitb_rn50_384",
        pretrainedif=False,
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False
    ):
        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            pretrainedif,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = _make_fusion_block(features, use_bn)

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)


        b,_,c,w,h=x.shape
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)  

        (layer_1_0, layer_1_1), (layer_2_0, layer_2_1),(layer_3_0, layer_3_1), (layer_4_0, layer_4_1) = layer_1.split(b), layer_2.split(b), layer_3.split(b), layer_4.split(b) 

        layer_1_rn0 = self.scratch.layer1_rn(layer_1_0)  
        layer_2_rn0 = self.scratch.layer2_rn(layer_2_0)  
        layer_3_rn0 = self.scratch.layer3_rn(layer_3_0)  
        layer_4_rn0 = self.scratch.layer4_rn(layer_4_0)  
        path_40 = self.scratch.refinenet4(layer_4_rn0)  
        path_30 = self.scratch.refinenet3(path_40, layer_3_rn0)  
        path_20 = self.scratch.refinenet2(path_30, layer_2_rn0)  
        path_10 = self.scratch.refinenet1(path_20, layer_1_rn0)  
        
        out_feature0=self.scratch.output_conv(path_10)  
        out_feature0 = out_feature0 / torch.norm(out_feature0, dim=1, keepdim=True)
        path_30=path_30 / torch.norm(path_30, dim=1, keepdim=True)
        path_20=path_20 / torch.norm(path_20, dim=1, keepdim=True)
        path_10=path_10 / torch.norm(path_10, dim=1, keepdim=True)



        layer_1_rn1 = self.scratch.layer1_rn(layer_1_1)  
        layer_2_rn1 = self.scratch.layer2_rn(layer_2_1)  
        layer_3_rn1 = self.scratch.layer3_rn(layer_3_1)  
        layer_4_rn1 = self.scratch.layer4_rn(layer_4_1)  
        path_41 = self.scratch.refinenet4(layer_4_rn1)  
        path_31 = self.scratch.refinenet3(path_41, layer_3_rn1)  
        path_21 = self.scratch.refinenet2(path_31, layer_2_rn1)  
        path_11 = self.scratch.refinenet1(path_21, layer_1_rn1)  

        out_feature1=self.scratch.output_conv(path_11)  
        out_feature1 = out_feature1 / torch.norm(out_feature1, dim=1, keepdim=True)
        path_31=path_31 / torch.norm(path_31, dim=1, keepdim=True)
        path_21=path_21 / torch.norm(path_21, dim=1, keepdim=True)
        path_11=path_11 / torch.norm(path_11, dim=1, keepdim=True)



        return path_30,path_31,path_20,path_21,path_10,path_11,out_feature0,out_feature1


class EndoMacher(DPT):
    def __init__(
        self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=False, **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.invert = invert

        super().__init__(**kwargs)

        if path is not None:
            self.load(path)

    def forward(self, x):
        path_30,path_31,path_20,path_21,path_10,path_11,out_feature0,out_feature1 = super().forward(x) 

        return path_30,path_31,path_20,path_21,path_10,path_11,out_feature0,out_feature1
