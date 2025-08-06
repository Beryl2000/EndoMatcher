# NOTE timm.models.layers is DEPRECATED, please use timm.layers, this is here to reduce breakages in transition
from timmvit.timm.layers.activations import *
from timmvit.timm.layers.adaptive_avgmax_pool import \
    adaptive_avgmax_pool2d, select_adaptive_pool2d, AdaptiveAvgMaxPool2d, SelectAdaptivePool2d
from timmvit.timm.layers.attention_pool2d import AttentionPool2d, RotAttentionPool2d, RotaryEmbedding
from timmvit.timm.layers.blur_pool import BlurPool2d
from timmvit.timm.layers.classifier import ClassifierHead, create_classifier
from timmvit.timm.layers.cond_conv2d import CondConv2d, get_condconv_initializer
from timmvit.timm.layers.config import is_exportable, is_scriptable, is_no_jit, set_exportable, set_scriptable, set_no_jit,\
    set_layer_config
from timmvit.timm.layers.conv2d_same import Conv2dSame, conv2d_same
from timmvit.timm.layers.conv_bn_act import ConvNormAct, ConvNormActAa, ConvBnAct
from timmvit.timm.layers.create_act import create_act_layer, get_act_layer, get_act_fn
from timmvit.timm.layers.create_attn import get_attn, create_attn
from timmvit.timm.layers.create_conv2d import create_conv2d
from timmvit.timm.layers.create_norm import get_norm_layer, create_norm_layer
from timmvit.timm.layers.create_norm_act import get_norm_act_layer, create_norm_act_layer, get_norm_act_layer
from timmvit.timm.layers.drop import DropBlock2d, DropPath, drop_block_2d, drop_path
from timmvit.timm.layers.eca import EcaModule, CecaModule, EfficientChannelAttn, CircularEfficientChannelAttn
from timmvit.timm.layers.evo_norm import EvoNorm2dB0, EvoNorm2dB1, EvoNorm2dB2,\
    EvoNorm2dS0, EvoNorm2dS0a, EvoNorm2dS1, EvoNorm2dS1a, EvoNorm2dS2, EvoNorm2dS2a
from timmvit.timm.layers.fast_norm import is_fast_norm, set_fast_norm, fast_group_norm, fast_layer_norm
from timmvit.timm.layers.filter_response_norm import FilterResponseNormTlu2d, FilterResponseNormAct2d
from timmvit.timm.layers.gather_excite import GatherExcite
from timmvit.timm.layers.global_context import GlobalContext
from timmvit.timm.layers.helpers import to_ntuple, to_2tuple, to_3tuple, to_4tuple, make_divisible, extend_tuple
from timmvit.timm.layers.inplace_abn import InplaceAbn
from timmvit.timm.layers.linear import Linear
from timmvit.timm.layers.mixed_conv2d import MixedConv2d
from timmvit.timm.layers.mlp import Mlp, GluMlp, GatedMlp, ConvMlp
from timmvit.timm.layers.non_local_attn import NonLocalAttn, BatNonLocalAttn
from timmvit.timm.layers.norm import GroupNorm, GroupNorm1, LayerNorm, LayerNorm2d
from timmvit.timm.layers.norm_act import BatchNormAct2d, GroupNormAct, convert_sync_batchnorm
from timmvit.timm.layers.padding import get_padding, get_same_padding, pad_same
from timmvit.timm.layers.patch_embed import PatchEmbed
from timmvit.timm.layers.pool2d_same import AvgPool2dSame, create_pool2d
from timmvit.timm.layers.squeeze_excite import SEModule, SqueezeExcite, EffectiveSEModule, EffectiveSqueezeExcite
from timmvit.timm.layers.selective_kernel import SelectiveKernel
from timmvit.timm.layers.separable_conv import SeparableConv2d, SeparableConvNormAct
from timmvit.timm.layers.split_attn import SplitAttn
from timmvit.timm.layers.split_batchnorm import SplitBatchNorm2d, convert_splitbn_model
from timmvit.timm.layers.std_conv import StdConv2d, StdConv2dSame, ScaledStdConv2d, ScaledStdConv2dSame
from timmvit.timm.layers.test_time_pool import TestTimePoolHead, apply_test_time_pool
from timmvit.timm.layers.trace_utils import _assert, _float_to_int
from timmvit.timm.layers.weight_init import trunc_normal_, trunc_normal_tf_, variance_scaling_, lecun_normal_

import warnings
warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
