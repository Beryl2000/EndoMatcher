import torch
import numpy as np
import torch.nn.functional as F


def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn

def with_metaclass(meta: type, *bases) -> type:
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(meta):  # type: ignore[misc, valid-type]
        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)

        @classmethod
        def __prepare__(cls, name, this_bases):
            return meta.__prepare__(name, bases)

    return type.__new__(metaclass, "temporary_class", (), {})

class VariableMeta(type):
    def __instancecheck__(cls, other):
        return isinstance(other, torch.Tensor)

class Variable(with_metaclass(VariableMeta, torch._C._LegacyVariableBase)):  # type: ignore[misc]
    pass



def compute_rmsr_loss_for_task(model, c12, f1d_locs1, f1d_locs2, bd, bd_r, response_map_generator, RMSR_loss):
    """
    Recreates the same multi-scale RMSR computation used in original script for a single task batch subset.
    Returns a scalar tensor (loss).
    """
    # Forward pass (keeps same forward outputs and naming as original)
    f_30, f_31, f_20, f_21, f_c1_task, f_c2_task, feat_map_1, feat_map_2 = model.forward(c12)

    def to_coarse_1D_locations(feature_1D, W_orig, H_orig, W_target, H_target):
        y_orig = feature_1D.view(feature_1D.size(0), -1) // W_orig
        x_orig = feature_1D.view(feature_1D.size(0), -1) % W_orig
        y_target = (y_orig * H_target) // H_orig
        x_target = (x_orig * W_target) // W_orig
        return (y_target * W_target + x_target).unsqueeze(-1)

    # Level: c1 (use f_c1_task / f_c2_task)
    H_orig, W_orig = feat_map_1.size(2), feat_map_1.size(3)
    H_target, W_target = f_c1_task.size(2), f_c1_task.size(3)
    cl1 = to_coarse_1D_locations(f1d_locs1, W_orig, H_orig, W_target, H_target)
    cl2 = to_coarse_1D_locations(f1d_locs2, W_orig, H_orig, W_target, H_target)
    bdr_interp = F.interpolate(bd_r, size=(H_target, W_target), mode='bilinear', align_corners=False)
    bd_interp = F.interpolate(bd, size=(H_target, W_target), mode='bilinear', align_corners=False)
    rsp_c1 = response_map_generator([f_c2_task, f_c1_task, cl2, bd_interp])
    rsp_c2 = response_map_generator([f_c1_task, f_c2_task, cl1, bdr_interp])
    rmsrl_10 = RMSR_loss([rsp_c1, cl1, bd_interp])
    rmsrl_11 = RMSR_loss([rsp_c2, cl2, bdr_interp])

    # Level: 20 (use f_20 / f_21)
    H_target, W_target = f_20.size(2), f_20.size(3)
    cl1 = to_coarse_1D_locations(f1d_locs1, W_orig, H_orig, W_target, H_target)
    cl2 = to_coarse_1D_locations(f1d_locs2, W_orig, H_orig, W_target, H_target)
    bdr_interp = F.interpolate(bd_r, size=(H_target, W_target), mode='bilinear', align_corners=False)
    bd_interp = F.interpolate(bd, size=(H_target, W_target), mode='bilinear', align_corners=False)
    rsp_c1 = response_map_generator([f_21, f_20, cl2, bd_interp])
    rsp_c2 = response_map_generator([f_20, f_21, cl1, bdr_interp])
    rmsrl_20 = RMSR_loss([rsp_c1, cl1, bd_interp])
    rmsrl_21 = RMSR_loss([rsp_c2, cl2, bdr_interp])

    # Level: 30 (use f_30 / f_31)
    H_target, W_target = f_30.size(2), f_30.size(3)
    cl1 = to_coarse_1D_locations(f1d_locs1, W_orig, H_orig, W_target, H_target)
    cl2 = to_coarse_1D_locations(f1d_locs2, W_orig, H_orig, W_target, H_target)
    bdr_interp = F.interpolate(bd_r, size=(H_target, W_target), mode='bilinear', align_corners=False)
    bd_interp = F.interpolate(bd, size=(H_target, W_target), mode='bilinear', align_corners=False)
    rsp_c1 = response_map_generator([f_31, f_30, cl2, bd_interp])
    rsp_c2 = response_map_generator([f_30, f_31, cl1, bdr_interp])
    rmsrl_30 = RMSR_loss([rsp_c1, cl1, bd_interp])
    rmsrl_31 = RMSR_loss([rsp_c2, cl2, bdr_interp])

    # Fine feature maps (feat_map_1 / feat_map_2)
    rsp_1 = response_map_generator([feat_map_2, feat_map_1, f1d_locs2, bd])
    rsp_2 = response_map_generator([feat_map_1, feat_map_2, f1d_locs1, bd_r])
    rmsrl_1 = RMSR_loss([rsp_1, f1d_locs1, bd])
    rmsrl_2 = RMSR_loss([rsp_2, f1d_locs2, bd_r])

    # Weighted combination (keeps identical weights as original)
    rmsrl = (0.5 * rmsrl_30 + 0.5 * rmsrl_31) / 8 + \
            (0.5 * rmsrl_20 + 0.5 * rmsrl_21) / 4 + \
            (0.5 * rmsrl_10 + 0.5 * rmsrl_11) / 2 + \
            (0.5 * rmsrl_1 + 0.5 * rmsrl_2)
    return rmsrl

