import torch
from torch import nn
class NeuconWLoss(nn.Module):
    """
    Equation 13 in the NeRF-W paper.
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss (1st term in equation 13)
        s_l: sigma loss (3rd term in equation 13)
    """
    def __init__(self, coef=1, igr_weight=0.1, mask_weight=0.1, depth_weight=0.1, floor_weight=0.01, config=None):
        super().__init__()
        self.coef = coef
        self.igr_weight = igr_weight
        self.mask_weight = mask_weight
        self.depth_weight = depth_weight
        self.floor_weight = depth_weight
        
        self.config = config

    def forward(self, inputs, targets, masks=None):
        ret = {}
        if masks is None:
                masks = torch.ones((targets.shape[0], 1)).to(targets.device)
        mask_sum = masks.sum() + 1e-5
        color_error = (inputs['color'] - targets) * masks
        ret['color_loss'] = torch.nn.functional.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum

        ret['normal_loss'] = self.igr_weight * inputs['gradient_error'].mean()

        if self.config.NEUCONW.MESH_MASK_LIST is not None:
            ret['mask_error'] = self.mask_weight * inputs['mask_error'].mean()

        if self.config.NEUCONW.DEPTH_LOSS:
            ret['sfm_depth_loss'] = self.depth_weight * inputs['sfm_depth_loss'].mean()

        if self.config.NEUCONW.FLOOR_NORMAL:
            ret['floor_normal_error'] = self.floor_weight * inputs['floor_normal_error'].mean()

        for k, v in ret.items():
            ret[k] = self.coef * v

        return ret

loss_dict = {'neuconw': NeuconWLoss}