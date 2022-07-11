import torch
from kornia.losses import ssim as dssim
import lpips

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = dssim(image_pred, image_gt, 3, reduction) # dissimilarity in [0, 1]
    return 1-2*dssim_ # in [-1, 1]

def lpips_loss(image_pred, image_gt, net='alex'):
    """
    image_pred and image_gt: (1, 3, H, W)
    net: choice: ['alex', 'vgg', 'squeeze'] see https://github.com/richzhang/PerceptualSimilarity/blob/8db312a14945977ab00268f95499d9fe5327cc36/lpips/lpips.py#L41-L49
    """
    loss_fn = lpips.LPIPS(net=net)
    with torch.no_grad():
        loss = loss_fn(image_pred, image_gt)
    return loss