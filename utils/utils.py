import os
import torch
import random
import numpy as np
from collections import OrderedDict

import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.feat = None
        self.grad = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.feat = out          # (B, L, C)

    def _backward_hook(self, module, grad_in, grad_out):
        self.grad = grad_out[0]  # (B, L, C)


def plot_cam(cam, title):
    cam = cam.detach().cpu().numpy()   # (L, C)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        cam.T,               # (C, L)
        aspect='auto',
        origin='lower',
        cmap='viridis',
    )
    plt.colorbar()
    plt.xlabel("Sequence index (L)")
    plt.ylabel("Channel (C)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def normalize(cam):
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    return cam

def set_seed(seed):
    """
    set seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def CutMix(sample, exchange_stego, M):
    """
    在batch维度交换两个张量的样本

    Args:
        tensor1: 形状为 (batch_size, seq_len, feature) 的张量
        tensor2: 形状为 (batch_size, seq_len, feature) 的张量
        swap_ratio: 交换的比例（0-1之间）

    Returns:
        交换后的两个张量
    """

    bbx1, bby1, bbx2, bby2 = M

    sample[bbx1:bbx2, bby1:bby2] = exchange_stego[bbx1:bbx2, bby1:bby2]
    #lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (sample.size()[-1] * sample.size()[-2]))

    #return  sample, lam
    return sample


def CutMix_Matrix(lam):
    W = 50
    H = 20
    # cut_rat = np.sqrt(1. - lam)
    cut_rat = 1. - lam
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def save_checkpoint(epoch, model_state_dict, optimizer_state_dict, loss,best_acc, prefix):
    directory = os.path.dirname(prefix)
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,  # 保持标准state_dict格式
        'optimizer_state_dict': optimizer_state_dict,
        'loss': loss,
        'checkpoint_type': 'model',
        'best_acc':best_acc
    }

    # save model
    torch.save(checkpoint, prefix + f'epoch_{epoch}.pth.tar')
    print('save beat check :' + prefix + f'epoch_{epoch}.pth.tar')


def save_model(epoch, model_state_dict, optimizer_state_dict, loss, prefix):
    directory = os.path.dirname(prefix)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 定义要保存的模块前缀
    modules_to_save = [
        'embedding',
        'position_embedding',
        'OriginalBackbone',
        'CalibrationBackbone',
        'CVIM',
        'Neck',
        'Head'
    ]
    # 创建过滤后的状态字典（保持OrderedDict结构）
    filtered_state_dict = OrderedDict()

    for name, param in model_state_dict.items():
        if any(name.startswith(module + '.') or name == module for module in modules_to_save):
            filtered_state_dict[name] = param  # 直接使用state_dict中的值

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': filtered_state_dict,  # 保持标准state_dict格式
        'optimizer_state_dict': optimizer_state_dict,
        'loss': loss,
        'saved_modules': modules_to_save,
        'checkpoint_type': 'selective_modules'
    }

    # save model
    torch.save(checkpoint, prefix + f'epoch_{epoch}_best.pth.tar')
    print('save beat check :' + prefix + f'epoch_{epoch}_best.pth.tar')