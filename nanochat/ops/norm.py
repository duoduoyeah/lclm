import torch.nn.functional as F


def norm(x):
    return F.rms_norm(x, (x.size(-1),))
