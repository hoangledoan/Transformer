import torch
from torch.nn.functional import log_softmax
from torch import nn


class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing):
        super().__init__()
        self.smoothing = smoothing
        assert 0 <= self.smoothing < 1

    def _smooth_one_hot(self, targets: torch.Tensor, n_classes):
        with torch.no_grad():
            targets = torch.empty(size=tuple(targets.size()) + (n_classes,), device=targets.device).fill_(
                self.smoothing / (n_classes - 1)).scatter_(-1, targets.data.unsqueeze(-1), 1. - self.smoothing)
        return targets

    def forward(self, logits, targets, mask = None, lengths = None):
        if lengths is None:
            lengths = torch.tensor([logits.shape[-1]])
        
        if mask is None:
            mask = torch.tensor([1])

        targets_one_hot = self._smooth_one_hot(targets, logits.shape[-1])
        loss = log_softmax(logits, -1)
        targets_one_hot.to("cuda")
        mask.to("cuda")
        loss = (- loss * targets_one_hot * mask.unsqueeze(-1)) / lengths[..., None, None]
        return torch.sum(loss) / len(lengths)
