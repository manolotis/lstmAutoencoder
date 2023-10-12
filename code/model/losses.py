import torch.nn as nn


class WeightedLoss(nn.Module):
    def __init__(self, loss_fn):
        super(WeightedLoss, self).__init__()
        self.loss_fn = loss_fn

    def _check_dimensions(self, outputs, targets, weights):
        if outputs.shape != targets.shape:
            raise ValueError(f"Outputs and targets must have same dimensions. "
                             f"Targets shape {targets.shape}, "
                             f"outputs shape: {outputs.shape}, "
                             f"weights shape: {weights.shape}")
        if weights is None:
            return
        if len(weights.shape) != len(targets.shape) and len(weights.shape) != len(targets.shape) - 1:
            # must be same number of dimensions, or one less.
            raise ValueError("Weight dimensions incompatible with provided input")

    def _weigh_output(self, out, weights):
        if weights is not None:
            # check if weights need an extra axis
            if len(weights.shape) == len(out.shape) - 1:
                weights = weights.unsqueeze(-1)
            out = out * weights
        return out

    def forward(self, outputs, targets, weights=None):
        self._check_dimensions(outputs, targets, weights)

        out = self.loss_fn(outputs, targets, reduction='none')
        out = self._weigh_output(out, weights)

        loss = out.mean()

        return loss
