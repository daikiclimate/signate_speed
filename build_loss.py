import torch
import torch.nn as nn


def build_loss_func(config: dict, weight: list = None):
    # criterion = nn.L1Loss()
    criterion = WightedL1Loss(distance=True)
    return criterion


class WightedL1Loss(object):
    def __init__(self, distance=False):
        self._criterion = nn.L1Loss()
        self._criterion = nn.MSELoss()
        self._distance = distance

    def __call__(self, pred, target, mask):
        batch_size = pred.shape[0]
        total_loss = torch.zeros(1).to(pred.device)

        for i in range(batch_size):
            m = mask[i]
            p = pred[i]
            valid_pred = p[m == 1, 0]
            valid_target = target[i][m == 1, 0]
            # valid_target = target[i][20:]
            loss = self._criterion(valid_pred, valid_target)
            total_loss += loss
            if self._distance:
                valid_pred_d = p[m == 1, 1]
                valid_target_d = target[i][m == 1, 1]
                loss_d = self._criterion(valid_pred_d, valid_target_d)
                total_loss += loss_d

        return total_loss
