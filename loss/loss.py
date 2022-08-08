from . import BaseLoss
import torch

class Loss(BaseLoss):
    def __init__(self, args):
        super(Loss, self).__init__(args)

        self.loss_name = []

        for k, _ in self.loss_dict.items():
            self.loss_name.append(k)

    def compute(self, sample, output):
        loss_val = []

        for loss_type in self.loss_dict:
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']
            if loss_func is None:
                continue

            gt = sample['gt']

            loss_tmp = loss['weight'] * loss_func(output, gt)
            loss_val.append(loss_tmp)

        loss_val = torch.stack(loss_val)
        loss_sum = torch.sum(loss_val, dim=0, keepdim=True)

        # Accumulate loss
        loss_val = torch.cat((loss_val, loss_sum))
        loss_val = torch.unsqueeze(loss_val, dim=0).detach()

        return loss_sum, loss_val