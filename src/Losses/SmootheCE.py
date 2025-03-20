#TODO citation

import torch
from torch.nn import functional as F

import Parameters as pa

class SmoothCE(torch.nn.Module):

    def __init__(self, eps=0.4, trg_pad_idx=1, reduction='mean', mode = 0):
        super(SmoothCE, self).__init__()
        self.eps = eps
        self.trg_pad_idx = trg_pad_idx
        self.reduction = reduction
        self.mode = mode
        if mode == 1:
            self.criterion2 = torch.nn.CrossEntropyLoss( ignore_index=self.trg_pad_idx)

    def forward(self, pred, gold,reduc = 'mean',batches_flattened=True, weight_for_batches = None):
        if self.mode == 1 :
            loss = self.criterion2(pred, gold)
            return loss

        if batches_flattened == False:
            n_class = pred.size(2)
            one_hot = torch.zeros_like(pred,device='cuda')
            one_hot = one_hot.scatter(2, gold.view(gold.shape[0],-1, 1), 1)
            one_hot_eps = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=2)

            non_pad_mask = gold.ne(self.trg_pad_idx)
            loss = -(one_hot_eps * log_prb).sum(dim=2)
            if weight_for_batches is not None:
                for i in range(weight_for_batches.shape[0]):
                    loss[i] = loss[i] * weight_for_batches[i]


            return torch.mean(loss.masked_select(non_pad_mask))

        if self.eps >= 0.:
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred,device='cuda')
            one_hot = one_hot.scatter(1, gold.view(-1, 1), 1)
            one_hot_eps = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            non_pad_mask = gold.ne(self.trg_pad_idx)
            loss = -(one_hot_eps * log_prb).sum(dim=1)
            if reduc=='sum':
                return torch.sum(loss.masked_select(non_pad_mask))

            loss = torch.mean(loss.masked_select(non_pad_mask))

            return loss

        return None
