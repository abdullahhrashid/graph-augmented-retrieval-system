import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#siglip loss with class-imbalance correction
class MultiPositiveSigmoidLoss(nn.Module):
    def __init__(self, init_t=math.log(10.0), init_b=0.0):
        super().__init__()
        #learnable temperature and bias
        self.t = nn.Parameter(torch.tensor(init_t))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, query_embs, doc_embs, labels):
        #torch.exp(self.t) to ensure the temperature scaling remains positive
        scores = (torch.matmul(query_embs, doc_embs.T) * torch.exp(self.t)) + self.b

        #dynamic pos_weight to handle extreme class imbalance
        num_pos = labels.sum().clamp(min=1.0)
        num_neg = labels.numel() - num_pos
        pos_weight = (num_neg / num_pos).detach()

        loss = F.binary_cross_entropy_with_logits(
            scores, labels.float(),
            pos_weight=pos_weight,
        )

        return loss

