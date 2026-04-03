import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#siglip loss
class MultiPositiveSigmoidLoss(nn.Module):
    def __init__(self, init_t=math.log(10.0), init_b=-10.0):
        super().__init__()
        #learnable temperature and bias
        self.t = nn.Parameter(torch.tensor(init_t))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, query_embs, doc_embs, labels):
        #torch.exp(self.t) to ensure the temperature scaling remains positive
        scores = (torch.matmul(query_embs, doc_embs.T) * torch.exp(self.t)) + self.b

        #calculate independent sigmoid losses
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())

        return loss
