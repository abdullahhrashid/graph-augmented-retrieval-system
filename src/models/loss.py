import torch
import torch.nn as nn
import torch.nn.functional as F

#really cool contrastive loss function, works on the basis of independent binary decisions per pair
class MultiNegativeSigmoidLoss(nn.Module):
    def __init__(self, temperature=0.07, neg_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.neg_weight = neg_weight

    def forward(self, query_embs, doc_embs):
        scores = torch.matmul(query_embs, doc_embs.T) / self.temperature

        B = scores.shape[0]
        device = scores.device

        pos_scores = scores[torch.arange(B), torch.arange(B)]

        mask = ~torch.eye(B, dtype=torch.bool, device=device)
        neg_scores = scores[mask].view(B, B - 1)

        pos_loss = -F.logsigmoid(pos_scores)
        neg_loss = -F.logsigmoid(-neg_scores).mean(dim=1)

        loss = pos_loss + self.neg_weight * neg_loss

        return loss.mean()
