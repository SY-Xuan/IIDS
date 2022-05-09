import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftEntropy(nn.Module):
    def __init__(self):
        super(SoftEntropy, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        loss = (-F.softmax(targets, dim=1).detach() * log_probs).mean(0).sum()
        return loss


class SoftLabelLoss(nn.Module):
    def __init__(self, alpha=1., T=20):
        super(SoftLabelLoss, self).__init__()
        self.alpha = alpha
        self.T = T
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, p_logit, softlabel):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        p_logit = p_logit.view(p_logit.size(0), -1)
        log_probs = self.logsoftmax(p_logit / self.T)

        return self.T * self.alpha * self.kl_div(log_probs, softlabel)
