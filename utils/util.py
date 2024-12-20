
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from modules import NNDModule
nn_match = NNDModule()


def cal_loss(pred, gold, weight, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss_per_class = -(one_hot * log_prb).sum(dim=0) / gold.size(0)
        weight_loss = weight * loss_per_class
        loss = weight_loss.sum(dim=0)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

def CB_loss(labels, logits, samples_per_cls):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    beta = 0.9999
    gamma = 2.0
    no_of_classes = 4
    loss_type = "softmax"
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    if 0 in effective_num:
        effective_num[effective_num == 0] = 10000
    weights = (1.0 - beta) / np.array(effective_num) 
    weights = weights / np.sum(weights) * no_of_classes

    # labels_one_hot = F.one_hot(labels, no_of_classes).float().to(device)
    eps = 0.2
    n_class = logits.size(1)
    labels_one_hot = torch.zeros_like(logits).scatter(1, labels.view(-1, 1), 1)


    weights = torch.tensor(weights).float().to(device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        # pred = logits.softmax(dim = 1)
        # cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)

        #
        one_hot = labels_one_hot * (1 - eps) + (1 - labels_one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logits, dim=1)

        cb_loss = (-weights * (one_hot * log_prb)).sum(dim=1).mean()


    return cb_loss


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


def xyz_loss( data, rec,):
    xyz_loss_type = 0
    xyz_chamfer_weight = 0.01

    dist1, dist2 = nn_match(data.contiguous(), rec.contiguous())
    dist2 = dist2 * xyz_chamfer_weight

    # Different variants of the Chamfer distance
    if xyz_loss_type == 0: # augmented Chamfer distance
        # loss = torch.max(torch.mean(torch.sqrt(dist1), 1), torch.mean(torch.sqrt(dist2), 1))
        dist1 = torch.mean(torch.sqrt(dist1), 1)
        dist2 = torch.mean(torch.sqrt(dist2), 1)
        loss = torch.max(dist1,dist2)
        loss = torch.mean(loss)
    elif xyz_loss_type == 1:
        loss = torch.mean(torch.sqrt(dist1), 1) + torch.mean(torch.sqrt(dist2), 1)
        loss = torch.mean(loss)
    elif xyz_loss_type == 2: # used in other papers
        loss = torch.mean(dist1) + torch.mean(dist2)

    return loss


class WeightedCrossEntropy(nn.Module):
    def __init__(self, class_weights=None):
        super(WeightedCrossEntropy,self).__init__()
        self.class_weights = class_weights

    def forward(self, pred, gold):
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(pred, gold)

        return loss

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
