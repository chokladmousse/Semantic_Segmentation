import numpy as np
import torch
import torch.nn as nn

def iou(pred, target, n_classes=10):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for undefined class ("9")
    for cls in range(n_classes - 1):  # last class is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = torch.sum(pred_inds & target_inds)  # PR: should we vectorize this?
        union = torch.sum(pred_inds | target_inds)
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection / union)

    return ious  #we may have to call ious.cpu()


def pixel_acc(pred, target):
    #matches = torch.zeros_like(pred) #BD: tensor.mean() complains with dtype is float/int64

    #matches[pred == target] = 1  # Check which pixels were correctly classified
    #acc = torch.mean(matches[target != 9])  # If the target was the unknown class don't include it

    pred = pred.view(-1)
    target = target.view(-1)

    return torch.mean(((pred == target) & (target != 9)).float())