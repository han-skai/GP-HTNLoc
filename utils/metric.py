# import pandas as pd
import numpy as np
import torch
# import torch.nn.functional as F
from sklearn.metrics import hamming_loss as hl
from sklearn.metrics import average_precision_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import accuracy_score

def average_precision(Outputs, target):
    # remove samples with all 0 or 1 targets
    non_all_zeros = ((target == 1).sum(dim=1) > 0) & ((target == 0).sum(dim=1) < target.size(1))
    Outputs = Outputs[non_all_zeros]
    target = target[non_all_zeros]

    # compute average precision for each sample
    avg_precisions = []
    for i in range(Outputs.size(0)):
        # get targets and outputs for this sample
        sample_target = target[i]
        sample_output = Outputs[i]

        # sort predictions by descending order of confidence
        sorted_output, sorted_indices = torch.sort(sample_output, descending=True)
        sample_target = sample_target[sorted_indices]

        # compute precision at each threshold
        tp = sample_target.cumsum(dim=0)
        fp = (1-sample_target).cumsum(dim=0)
        precision = tp.float() / (tp + fp).float()

        # compute average precision
        recall = tp.float() / sample_target.sum().float()
        ap = ((precision * sample_target.float()).sum() / sample_target.sum().float()).item()
        avg_precisions.append(ap)

    # compute mean of average precisions over all samples
    mean_avg_precision = torch.tensor(avg_precisions).mean().item()

    return mean_avg_precision


def hamming_loss(Pre_Labels, test_target):

    Pre_Labels = Pre_Labels.int()
    test_target = test_target.int()

    result = 0

    test_target[test_target != 1] = 0
    Pre_Labels[Pre_Labels != 1] = 0
    num_samples = test_target.size(0)

    for i in range(num_samples):
        Y_i = test_target[i, :]
        Y_hat_i = Pre_Labels[i, :]

        result_i = (Y_i | Y_hat_i).sum().item() - (Y_i & Y_hat_i).sum().item()

        if torch.isnan(torch.tensor(result_i)):
            result_i = 0
        result += result_i

    result = result / num_samples

    return result

def one_error(Output, target):
    """
    1.We remove samples with all-zero or all-one labels using the same code as before.
    2.We replace Output and target with temp_Output and temp_target.
    3.We compute the maximum predicted value and its corresponding index using torch.max. We then use advanced indexing
    to select the corresponding true labels (target[torch.arange(target.shape[0]), max_idxs]) and compare them to 1. We
    also check if the maximum predicted value is greater than 0.5. If both of these conditions are true, we consider the
    prediction to be correct. We then compute the mean of this binary indicator over all samples and subtract it from 1
    to obtain the one-error.
    """
    # Step 1: Remove samples with all-zero or all-one labels
    is_all_zero = torch.all(target == 0, dim=1)
    is_all_one = torch.all(target == 1, dim=1)    # 去除全0或全1的样本
    valid_samples = ~(is_all_zero | is_all_one)
    temp_Output = Output[valid_samples]
    temp_target = target[valid_samples]

    # Step 2: Replace Output and target with temp_Output and temp_target
    Output = temp_Output
    target = temp_target

    # Step 3: Compute one-error
    max_preds, max_idxs = torch.max(Output, dim=1)
    correct_preds = (target[torch.arange(target.shape[0]), max_idxs] == 1) & (max_preds > 0.5)
    one_error = 1 - correct_preds.float().mean().item()
    return one_error




def accuracy(Y_hat, Y):  
    Y[Y != 1] = 0
    Y_hat[Y_hat != 1] = 0

    num_samples = Y.shape[0]
    result = 0
    for i in range(num_samples):
        Y_i = Y[i, :]
        Y_hat_i = Y_hat[i, :]

        intersect = np.logical_and(Y_i, Y_hat_i)
        union = np.logical_or(Y_i, Y_hat_i)
        result_i = np.count_nonzero(intersect) / np.count_nonzero(union)
        if np.isnan(result_i):
            result_i = 0
        result += result_i

    result = result / num_samples
    return result



def ranking_loss(Outputs, target):
    RankingLoss = label_ranking_loss(target, Outputs)
    return RankingLoss


def coverage(Outputs, test_target):
    # Outputs: predicted outputs of the classifier, shape (num_instances, num_classes)
    # test_target: actual labels of the test instances, shape (num_instances, num_classes)

    num_instances, num_classes = Outputs.shape

    Label = [list(np.where(test_target[i, :] == 1)[0]) for i in range(num_instances)]
    # not_Label = [list(np.where(test_target[i, :] == -1)[0]) for i in range(num_instances)]
    Label_size = np.array([len(label) for label in Label])

    cover = 0
    for i in range(num_instances):
        temp = Outputs[i, :]
        index = np.argsort(temp)
        temp_min = num_classes + 1
        for m in range(Label_size[i]):
            loc = np.where(index == Label[i][m])[0][0]
            if loc < temp_min:
                temp_min = loc
        cover += (num_classes - temp_min + 1)

    Coverage = (cover / num_instances) - 1
    return Coverage

