import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

def unc_iou(y_score, y_true, thresh=.5):
    pred = (y_score > thresh).bool()
    target = y_true.bool()

    intersect = (pred & target).sum()
    union = (pred | target).sum()

    return intersect / union

def roc_pr(uncertainty_scores, uncertainty_labels, exclude=None):
    y_true = uncertainty_labels.flatten().detach().cpu().numpy()
    y_score = uncertainty_scores.flatten().detach().cpu().numpy()

    if exclude is not None:
        include = ~exclude.flatten().detach().cpu().numpy()
        y_true = y_true[include]
        y_score = y_score[include]

    pr, rec, tr = precision_recall_curve(y_true, y_score, drop_intermediate=True)
    fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=True)

    auroc = auc(fpr, tpr)
    aupr = average_precision_score(y_true, y_score)

    no_skill = np.sum(y_true) / len(y_true)

    return fpr, tpr, rec, pr, auroc, aupr, no_skill
