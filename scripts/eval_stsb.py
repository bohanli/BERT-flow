import os
import argparse
import json
import sys
import numpy as np
import scipy
import scipy.stats

def get_pred(fpath):
    with open(fpath) as f:
        x = [float(_) for _ in f.readlines()]
    return x

def get_gt(fpath, col, header=False):
    with open(fpath) as f:
        y = np.asarray([float(_.split('\t')[col]) for _ in f.readlines()[int(header):]])
    return y
    
def get_correlation(x, y):
    print("Pearson: %f" % pearson_r(x, y), end=", ")
    print("Spearman: %f" % scipy.stats.spearmanr(x, y).correlation)

def get_auc(pred, y):
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    print("AUC: %f" % metrics.auc(fpr, tpr))

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    corr_mat = np.corrcoef(x, y)
    return corr_mat[0, 1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--glue_path', type=str, default="../glue_data",
                        help='path to predicted sentence vectors')
    parser.add_argument('--task_name', type=str, default="sts-b",
                        help='path to predicted sentence vectors')
    parser.add_argument('--pred_path', type=str, 
                        help='path to predicted sentence vectors')
    parser.add_argument('--is_test', type=int, default=0,
                        help='eval/test set')
    args = parser.parse_args()

    x = get_pred(args.pred_path)
    if args.task_name.lower() == "sts-b":
        if args.is_test == 1:
            fpath = os.path.join(args.glue_path, "STS-B/sts-test.csv")
            y = get_gt(fpath, 4, 0)
        elif args.is_test == 0:
            fpath = os.path.join(args.glue_path, "STS-B/dev.tsv")
            y = get_gt(fpath, 9, 1)
        elif args.is_test == -1:
            fpath = os.path.join(args.glue_path, "STS-B/train.tsv")
            y = get_gt(fpath, 9, 1)
        else:
            raise NotImplementedError
    elif args.task_name.lower() == "sick-r":
        fpath = os.path.join(args.glue_path, "SICK-R/SICK_test_annotated.txt")
        y = get_gt(fpath, 3, 1)
    elif args.task_name.lower() == "mrpc-regression":
        fpath = os.path.join(args.glue_path, "MRPC-Regression/msr_paraphrase_test.txt")
        y = get_gt(fpath, 0, 1)
    else:
        raise NotImplementedError
    
    get_correlation(x, y)
    if args.task_name.lower() in ["mrpc-regression", "qnli-regression"]:
        get_auc(x, y)
        
