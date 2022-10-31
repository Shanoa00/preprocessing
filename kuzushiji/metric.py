"""
https://www.kaggle.com/anokas/kuzushiji-modified-f-1-score

Python equivalent of the Kuzushiji competition metric
(https://www.kaggle.com/c/kuzushiji-recognition/)
Kaggle's backend uses a C# implementation of the same metric. This version is
provided for convenience only; in the event of any discrepancies
the C# implementation is the master version.

Tested on Python 3.6 with numpy 1.16.4 and pandas 0.24.2.

Usage:
python f1.py --sub_path [submission.csv] --solution_path [groundtruth.csv]
"""

import argparse
import multiprocessing
from typing import List, Dict

import numpy as np
import pandas as pd

from torchmetrics.detection import MeanAveragePrecision
from pprint import pprint
import torch

def score_page(preds, truth):
    """
    Scores a single page.
    Args:
        preds: prediction string of labels and center points.
        truth: ground truth string of labels and bounding boxes.
    Returns:
        True/false positive and false negative counts for the page
    """
    tp = fp = fn = 0

    truth_indices = {
        'label': 0,
        'X': 1,
        'Y': 2,
        'Width': 3,
        'Height': 4
    }
    preds_indices = {
        'label': 0,
        'X': 1,
        'Y': 2
    }

    if pd.isna(truth) and pd.isna(preds):
        return {'tp': tp, 'fp': fp, 'fn': fn}

    if pd.isna(truth):
        fp += len(preds.split(' ')) // len(preds_indices)
        return {'tp': tp, 'fp': fp, 'fn': fn}

    if pd.isna(preds):
        fn += len(truth.split(' ')) // len(truth_indices)
        return {'tp': tp, 'fp': fp, 'fn': fn}

    truth = truth.split(' ')
    if len(truth) % len(truth_indices) != 0:
        raise ValueError('Malformed solution string')
    truth_label = np.array(truth[truth_indices['label']::len(truth_indices)])
    truth_xmin = \
        np.array(truth[truth_indices['X']::len(truth_indices)]).astype(float)
    truth_ymin = \
        np.array(truth[truth_indices['Y']::len(truth_indices)]).astype(float)
    truth_xmax = truth_xmin + \
        np.array(truth[truth_indices['Width']::len(truth_indices)])\
        .astype(float)
    truth_ymax = truth_ymin + \
        np.array(truth[truth_indices['Height']::len(truth_indices)])\
        .astype(float)

    preds = preds.split(' ')
    if len(preds) % len(preds_indices) != 0:
        raise ValueError('Malformed prediction string')
    preds_label = np.array(preds[preds_indices['label']::len(preds_indices)])
    preds_x = \
        np.array(preds[preds_indices['X']::len(preds_indices)]).astype(float)
    preds_y = \
        np.array(preds[preds_indices['Y']::len(preds_indices)]).astype(float)

    return score_boxes(
        truth_boxes=np.stack(
            [truth_xmin, truth_ymin, truth_xmax, truth_ymax]).T,
        truth_label=truth_label,
        preds_center=np.stack([preds_x, preds_y]),
        preds_label=preds_label,
    )


def score_boxes(truth_boxes, truth_label, preds_center, preds_label, ori_true, ori_pred, ori_scor):
    assert isinstance(preds_label, np.ndarray)
    tp = fp = fn = 0
    fppc = det_rate = 0
    # need to handle the same edge cases here as well
    if truth_boxes.shape[0] == 0 or preds_center.shape[0] == 0:
        #print("aqui~~~")
        fp += preds_center.shape[0]
        fn += truth_boxes.shape[0]
        return {'fppc':float(0), 'det rate':float(0), 'tp':tp, 'fp':fp, 'fn':fn, 'map':float(0), 'map_50':float(0) }

    # if len(ori_true)<70:
    #     print("TT:",ori_true)
    #     print("PP:",ori_pred)
    #     print("TL:",len(ori_true),"PL:",len(preds_label))
    maps= map_score(ori_pred,ori_scor,ori_true)
    #     print(maps)
    
    preds_x = preds_center[:, 0]
    preds_y = preds_center[:, 1]
    truth_xmin, truth_ymin, truth_xmax, truth_ymax = truth_boxes.T
    preds_unused = np.ones(len(preds_label)).astype(bool)
    for xmin, xmax, ymin, ymax, label in zip(
            truth_xmin, truth_xmax, truth_ymin, truth_ymax, truth_label):
        # Matching = point inside box & character same &
        # prediction not already used
        # print(("AA: ",(xmin < preds_x) , (xmax , preds_x) ,
        #             (ymin , preds_y) , (ymax , preds_y) ,
        #             (preds_label, label) , preds_unused))
        # print("")
        matching = ((xmin < preds_x) & (xmax > preds_x) &
                    (ymin < preds_y) & (ymax > preds_y) &
                    (preds_label == label) & preds_unused)
        #print("BB ",(matching))
        #print("BB ",(matching.sum()))
        if matching.sum() == 0:
            fn += 1
        else:
            tp += 1
            preds_unused[np.argmax(matching)] = False
    fp += preds_unused.sum()
    fppc = (fp)/(tp+fn) #(tp + fp - (tp))/(tp+fn)
    det_rate= tp/(tp+fn) #accuracy tp/(tp+fp+fn) 
    fppc= float("{0:.5f}".format(fppc)) #False positives per character
    det_rate= float("{0:.5f}".format(det_rate)) #detection rate (accuracy)
    # print("aa",{'fppc':fppc, 'det rate':det_rate, 'tp':tp, 'fp':fp, 'fn':fn, 
    #         'map':float(maps['map'].clone().detach().numpy()), 'map_50':float(maps['map_50'].clone().detach().numpy())})
    
    return {'fppc':fppc, 'det rate':det_rate, 'tp':tp, 'fp':fp, 'fn':fn, 
            'map':float(maps['map'].clone().detach().numpy()), 'map_50':float(maps['map_50'].clone().detach().numpy())}

def map_score(bbox_pred, ori_scor, bbox_target):
    #https://torchmetrics.readthedocs.io/en/latest/detection/mean_average_precision.html
    t_labelss= torch.zeros(bbox_target.shape[0])
    p_labelss= torch.zeros(bbox_pred.shape[0])
    p_scoress= ori_scor
    preds = [
    dict(
        boxes=bbox_pred,
        scores=p_scoress,
        labels=p_labelss,
    )
    ]
    target = [
    dict(
        boxes=bbox_target,
        labels=t_labelss,
    )
    ]
    metric = MeanAveragePrecision(box_format='xywh')
    metric.update(preds, target)
    #pprint(metric.compute())
    maps={'map': metric.compute()['map'], 'map_50':metric.compute()['map_50']}
    return maps

def kuzushiji_f1(sub, solution):
    """
    Calculates the competition metric.
    Args:
        sub: submissions, as a Pandas dataframe
        solution: solution, as a Pandas dataframe
    Returns:
        f1 score
    """
    if not all(sub['image_id'].values == solution['image_id'].values):
        raise ValueError("Submission image id codes don't match solution")

    pool = multiprocessing.Pool()
    results = pool.starmap(
        score_page, zip(sub['labels'].values, solution['labels'].values))
    pool.close()
    pool.join()

    return get_metrics(results)['f1']


def get_metrics(results: List[Dict]) -> Dict:
    #print("RR",results)
    #print("len",len(results['map']))
    tp = int(sum([x['tp'] for x in results]))
    fp = int(sum([x['fp'] for x in results]))
    fn = int(sum([x['fn'] for x in results]))
    maps = float(sum([x['map'] for x in results]))
    maps_50 = float(sum([x['map_50'] for x in results]))
    if (tp + fp) == 0 or (tp + fn) == 0:
        f1 = 0
        precision = 0
        fppc = 0
        det_rate= 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fppc = (fp)/(tp+fn)#(tp + fp - (tp))/(tp+fn)
        det_rate= tp/(tp+fn) #tp/(tp+fp+fn)
        if precision > 0 and recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0
    return {'fppc':float(fppc), 'det rate':float(det_rate), 'f1': float(f1), 'prec':float(precision),
            'map':maps/len(results), 'map_50':maps_50/len(results), 'tp': tp, 'fp': fp, 'fn': fn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    shell_args = parser.parse_args()
    sub = pd.read_csv(shell_args.sub_path)
    solution = pd.read_csv(shell_args.solution_path)
    sub = sub.rename(
        columns={'rowId': 'image_id', 'PredictionString': 'labels'})
    solution = solution.rename(
        columns={'rowId': 'image_id', 'PredictionString': 'labels'})
    score = kuzushiji_f1(sub, solution)
    print('F1 score of: {0}'.format(score))


if __name__ == '__main__':
    main()
