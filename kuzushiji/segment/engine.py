from pathlib import Path
import math
import sys
import time

from PIL import Image
import numpy as np
import torch

from . import utils
from ..data_utils import (
    to_coco, from_coco, get_image_path, SEG_FP, scale_boxes)
from ..utils import print_metrics
from ..viz import visualize_boxes
from ..metric import score_boxes, get_metrics
from .dataset import get_target_boxes_labels

import matplotlib.pyplot as plt
import torchvision


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(
            optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(
            data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #print("img:", images[0].shape)
        ###plot train loader by epoch ###
        # grid_img= list(image.to('cpu') for image in images)
        # grid_img = torchvision.utils.make_grid(grid_img, nrow=6)
        # plt.imshow(grid_img.permute(1, 2, 0))
        # plt.show()
        
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    return {k: m.global_avg for k, m in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, output_dir, threshold, mmdet=False):
    cpu_device = torch.device('cpu')
    model.eval()
    #print(model)
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'
    scores = [] #scores boxes: {'tp': tp, 'fp': fp, 'fn': fn}
    clf_gt = [] #

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        ###plot train loader by epoch ###
        # grid_img= list(image.to('cpu') for image in images)
        # grid_img = torchvision.utils.make_grid(grid_img, nrow=6)
        # plt.imshow(grid_img.permute(1, 2, 0))
        # plt.show()
        
        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        # if mmdet:
        #     outputs=model(return_loss=False, rescale=True, **data)
        # print("pppp",outputs) ## ['boxes'], ['labels'], ['scores']
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        evaluator_time = time.time()
        for target, image, output in zip(targets, images, outputs):
            item = data_loader.dataset.df.iloc[target['idx'].item()]
            #print("AA", item)
            #print("BB", target)
            del target
            target_boxes, target_labels = get_target_boxes_labels(item)
            target_boxes = torch.from_numpy(target_boxes)
            boxes = output['boxes'][output['scores'] >= threshold]
            boxes = to_coco(boxes)
            
            with Image.open(get_image_path(
                    item, data_loader.dataset.root)) as original_image:
                ow, oh = original_image.size
            _, h, w = image.shape
            w_scale = ow / w
            h_scale = oh / h
            scaled_boxes = scale_boxes(boxes, w_scale, h_scale)
            scoress= output['scores'][output['scores'] >= threshold]
            # if item['image_id']=='200021869-00003_1':
            #     print(item)
            #     print("FFFF ", len(np.ones(target_labels.shape[0])), len(np.ones(boxes.shape[0])))
            #     print("Tar", target_boxes)
            #     print("Scal",scaled_boxes)
            #     print("score",scoress)
            scores.append(
                dict(score_boxes(
                    truth_boxes=from_coco(target_boxes).numpy(),
                    truth_label=np.ones(target_labels.shape[0]),
                    preds_center=torch.stack(
                        [scaled_boxes[:, 0] + scaled_boxes[:, 2] * 0.5,
                         scaled_boxes[:, 1] + scaled_boxes[:, 3] * 0.5]
                    ).t().numpy(),
                    preds_label=np.ones(boxes.shape[0]),
                    ori_true=target_boxes,
                    ori_pred=scaled_boxes,
                    ori_scor= scoress,
                ), image_id=item.image_id))
            #print("SS:",scores)
            clf_gt.append({
                'labels': get_clf_gt(
                    target_boxes=target_boxes,
                    target_labels=target_labels,
                    boxes=scaled_boxes),
                'image_id': item.image_id,
            })
            
            
            if output_dir:
                unscaled_target_boxes = scale_boxes(
                    target_boxes, 1 / w_scale, 1 / h_scale)
                _save_predictions(
                    image, boxes, unscaled_target_boxes,
                    path=output_dir / f'{item.image_id}.jpg')

        evaluator_time = time.time() - evaluator_time
        metric_logger.update(
            model_time=model_time, evaluator_time=evaluator_time)

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    metrics = get_metrics(scores)
    print_metrics(metrics)
    return metrics, (scores, clf_gt)


def _save_predictions(image, boxes, target, path: Path):
    image = (image.detach().cpu() * 255).to(torch.uint8)
    image = np.rollaxis(image.numpy(), 0, 3)
    image = visualize_boxes(image, boxes, thickness=3)
    image = visualize_boxes(image, target, color=(0, 255, 0), thickness=2)
    Image.fromarray(image).save(path)
    

def get_clf_gt(target_boxes, target_labels, boxes, min_iou=0.5) -> str:
    """ Create ground truth for classification from predicted boxes
    in the same format as original ground truth, with addition of a class for
    false negatives. Perform matching using box IoU.
    """
    if boxes.shape[0] == 0:
        return ''
    if target_boxes.shape[0] == 0:
        labels = [SEG_FP] * boxes.shape[0]
    else:
        ious = bbox_overlaps(from_coco(target_boxes).numpy(),
                             from_coco(boxes).numpy())
        ious2 = bbox_overlaps2(from_coco(target_boxes).numpy(),
                             from_coco(boxes).numpy())
        ious_argmax = np.argmax(ious, axis=0)
        #ious_argmax2 = np.argmax(ious2, axis=0)
        #print("AAA ", ious_argmax, ious_argmax2)
        assert ious_argmax.shape == (boxes.shape[0],)
        labels = []
        for k in range(boxes.shape[0]):
            n = ious_argmax[k]
            if ious[n, k] >= min_iou:
                label = target_labels[n]
            else:
                label = SEG_FP
            labels.append(label)
    return ' '.join(
        label + ' ' + ' '.join(str(int(round(float(x)))) for x in box)
        for box, label in zip(boxes, labels))


def bbox_overlaps(
        bboxes1: np.ndarray, bboxes2: np.ndarray, mode='iou') -> np.ndarray:
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    GH:open-mmlab/mmdetection/mmdet/core/evaluation/bbox_overlaps.py

    Soft non-maximum suppression (Soft-NMS): instead of completely removing the proposals 
    with high IOU and high confidence, reduce the confidences of the proposals proportional to IOU value
    
    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious

def bbox_overlaps2(bboxes1,
                  bboxes2,
                  mode='iou',
                  eps=1e-6,
                  use_legacy_coordinate=True):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1 (ndarray): Shape (n, 4)
        bboxes2 (ndarray): Shape (k, 4)
        mode (str): IOU (intersection over union) or IOF (intersection
            over foreground)
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Default: False.
    Returns:
        ious (ndarray): Shape (n, k)
    """

    assert mode in ['iou', 'iof']
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
        bboxes1[:, 3] - bboxes1[:, 1] + extra_length)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
        bboxes2[:, 3] - bboxes2[:, 1] + extra_length)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(
            y_end - y_start + extra_length, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious
