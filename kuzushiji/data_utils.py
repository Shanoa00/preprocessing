import re
from pathlib import Path
from typing import Union
from typing import Dict, List, Tuple

import cv2
import jpeg4py
from matplotlib.pyplot import box
import pandas as pd
import numpy as np
import torch


cv2.setNumThreads(1)
DATA_ROOT = Path(__file__).parent.parent / 'data/Nancho_dataset'
train_name= 'train_images'  #  'train_adap_morpho2_all', 'train_ori_all'
TRAIN_ROOT = DATA_ROOT / train_name
TEST_ROOT = DATA_ROOT / 'test_images' #  'test_adap_morpho2_all', 'test_adap_morpho2_all' #'test_ori_all'
#print("ZZZ", type(TRAIN_ROOT))
UNICODE_MAP = {codepoint: char for codepoint, char in
               pd.read_csv(DATA_ROOT / 'unicode_translation.csv').values}


SEG_FP = 'seg_fp'  # false positive from segmentation


def load_train_df(path=DATA_ROOT / 'train.csv'):
    df = pd.read_csv(path)
    df['labels'].fillna(value='', inplace=True)
    return df

# def add_val_id(dataf):   ##added by mau to work with datasets w/ only one book id
#     df_val= dataf.sample(frac=0.2)
#     df_val['book_id']='val'
#     df_rem=dataf[~dataf.index.isin(df_val.index)]
#     df_val2=df_rem.sample(frac=0.26)
#     df_val2['book_id']='val2'
#     dataf.loc[df_val.index]=df_val
#     dataf.loc[df_val2.index]=df_val2
#     return dataf

def add_val_id(dataf, fract=[0.2, .31, .47, 1]):
    df_val=dataf.sample(frac=0.2)
    
    for i, val in enumerate(fract):    
        df_val['book_id']='val_'+str(i+1)
        #print('val_'+str(i+1), len(df_val))
        if i==0:
            df_rem= dataf[~dataf.index.isin(df_val.index)]
            #print(len(df_rem))
        else:
            df_rem= df_rem[~df_rem.index.isin(df_val.index)]
        dataf.loc[df_val.index]=df_val
        df_val=df_rem.sample(frac=val)
    return dataf

def load_train_valid_df(fold: int, n_folds: int):
    df_pre = load_train_df() #change df->df_pre
    df_pre['book_id'] = df_pre['image_id'].apply(get_book_id) #change df->df_pre
    df = add_val_id(df_pre)  #Added by mau 
    #print(df['book_id'])
    book_ids = np.array(sorted(set(df['book_id'].values)))
    with_counts = list(zip(
        book_ids,
        df.groupby('book_id')['image_id'].agg('count').loc[book_ids].values))
    #print(with_counts)
    #"""[('100241706', 66), ('100249371', 63), ('100249376', 104), ('100249416', 58), ('100249476', 46), ('100249537', 159), ('200003076', 282), ('200003967', 73), ('200004148', 124), ('200005598', 95), ('200006663', 10), ('200014685', 60), ('200014740', 154), ('200015779', 267), ('200021637', 33), ('200021644', 84), ('200021660', 169), ('200021712', 158), ('200021763', 94), ('200021802', 105), ('200021851', 53), ('200021853', 77), ('200021869', 30), ('200021925', 33), ('200022050', 24), ('brsk', 218), ('hnsd', 500), ('umgy', 466)]"""
    with_counts.sort(key=lambda x: x[1])
    print("Data counts:",with_counts) #Check book id and how many imgs are in per book 
    valid_book_ids = [book_id for i, (book_id, _) in enumerate(with_counts)
                      if i % n_folds == fold]
    #print("valid_books",valid_book_ids)
    train_book_ids = [book_id for book_id in book_ids
                      if book_id not in valid_book_ids]
    #print(train_book_ids)
    return tuple(df[df['book_id'].isin(ids)].copy()
                 for ids in [train_book_ids, valid_book_ids])

def make_path(path: Union[str, Path]): ##Added by mau
    ppath = Path(path)  # This converts a str to a Path.  If already a Path, nothing changes.
    #print(type(path))
    return ppath

def get_image_path(item, root: Path = None) -> Path:
    if root is None:
        root = TEST_ROOT if item.image_id.startswith('test_') else TRAIN_ROOT
    root= make_path(root)
    #print(type(root))
    path = root / f'{item.image_id}.jpg'
    assert path.exists(), path
    return path


def read_image(path: Path) -> np.ndarray:
    if path.parent.name == train_name:
        np_path = get_image_np_path(path)
        if np_path.exists():
            return np.load(np_path)
    return jpeg4py.JPEG(str(path)).decode()


def get_image_np_path(path):
    return path.parent / f'{path.stem}.npy'


def get_target_boxes_labels(item):
    if item.labels:
        labels = np.array(item.labels.split(' ')).reshape(-1, 5)
    else:
        labels = np.zeros((0, 5))
    boxes = labels[:, 1:].astype(np.float)
    labels = labels[:, 0]
    #print("AAA ",labels, boxes)
    return boxes, labels


def get_encoded_classes() -> Dict[str, int]:
    classes = {SEG_FP}
    df_train = load_train_df()
    for s in df_train['labels'].values:
        x = s.split()
        classes.update(x[i] for i in range(0, len(x), 5))
    return {cls: i for i, cls in enumerate(sorted(classes))}


def get_book_id(image_id):
    book_id = re.split(r'[_-]', image_id)[0]
    m = re.search(r'^[a-z]+', book_id)
    if m:
        return m.group()
    else:
        return book_id


def to_coco(boxes: torch.Tensor) -> torch.Tensor:
    """ Convert from pytorch detection format to COCO format.
    """
    boxes = boxes.clone()
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]
    return boxes


def from_coco(boxes: torch.Tensor) -> torch.Tensor:
    """ Convert from CODO to pytorch detection format.
    """
    boxes = boxes.clone()
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    return boxes


def scale_boxes(
        boxes: torch.Tensor, w_scale: float, h_scale: float) -> torch.Tensor:
    return torch.stack([
        boxes[:, 0] * w_scale,
        boxes[:, 1] * h_scale,
        boxes[:, 2] * w_scale,
        boxes[:, 3] * h_scale,
        ]).t()


def submission_item(image_id, prediction):
    return {
        'image_id': image_id,
        'labels': ' '.join(
            ' '.join([p['cls']] +
                     [str(int(round(v))) for v in p['center']])
            for p in prediction),
    }


def get_sequences(
        boxes: List[Tuple[float, float, float, float]],
        ) -> List[List[int]]:
    """ Return a list of sequences from bounding boxes.
    """
    # TODO expand tall boxes (although this is quite rarely needed)
    boxes = np.array(boxes)
    next_indices = {}
    for i, box in enumerate(boxes):
        x0, y0, w, h = box
        x1, _ = x0 + w, y0 + h
        bx0 = boxes[:, 0]
        bx1 = boxes[:, 0] + boxes[:, 2]
        by0 = boxes[:, 1]
        by1 = boxes[:, 1] + boxes[:, 3]
        w_intersecting = (
            ((bx0 >= x0) & (bx0 <= x1)) |
            ((bx1 >= x0) & (bx1 <= x1)) |
            ((x0 >= bx0) & (x0 <= bx1)) |
            ((x1 >= bx0) & (x1 <= bx1))
        )
        higher = w_intersecting & (by0 < y0)
        higher_indices, = higher.nonzero()
        if higher_indices.shape[0] > 0:
            closest = higher_indices[np.argmax(by1[higher_indices])]
            next_indices[closest] = i
    next_indices_values = set(next_indices.values())
    starts = {i for i in range(len(boxes)) if i not in next_indices_values}
    sequences = []
    for i in starts:
        seq = [i]
        next_idx = next_indices.get(i)
        while next_idx is not None:
            seq.append(next_idx)
            next_idx = next_indices.get(next_idx)
        sequences.append(seq)
    return sequences

### Crop each image---- ######Added by Mau
#### cropImg(image) is added in dataset.py class!!!

def findLines(mask, thresh, kernel_v, iterats):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_v[0],kernel_v[1]))
    detect = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=iterats)
    cnts = cv2.findContours(detect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(mask, [c], -1, (255,255,255), -1)
    return mask

def findPoints(contorns):
    cc = np.array([[0, 0]], dtype=int)
    # iterate through every contour
    for i, c in enumerate(contorns):
        # reshape array storing contour boundary points
        c_modified = c.reshape(len(contorns[i]), 2)
        # concatenate with initialized array
        cc = np.concatenate((cc, c_modified), axis = 0)
    # avoiding first element in the initialized array
    new_cc = cc[1:]
    return list([[np.min(new_cc[:,1]), np.max(new_cc[:,1])], [np.min(new_cc[:,0]), np.max(new_cc[:,0])]])

def cropImg(image):
    extra= 20
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,27,3) #11
    mask = np.zeros(image.shape, dtype=np.uint8)
    horizontal= findLines(mask, thresh, ([80,1]), 1)
    mask = findLines(horizontal, thresh, ([1,55]), 2)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=14)
    cnts, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bottom, top= findPoints(cnts)
    cropped= image[bottom[0]-extra:bottom[1]+extra, top[0]-extra:top[1]+extra]
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)