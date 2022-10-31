import random
from pathlib import Path
from typing import Callable

import albumentations as A
from albumentations.pytorch.transforms import ToTensor
import numpy as np
import pandas as pd
import torch.utils.data

from ..data_utils import get_image_path, read_image, get_target_boxes_labels, cropImg


def get_transform(train: bool) -> Callable:
    train_initial_size = 2048 #2000->Good results #max=1024
    crop_min_max_height = (400, 533) # (560,693) batsize:12 -> (550, 683), original(520, 653)
    crop_width = 512 ##595, batsize:12 ->595  original (575)
    crop_height = 384 #467, #batsize:12 ->467   original (447)
    if train:
        transforms = [
            ##A.CenterCrop(p=1, height=987, width=1396), No ayuda al modelo, por lo tanto se descarta su uso
            ##checar la funcion que permitia cortar el texto segun su bbox y ver donde se agrega
            ##Se encuentra en nancho_play -> data_info -> augmentations. jupyter
            A.LongestMaxSize(max_size=train_initial_size),
            A.RandomSizedCrop(
                min_max_height=crop_min_max_height,
                width=crop_width,
                height=crop_height,
                w2h_ratio=crop_width / crop_height,
            ),
            A.HueSaturationValue( 
                hue_shift_limit=7,
                sat_shift_limit=10,
                val_shift_limit=10,
            ),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.RandomContrast(limit=0.05, p=0.75),  #ambos ayudan al modelo!!
            A.RandomBrightness(limit=0.05, p=0.75),
            # A.RGBShift(always_apply=False, p=0.75, 
            #            r_shift_limit=(-20, 20), 
            #            g_shift_limit=(-20, 20), 
            #            b_shift_limit=(-20, 20)
            # ),
            #A.RandomContrast(limit=(-0.2, 0.2), p=0.5), #0.05
            #A.RandomBrightness(limit=(-0.2, 0.2), p=0.5),   ###0.05
        ]
        """ Best values: f1:95.060 ->100 epochs, train_initial_size= 1600
            Best values: f1:95.072 ->50 epochs, train_initial_size= 2000       
        train_initial_size = 2000 #1600->Good results #max=1024
    crop_min_max_height = (560,693) #batsize:12 -> (550, 683), original(520, 653)
    crop_width = 595 ##batsize:12 ->595  original (575)
    crop_height = 467 ##batsize:12 ->467   original (447)
    if train:
        transforms = [
            #A.CenterCrop(p=1, height=987, width=1396), No ayuda al modelo, por lo tanto se descarta su uso
            A.LongestMaxSize(max_size=train_initial_size),
            A.RandomSizedCrop(
                min_max_height=crop_min_max_height,
                width=crop_width,
                height=crop_height,
                w2h_ratio=crop_width / crop_height,
            ),
            A.HueSaturationValue( 
                hue_shift_limit=7,
                sat_shift_limit=10,
                val_shift_limit=10,
            ),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            # A.RGBShift(always_apply=False, p=0.75, 
            #            r_shift_limit=(-20, 20), 
            #            g_shift_limit=(-20, 20), 
            #            b_shift_limit=(-20, 20)
            # ),
            #A.RandomContrast(limit=(-0.2, 0.2), p=0.5), #0.05
            #A.RandomBrightness(limit=(-0.2, 0.2), p=0.5),   ###0.05
            A.RandomContrast(limit=0.05, p=0.75),  #ambos ayudan al modelo!!
            A.RandomBrightness(limit=0.05, p=0.75),
        ]
        """
        #########----------------------------------------------------------------------
        ########Previous transformations that didn't work:
        # A.HueSaturationValue(
        #         hue_shift_limit=(-20, 20), #7
        #         sat_shift_limit=(-20, 20), #10
        #         val_shift_limit=(-20, 20), #10
        #     ),
        #     A.RandomBrightnessContrast(
        #         brightness_limit=(-0.2, 0.2), 
        #         contrast_limit=(-0.2, 0.2), 
        #         brightness_by_max=True
        #     ),
        # A.ShiftScaleRotate(
        # p=0.50, rotate_limit=1.5,
        # scale_limit=0.05, border_mode=0
        # )        
        
    else:
        test_size = int(train_initial_size *
                        crop_height / np.mean(crop_min_max_height))
        print(f'Test image max size {test_size} px')
        transforms = [
            A.LongestMaxSize(max_size=test_size),
        ]
    transforms.extend([
        ToTensor(),
    ])
    return A.Compose(
        transforms,
        bbox_params={ #Albumentations should do to the augmented bounding boxes if their size has changed after augmentation
            'format': 'coco',
            'min_area': 0, ##If the area of a bounding box after augmentation becomes smaller than min_area, Albumentations will drop that box
            'min_visibility': 0.5, ##if the augmentation process cuts the most of the bounding box, that box won't be present in the returned list 
            'label_fields': ['labels'],
        },
    )


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, transform: Callable, root: Path,
                 skip_empty: bool):
        self.df = df
        self.root = root
        self.transform = transform
        self.skip_empty = skip_empty

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        #print(get_image_path(item, self.root))
        image = read_image(get_image_path(item, self.root))
        #image = cropImg(image)  ####Added by mau!!
        #print("ff:", image.shape)
        h, w, _ = image.shape
        bboxes, labels = get_target_boxes_labels(item)
        # clip bboxes (else albumentations fails)
        bboxes[:, 2] = (np.minimum(bboxes[:, 0] + bboxes[:, 2], w)
                        - bboxes[:, 0])
        bboxes[:, 3] = (np.minimum(bboxes[:, 1] + bboxes[:, 3], h)
                        - bboxes[:, 1])
        xy = {
            'image': image,
            'bboxes': bboxes,
            'labels': np.ones_like(labels, dtype=np.long),
        }
        xy = self.transform(**xy)
        if not xy['bboxes'] and self.skip_empty:
            return self[random.randint(0, len(self.df) - 1)]
        image = xy['image']
        boxes = torch.tensor(xy['bboxes']).reshape((len(xy['bboxes']), 4))
        # convert to pytorch detection format
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        target = {
            'boxes': boxes,
            'labels': torch.tensor(xy['labels'], dtype=torch.long),
            'idx': torch.tensor(idx),
        }
        return image, target
