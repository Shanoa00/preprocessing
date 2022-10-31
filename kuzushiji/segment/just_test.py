from pathlib import Path
import pandas as pd
import torch
import torch.utils.data
from torch import nn, tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from . import utils 
from .engine import train_one_epoch, evaluate
from .dataset import Dataset, get_transform
from ..data_utils import DATA_ROOT, TRAIN_ROOT, TEST_ROOT, load_train_valid_df

import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid, draw_bounding_boxes
from torchvision.transforms import ToPILImage
import torchvision.transforms.functional as F
import torchvision

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

from mmcv import Config, DictAction
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

""" 
import pickle
#train
path= '/home/mauricio/Documents/Pytorch/mmdetection/code/mmdetection/work_dirs/hr32_m/'
test = open(path+'test_result.pkl', 'rb') #val_crop
# dump information to that file
data = pickle.load(test)

print(len(data))
#print(data[0])

print('Loading data...')
# df_valid = pd.read_csv(path+'data/' + 'sample_submission.csv')
#print(df_valid)


# dataset_test = Dataset(
        # df_valid, get_transform(train=False), test_img, skip_empty=False)
#print("Test:",dataset_test.__len__())
#print(next(enumerate(dataset_test)))

# print('Creating data loaders...')
# test_sampler = torch.utils.data.SequentialSampler(dataset_test)
# data_loader_test = torch.utils.data.DataLoader(
#         dataset_test, batch_size=1,
#         sampler=test_sampler, num_workers=4,
#         collate_fn=utils.collate_fn)
# print("BBB", len(data_loader_test))

# _, eval_results = evaluate(
#             model, data_loader_test, device='cuda:0', output_dir=output_dir,
#             threshold=0.5)

# cpu_device = torch.device('cpu')
# model.eval()
# metric_logger = utils.MetricLogger(delimiter='  ')
# header = 'Test:'
# scores = [] #scores boxes: {'tp': tp, 'fp': fp, 'fn': fn}
# clf_gt = [] #

# for images, targets in metric_logger.log_every(data_loader_test, 100, header):
#     images = list(img.to('cuda:0') for img in images)
#     targets = [{k: v.to('cuda:0') for k, v in t.items()} for t in targets]
#     ###plot train loader by epoch ###
#     # grid_img= list(image.to('cpu') for image in images)
#     # grid_img = torchvision.utils.make_grid(grid_img, nrow=6)
#     # plt.imshow(grid_img.permute(1, 2, 0))
#     # plt.show()
#     print(images)       
 
# # build the model from a config file and a checkpoint file

#model=torch.load(path+model_path)
#print(model)
"""
carpeta= 'hr32_m'
path= '/home/mauricio/Documents/Pytorch/mmdetection/'
model_path=path+'code/mmdetection/work_dirs/'+carpeta+'/best_mAP_epoch_9.pth'
test_img= path+'data/test_images'
config_file = path+'code/mmdetection/configs/kuzushiji.py'
output_dir = path+'code/mmdetection/work_dirs/'+carpeta+'/results'

cfg = Config.fromfile(config_file)
# replace the ${key} with the value of cfg.key
cfg = replace_cfg_vals(cfg)
cfg = compat_cfg(cfg)

print('Loading data...')
test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

#print(cfg.data.test)
# in case the test dataset is concatenated
if isinstance(cfg.data.test, dict):
    cfg.data.test.test_mode = True
    if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(
            cfg.data.test.pipeline)
elif isinstance(cfg.data.test, list):
    for ds_cfg in cfg.data.test:
        ds_cfg.test_mode = True
    if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
        for ds_cfg in cfg.data.test:
            ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

test_loader_cfg = {
    **test_dataloader_default_args,
    **cfg.data.get('test_dataloader', {})
}

# build the dataloader
dataset = build_dataset(cfg.data.test)
#print("cc", cfg.data.test)
data_loader_test = build_dataloader(dataset, **test_loader_cfg)
print("Images: ",len(data_loader_test))
print("a", next(iter(data_loader_test))['img_metas'][0].data[0][0]['ori_filename'])

print('Loading model...')
# build the model and load checkpoint
cfg.model.train_cfg = None
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
fp16_cfg = cfg.get('fp16', None)
checkpoint = load_checkpoint(model, model_path, map_location='cpu')
# old versions did not save class info in checkpoints, this walkaround is
# for backward compatibility
if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    model.CLASSES = dataset.CLASSES
cfg.device = get_device()
model = build_dp(model, cfg.device, device_ids=[0])
#model = init_detector(config_file, model_path, device='cuda:0')
#print(model)
_, eval_results = evaluate(
            model, data_loader_test, device='cuda:0', output_dir=output_dir,
            threshold=0.5)