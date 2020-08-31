import argparse
import importlib
import logging
import os

import torch

import torch.nn.functional as F

import tqdm
import voc12.dataloader

from misc import pyutils
from torchvision.utils import save_image
import pdb
# Set logger
logger = logging.getLogger('Wavelet')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
def get_args():
    parser = argparse.ArgumentParser(description='Wavelet')
    parser.add_argument('--name', type=str, default='Wavelet')
    parser.add_argument('--wavelet_layer', type=str, default='net.wavelet')
    
    # Dataset
    parser.add_argument('--train_list', default='voc12/train_aug.txt', type=str)
    
    # Directory
    parser.add_argument('--voc12_root', type=str, default='VOCdevkit/VOC2012',
                        help="path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
    
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logger.info(args)
    
    # Directory
    root_dir = args.name
    pyutils.make_directory(root_dir)
    

        
    # Dataset
    train_dataset = voc12.dataloader.VOC12ImageDataset(args.train_list, voc12_root=args.voc12_root, img_normal=False,to_torch=True)
    
    
    for idx, train_data in tqdm.tqdm(enumerate(train_dataset), total=len(train_dataset)):
        img = torch.from_numpy(train_data['img']).float()
        name = train_data['name']
        _, height, width  = img.shape
        
        #feature = torch.nn.AvgPool2d(kernel_size=2)(img)
        HH = img - F.interpolate(F.avg_pool2d(img, kernel_size=4).unsqueeze(0) ,size=[height, width], mode='nearest').squeeze()
        pdb.set_trace()
        #LL = F.sigmoid(HH)
        #save_image(HH, os.path.join(root_dir, name + '.png'))
        
        
        