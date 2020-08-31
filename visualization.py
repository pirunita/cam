import argparse
#import imageio 
import logging
import os

import cv2
import numpy as np
import torch.nn.functional as F

import tqdm
import voc12.dataloader

from misc import pyutils, imutils

# Set logger
logger = logging.getLogger('CAM')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

def get_args():
    parser = argparse.ArgumentParser(description='CAM')
    parser.add_argument('--name', type=str, default='CAM')
    parser.add_argument('--session', type=int, default=5)
    
    # Dataset
    parser.add_argument('--train_list', type=str, default='voc12/train_aug.txt')
    # Directory
    parser.add_argument('--voc12_root', type=str, default='VOCdevkit/VOC2012',
                        help="path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
    parser.add_argument('--cam_out_path', type=str, default='cam')
    parser.add_argument('--cam_scale_factor', type=float, default=0.5)
    parser.add_argument('--heatmap_path', type=str, default='heatmap')
    
    return parser.parse_args()

if __name__=='__main__':
    args = get_args()
    logger.info(args)
    
    # Directory
    root_dir = args.name
    session_dir = os.path.join(root_dir, str(args.session))
    cam_out_dir = os.path.join(session_dir, args.cam_out_path)
    heatmap_dir = os.path.join(session_dir, args.heatmap_path)
    pyutils.make_directory(heatmap_dir)
    
    # Dataset
    train_dataset = voc12.dataloader.VOC12ImageDataset(args.train_list, voc12_root=args.voc12_root, img_normal=False,to_torch=False)
    
    for train_data in tqdm.tqdm(train_dataset):
        img = train_data['img']
        
        #img = imutils.pil_rescale(img, args.cam_scale_factor, 3)
        
        height, width, _ = img.shape
        name = train_data['name']
        cam_dict = np.load(os.path.join(cam_out_dir, name + '.npy'), allow_pickle=True).item()
        cams = cam_dict['high_res']
        print('cams.', cams.shape)
        cams = np.max(cams, axis=0)
        heatmap = cams - np.min(cams)
        heatmap = heatmap / np.max(heatmap)
        heatmap = np.uint8(255 * heatmap)
        
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        final_image = cv2.addWeighted(heatmap, 0.6, img, 0.4, 0)
        
        cv2.imwrite(os.path.join(heatmap_dir, name + '.png'), final_image)
        """
        cv2.imshow('fuck', final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        
        #print('name', name, 'img', img.shape, 'cam', cams.shape)
        
    