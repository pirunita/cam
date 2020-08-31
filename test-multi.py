import argparse
import importlib
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import voc12.dataloader

from misc import imutils, torchutils, pyutils

# Set logger
logger = logging.getLogger('CAM')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

torch.backends.cudnn.enabled=True
def get_args():
    parser = argparse.ArgumentParser(description='CAM')
    parser.add_argument('--name', type=str, default='CAM')
    parser.add_argument('--session', type=int, default=0)
    parser.add_argument('--checkpoints_path', type=str, default='checkpoints')
    
    # Environment Setting
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--voc12_root', type=str, default='VOCdevkit/VOC2012',
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
    
    # Dataset
    parser.add_argument('--train_list', default="voc12/train_aug.txt", type=str)
    parser.add_argument('--val_list', default="voc12/val.txt", type=str)
    parser.add_argument('--infer_list', default='voc12/train.txt', type=str)
    
    # Class Activation Map
    parser.add_argument('--cam_backbone', type=str, default='resnet')
    parser.add_argument('--cam_network', type=str, default='net.resnet50_cam')
    parser.add_argument('--cam_weights_name', type=str, default='res50_cam.pth')
    parser.add_argument('--cam_crop_size', type=int, default=512)
    parser.add_argument('--cam_batch_size', type=int, default=1)
    parser.add_argument('--cam_num_workers', type=int, default=os.cpu_count()//2)
    parser.add_argument('--cam_out_path', type=str, default='cam')
    #parser.add_argument('--cam_num_epoches', type=int, default=5)
    #parser.add_argument('--cam_weight_decay', type=float, default=1e-4)
    #parser.add_argument('--cam_learning_rate', type=float, default=0.1)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")
    
    return parser.parse_args()


def _work(process_id, model, dataset, args, cam_out_dir):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = torch.utils.data.DataLoader(dataset=databin, shuffle=False, num_workers=args.cam_num_workers // n_gpus, pin_memory=False)
    
    with torch.no_grad(), torch.cuda.device(process_id):
        model.cuda()
        for iter, pack in tqdm.tqdm(enumerate(data_loader)):
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']
            
            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)
            
            outputs = [model(img[0].cuda(non_blocking=True)) for img in pack['img']]
            
            strided_cam = torch.sum(torch.stack([F.interpolate(torch.unsqueeze(out, 0), strided_size, mode='bilinear', align_corners=False)[0] for out in outputs]), 0)
            
            highres_cam = [F.interpolate(torch.unsqueeze(out, 1), strided_up_size, mode='bilinear', align_corners=False) for out in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0]
            
            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_avg_pool2d(strided_cam, (1, 1)) + 1e-5
            
            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_avg_pool2d(highres_cam, (1, 1)) + 1e-5
            
            
            np.save(os.path.join(cam_out_dir, img_name + '.npy'),
                    {'keys': valid_cat, 'cam': strided_cam.cpu(), 'high_res': highres_cam.cpu().numpy()})
            
if __name__=='__main__':
    args = get_args()
    logger.info(args)
    
    # Directory
    root_dir = args.name
    session_dir = os.path.join(root_dir, str(args.session))
    
    checkpoints_dir = os.path.join(session_dir, args.checkpoints_path)
    cam_out_dir = os.path.join(session_dir, args.cam_out_path)
    pyutils.make_directory(cam_out_dir)
    
    # Setting Environment
    if args.use_cuda and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    n_gpus = torch.cuda.device_count()
    
    # Dataset
    train_dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list, voc12_root=args.voc12_root, \
                                                                    scales=args.cam_scales)
    
    train_dataset = torchutils.split_dataset(train_dataset, n_gpus)
    
    
    
    # Build model
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    print(os.path.join(checkpoints_dir, args.cam_weights_name))
    model.load_state_dict(torch.load(os.path.join(checkpoints_dir, args.cam_weights_name)), strict=True)
    model.eval()
    
    logger.info('[')
    torch.multiprocessing.spawn(fn=_work, args=(model, train_dataset, args, cam_out_dir), nprocs=n_gpus, join=True)
    logger.info(']')
    
    torch.cuda.empty_cache()
    
    
    
