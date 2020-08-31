import argparse
import importlib
import logging
import os

import torch
import tqdm
import torch.nn.functional as F

import voc12.dataloader

from tensorboardX import SummaryWriter
from misc import torchutils
from misc import pyutils

# Set logger
logger = logging.getLogger('CAM')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

torch.backends.cudnn.enabled = True
    

def get_args():
    parser = argparse.ArgumentParser(description='CAM')
    parser.add_argument('--name', type=str, default='CAM')
    parser.add_argument('--session', type=int, default=10)
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--checkpoints_path', type=str, default='checkpoints')
    parser.add_argument('--tensorboard_path', type=str, default='tensorboard')
    # Environment setting
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--voc12_root", type=str, default='VOCdevkit/VOC2012',
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
    

    # Dataset
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    
    # Class Activation Map
    parser.add_argument('--cam_backbone', type=str, default='resnet')
    parser.add_argument('--cam_network', type=str, default='net.resnet50_cam')
    parser.add_argument('--cam_weights_name', type=str, default='res50_cam.pth')
    parser.add_argument('--cam_scale_factor', type=float, default=1)
    parser.add_argument('--cam_crop_size', type=int, default=512)
    parser.add_argument('--cam_batch_size', type=int, default=16)
    parser.add_argument('--cam_num_epoches', type=int, default=5)
    parser.add_argument('--cam_weight_decay', type=float, default=1e-4)
    parser.add_argument('--cam_learning_rate', type=float, default=0.1)
    
    
    return parser.parse_args()


if __name__=='__main__':
    args = get_args()
    logger.info(args)
    
    # Directory
    root_dir = args.name
    session_dir = os.path.join(root_dir, str(args.session))
    log_dir = os.path.join(session_dir, args.log_path)
    checkpoints_dir = os.path.join(session_dir, args.checkpoints_path)
    tensorboard_dir = os.path.join(session_dir, args.tensorboard_path)
    pyutils.make_directory(root_dir, session_dir, log_dir, checkpoints_dir, tensorboard_dir)
    
    # Setting Environment
    if args.use_cuda and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    
    # Dataset
    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root, \
                                                                resize_long=(320, 640), hor_flip=True, \
                                                                crop_size=512, crop_method='random')
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.cam_batch_size, \
                                                   shuffle=True, num_workers=args.num_workers, \
                                                   pin_memory=True, drop_last=True)
    
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches
    
    # Visualization
    log_file_dir = os.path.join(log_dir, '{}_{}.txt'.format(args.session, args.cam_backbone))
    log_writer = open(log_file_dir, 'a')
    log_writer.write(str(args) + '\n')
    board = SummaryWriter(logdir=os.path.join(tensorboard_dir, ))
    
    # Build model
    model = getattr(importlib.import_module(args.cam_network), 'Net')()
    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)
    avg_meter = pyutils.AverageMeter()
    
    model.to(device)
    model.train()
    
    for epoch in range(args.cam_num_epoches):
        logger.info('Epoch %d/%d' % (epoch + 1, args.cam_num_epoches))
        train_loss_sum = 0
        train_iter_sum = 0
        for step, pack in tqdm.tqdm(enumerate(train_dataloader)):
            img = pack['img'].cuda()
            #img = F.interpolate(img, scale_factor=args.cam_scale_factor, mode='bilinear', align_corners=True)
            label = pack['label'].cuda(non_blocking=True)
            
            x = model(img)
            
            loss = F.multilabel_soft_margin_loss(x, label)
            
            avg_meter.add({'loss1': loss.item()})
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_iter_sum += 1
            
            if (optimizer.global_step - 1) % 100 == 0:
                avg_loss = avg_meter.pop('loss1')
                logger.info('step:%5d/%5d, loss: %.4f, lr:%.4f' %(optimizer.global_step-1, max_step, avg_loss, optimizer.param_groups[0]['lr']))
                log_writer.write('step:%5d/%5d, loss: %.4f, lr:%.4f \n' %(optimizer.global_step-1, max_step, avg_loss, optimizer.param_groups[0]['lr']))
                board.add_scalar('Train/loss', avg_loss, (optimizer.global_step - 1) / 100)
                log_writer.flush()
        
        checkpoint_name = os.path.join(checkpoints_dir, '{}_{}.pth'.format(args.session,epoch))
        torchutils.save_checkpoint(model.state_dict(), checkpoint_name)
    # MultiGPU model.module.state_dict(), SingleGPU model.state_dict()
    checkpoint_name = os.path.join(checkpoints_dir, args.cam_weights_name)
    torchutils.save_checkpoint(model.state_dict(), checkpoint_name)
    torch.cuda.empty_cache()
