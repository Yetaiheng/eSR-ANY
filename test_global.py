#测试重参数化后的全局测试的代码，也可以测试未重参数化的
from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
from datas.benchmark import Benchmark
from datas.div2k import DIV2K
from models.ecbsr import ECBSR
from models.ada_conv_ECB import ECBSR_5_ada
from torch.utils.data import DataLoader
from collections import OrderedDict
from models.re_param_all import ECBSR_re, ECBSR_5_ada_share_re,ECBSR_5_ada_share_local_re,ECBSR_train
from models.ada_conv_ECB_share_local import ECBSR_5_ada_share_local
from models.ecbsr_5 import ECBSR_5_one,ECBSR_5_one_scale
from models.ecbsr_5_re import ECBSR_5_one_re

import math
import argparse, yaml
import utils
import os
import cv2
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '1234'
from tqdm import tqdm
import logging
import sys
from PIL import Image
import numpy as np
import time
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description='ECBSR')

## yaml configuration files
parser.add_argument('--config', type=str, default='./configs/ecbsr_x2_m4c8_prelu.yml', help = 'pre-config file for training')

## paramters for ecbsr
parser.add_argument('--scale', type=int, default=2, help = 'scale for sr network')
parser.add_argument('--colors', type=int, default=1, help = '1(Y channls of YCbCr)')
parser.add_argument('--m_ecbsr', type=int, default=4, help = 'number of ecb')
parser.add_argument('--c_ecbsr', type=int, default=8, help = 'channels of ecb')
parser.add_argument('--idt_ecbsr', type=int, default=0, help = 'incorporate identity mapping in ecb or not')
parser.add_argument('--act_type', type=str, default='prelu', help = 'prelu, relu, splus, rrelu')
parser.add_argument('--pretrain', type=str, default= None, help = 'path of pretrained model')

## parameters for model training
parser.add_argument('--patch_size', type=int, default=64, help = 'patch size of HR image')
parser.add_argument('--batch_size', type=int, default=32, help = 'batch size of training data')
parser.add_argument('--data_repeat', type=int, default=1, help = 'times of repetition for training data')
parser.add_argument('--data_augment', type=int, default=1, help = 'data augmentation for training')
parser.add_argument('--epochs', type=int, default=600, help = 'number of epochs')
parser.add_argument('--test_every', type=int, default=1, help = 'test the model every N epochs')
parser.add_argument('--log_every', type=int, default=1, help = 'print log of loss, every N steps')
parser.add_argument('--log_path', type=str, default="./experiments/")
parser.add_argument('--lr', type=float, default=30e-4, help = 'learning rate of optimizer')#学习率翻倍
parser.add_argument('--store_in_ram', type=int, default=0, help = 'store the whole training data in RAM or not')

## hardware specification
parser.add_argument('--gpu_id', type=int, default=0, help = 'gpu id for training')
parser.add_argument('--threads', type=int, default=1, help = 'number of threads for training')

## dataset specification
parser.add_argument('--div2k_hr_path', type=str, default='/remote-home/share/DATA/super_resolution/DIV2K/DIV2K_train_HR', help = '')
parser.add_argument('--div2k_lr_path', type=str, default='/remote-home/share/DATA/super_resolution/DIV2K/DIV2K_train_LR_bicubic', help = '')
parser.add_argument('--set5_hr_path', type=str, default='/remote-home/share/DATA/super_resolution/Set5/HR', help = '')
parser.add_argument('--set5_lr_path', type=str, default='/remote-home/share/DATA/super_resolution/Set5/LR_bicubic', help = '')
parser.add_argument('--set14_hr_path', type=str, default='/remote-home/share/DATA/super_resolution/Set14/HR', help = '')
parser.add_argument('--set14_lr_path', type=str, default='/remote-home/share/DATA/super_resolution/Set14/LR_bicubic', help = '')
parser.add_argument('--b100_hr_path', type=str, default='/remote-home/share/DATA/super_resolution/B100/HR', help = '')
parser.add_argument('--b100_lr_path', type=str, default='/remote-home/share/DATA/super_resolution/B100/LR_bicubic', help = '')
parser.add_argument('--u100_hr_path', type=str, default='/remote-home/share/DATA/super_resolution/Urban100/HR', help = '')
parser.add_argument('--u100_lr_path', type=str, default='/remote-home/share/DATA/super_resolution/Urban100/LR_bicubic', help = '')
parser.add_argument('--manga109_hr_path', type=str, default='/remote-home/share/DATA/super_resolution/Manga109/HR', help = '')
parser.add_argument('--manga109_lr_path', type=str, default='/remote-home/share/DATA/super_resolution/Manga109/LR_bicubic', help = '')
parser.add_argument('--local_rank', type=int, default=0)

def down1(img1,img2):
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    img1, img2 = np.squeeze(img1), np.squeeze(img2)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
        
    M, N = img1.shape[:2]
    # automatically downsample
    f = max(1, round(min(M, N)/1024))
    if f > 1:
        lpf = np.ones((f, f))
        lpf = lpf / lpf.sum()
        img1 = cv2.filter2D(img1, -1, lpf)[0::f, 0::f]
        img2 = cv2.filter2D(img2, -1, lpf)[0::f, 0::f]
    M, N = img1.shape[:2]
    print(M,N)
    img1, img2 = torch.from_numpy(img1), torch.from_numpy(img2)
    return img1.view(1,1,M,N), img2.view(1,1,M,N)

def init_dist(local_rank, args):
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    torch.cuda.set_device(local_rank)
    #torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=torch.cuda.device_count(),rank=args.local_rank)
    dist.barrier()

if __name__ == '__main__':
    
    args = parser.parse_args()
    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)
    
    if args.colors == 3:
        raise ValueError("ECBSR is trained and tested with colors=1.")

    #initialization
    rank = args.local_rank
    if rank>= 1:
        init_dist(rank,args)

    # cpu or gpu
    device = torch.device("cuda")
    #device = torch.device('cpu')
    
    div2k = DIV2K(
    args.div2k_hr_path, 
    args.div2k_lr_path, 
    train=False, 
    augment=args.data_augment, 
    scale=args.scale, 
    colors=args.colors, 
    patch_size=args.patch_size, 
    repeat=args.data_repeat, 
    store_in_ram=True
)

    print(args.patch_size)
    set5  = Benchmark(args.set5_hr_path, args.set5_lr_path, scale=args.scale, colors=args.colors, store_in_ram=args.store_in_ram)
    set14 = Benchmark(args.set14_hr_path, args.set14_lr_path, scale=args.scale, colors=args.colors, store_in_ram=args.store_in_ram)
    b100  = Benchmark(args.b100_hr_path, args.b100_lr_path, scale=args.scale, colors=args.colors, store_in_ram=args.store_in_ram)
    u100  = Benchmark(args.u100_hr_path, args.u100_lr_path, scale=args.scale, colors=args.colors, store_in_ram=args.store_in_ram)
    manga109  = Benchmark(args.manga109_hr_path, args.manga109_lr_path, scale=args.scale, colors=args.colors, store_in_ram=args.store_in_ram)
    
    valid_dataloaders = []
    valid_dataloaders += [{'name': 'set5', 'dataloader': DataLoader(dataset=set5, batch_size=1, shuffle=False)}]
    valid_dataloaders += [{'name': 'set14', 'dataloader': DataLoader(dataset=set14, batch_size=1, shuffle=False)}]
    valid_dataloaders += [{'name': 'b100', 'dataloader': DataLoader(dataset=b100, batch_size=1, shuffle=False)}]
    valid_dataloaders += [{'name': 'u100', 'dataloader': DataLoader(dataset=u100, batch_size=1, shuffle=False)}]
    valid_dataloaders += [{'name': 'manga109', 'dataloader': DataLoader(dataset=manga109, batch_size=1, shuffle=False)}]
    valid_dataloaders += [{'name': 'div2k', 'dataloader': DataLoader(dataset=div2k, batch_size=1, shuffle=False)}]
    
    ## definitions of model, loss, and optimizer
    #model = ECBSR_train(module_nums=4, channel_nums=args.c_ecbsr, with_idt=args.idt_ecbsr, act_type=args.act_type, scale=args.scale, colors=args.colors).to(device)
    #model = ECBSR_5_ada_share().to(device)
    #model = ECBSR_5_ada_share_local().to(device)
    
    #model = ECBSR_re(module_nums=args.m_ecbsr, channel_nums=args.c_ecbsr, with_idt=args.idt_ecbsr, act_type=args.act_type, scale=args.scale, colors=args.colors).to(device)
    #model = ECBSR_5_ada_share_re().to(device)
    #model = ECBSR_5_ada_share_local_re().to(device)

    #model = ECBSR_5_one().to(device)
    model = ECBSR_5_one_re().to(device)
    print(model)
    #device_ids = [0,1]

    print('lr: '+ str(args.lr))
    # model_x2_ECBSR_64.pt   output/model_x2_ada_share_64.pt   output/model_x2_ada_every_64.pt    output/model_x2_5_one_256.pt
    # model_x2_ECBSR_128.pt  output/model_x2_ada_share_128.pt  output/model_x2_ada_every_128.pt
    print("load pretrained model: {}!".format('output/model_x2_5_one_256.pt'))
    state_dict = torch.load('output/model_x2_5_one_256.pt', map_location=device)
    # 多GPU 模型参数去掉 module
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # 去掉 `module.`
	    # name = k.replace(“module.", "")
        state_dict_new[name] = v
 
    # 模型加载参数（去掉module）
    model.load_state_dict(state_dict_new)
    for name, para in model.named_parameters():
        # 权重全部冻结
        print(name)
        para.requires_grad_(False)

    ## auto-generate the output logname
    timestamp = utils.cur_timestamp_str()
    experiment_name = "ecbsr-x{}-m{}c{}-{}-{}".format(args.scale, args.m_ecbsr, args.c_ecbsr, args.act_type, timestamp)
    experiment_path = os.path.join(args.log_path, experiment_name)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    experiment_model_path = os.path.join(experiment_path, 'models')
    if not os.path.exists(experiment_model_path):
        os.makedirs(experiment_model_path)

    log_name = os.path.join(experiment_path, "log.txt")
    sys.stdout = utils.ExperimentLogger(log_name, sys.stdout)
    stat_dict = utils.get_stat_dict()

    ## save training paramters
    exp_params = vars(args)
    exp_params_name = os.path.join(experiment_path, 'config.yml')
    with open(exp_params_name, 'w') as exp_params_file:
        yaml.dump(exp_params, exp_params_file, default_flow_style=False)

    model.rep_params() #load parameters
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        stat_dict['epochs'] = epoch
        
        print("##===========Epoch: {}=============##".format(epoch))
        if (epoch + 1) % args.test_every == 0:
            torch.set_grad_enabled(False)
            test_log = ""
            model = model.eval()
            index_dataset = 0
            for valid_dataloader in valid_dataloaders:
                avg_psnr = 0.0
                avg_ssim = 0.0
                name = valid_dataloader['name']
                loader = valid_dataloader['dataloader']
                sum_time = 0
                index = 0
                index_dataset += 1
                for lr, hr in tqdm(loader, ncols=80):
                    lr, hr = lr.to(device), hr.to(device)
                    torch.cuda.synchronize()
                    timer_test_start = time.time()
                    sr = model(lr)
                    torch.cuda.synchronize()
                    timer_test_end = time.time()
                    sum_time += timer_test_end - timer_test_start
                    # crop
                    hr = hr[:, :, args.scale:-args.scale, args.scale:-args.scale]
                    sr = sr[:, :, args.scale:-args.scale, args.scale:-args.scale]
                    # quantize
                    hr = hr.clamp(0, 255)
                    sr = sr.clamp(0, 255)
                    # calculate psnr
                    psnr = utils.calc_psnr(sr, hr)       
                    ssim = utils.calc_ssim(sr, hr)  
                    #print(psnr)
                    avg_psnr += psnr
                    avg_ssim += ssim
                    #输出重建后图片
                    _,_,H,W = sr.size()
                    output_rgb = sr.view(1,H,W).expand(3, -1, -1).permute(1, 2, 0).cpu().numpy()
                    index += 1
                    # Image.fromarray(
                    # np.uint8(np.round(output_rgb))
                    # ).save(
                    # 'output_image/'  + str(index_dataset) + '_' + str(index) +'.png'
                    # )
                print('推理时间： '+str(sum_time))
                avg_psnr = round(avg_psnr/len(loader), 2)
                avg_ssim = round(avg_ssim/len(loader), 4)
                stat_dict[name]['psnrs'].append(avg_psnr)
                stat_dict[name]['ssims'].append(avg_ssim)
                if stat_dict[name]['best_psnr']['value'] < avg_psnr:
                    stat_dict[name]['best_psnr']['value'] = avg_psnr
                    stat_dict[name]['best_psnr']['epoch'] = epoch
                if stat_dict[name]['best_ssim']['value'] < avg_ssim:
                    stat_dict[name]['best_ssim']['value'] = avg_ssim
                    stat_dict[name]['best_ssim']['epoch'] = epoch
                test_log += "[{}-X{}], PSNR/SSIM: {:.2f}/{:.4f} (Best: {:.2f}/{:.4f}, Epoch: {}/{})\n".format(
                    name, args.scale, float(avg_psnr), float(avg_ssim), 
                    stat_dict[name]['best_psnr']['value'], stat_dict[name]['best_ssim']['value'], 
                    stat_dict[name]['best_psnr']['epoch'], stat_dict[name]['best_ssim']['epoch'])
            # print log & flush out
            print(test_log)
            sys.stdout.flush()
            # save model
            saved_model_path = os.path.join(experiment_model_path, 'model_x{}_{}.pt'.format(args.scale, epoch))
            torch.save(model.state_dict(), saved_model_path)
            torch.set_grad_enabled(False)
            # save stat dict
            ## save training paramters
            stat_dict_name = os.path.join(experiment_path, 'stat_dict.yml')
            with open(stat_dict_name, 'w') as stat_dict_file:
                yaml.dump(stat_dict, stat_dict_file, default_flow_style=False)
                

