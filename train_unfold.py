#测试时候使用小patch测试的训练代码
import torch
import torch.nn as nn
import torch.nn.functional as F
from datas.benchmark import Benchmark
from datas.div2k import DIV2K
from models.ecbsr import ECBSR
from models.ada_conv_ECB import ECBSR_5_ada
from models.ada_conv_ECB_share import ECBSR_5_ada_share
from models.SRCNN import ESPCN,SRCNN
from models.ada_conv_ECB_share_local import ECBSR_5_ada_share_local,ECBSR_5_ada_share_local_cat
from torch.utils.data import DataLoader
import math
import argparse, yaml
import utils
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '1234'
from tqdm import tqdm
import logging
import sys
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
parser.add_argument('--pretrain', type=str, default=None, help = 'path of pretrained model')

## parameters for model training
parser.add_argument('--patch_size', type=int, default=64, help = 'patch size of HR image')
parser.add_argument('--batch_size', type=int, default=32, help = 'batch size of training data')
parser.add_argument('--data_repeat', type=int, default=1, help = 'times of repetition for training data')
parser.add_argument('--data_augment', type=int, default=1, help = 'data augmentation for training')
parser.add_argument('--epochs', type=int, default=600, help = 'number of epochs')
parser.add_argument('--test_every', type=int, default=1, help = 'test the model every N epochs')
parser.add_argument('--log_every', type=int, default=1, help = 'print log of loss, every N steps')
parser.add_argument('--log_path', type=str, default="./experiments/")
parser.add_argument('--lr', type=float, default=80e-4, help = 'learning rate of optimizer')
parser.add_argument('--store_in_ram', type=int, default=0, help = 'store the whole training data in RAM or not')

## hardware specification
parser.add_argument('--gpu_id', type=int, default=1, help = 'gpu id for training')
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
parser.add_argument('--local_rank', type=int, default=0)

def init_dist(local_rank, args):
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=torch.cuda.device_count(),rank=args.local_rank)
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

    device = torch.device("cuda")
    # if args.gpu_id >= 0 and torch.cuda.is_available():
    #     print("use cuda & cudnn for acceleration!")
    #     print("the gpu id is: {}".format(args.gpu_id))
    #     device = torch.device('cuda:{}'.format(args.gpu_id))
    #     torch.backends.cudnn.benchmark = True
    # else:
    #     print("use cpu for training!")
    #     device = torch.device('cpu')
    # torch.set_num_threads(args.threads)

    div2k = DIV2K(
        args.div2k_hr_path, 
        args.div2k_lr_path, 
        train=True, 
        augment=args.data_augment, 
        scale=args.scale, 
        colors=args.colors, 
        patch_size=args.patch_size, 
        repeat=args.data_repeat, 
        store_in_ram=args.store_in_ram
    )
    print(args.patch_size)
    set5  = Benchmark(args.set5_hr_path, args.set5_lr_path, scale=args.scale, colors=args.colors, store_in_ram=args.store_in_ram)
    set14 = Benchmark(args.set14_hr_path, args.set14_lr_path, scale=args.scale, colors=args.colors, store_in_ram=args.store_in_ram)
    b100  = Benchmark(args.b100_hr_path, args.b100_lr_path, scale=args.scale, colors=args.colors, store_in_ram=args.store_in_ram)
    u100  = Benchmark(args.u100_hr_path, args.u100_lr_path, scale=args.scale, colors=args.colors, store_in_ram=args.store_in_ram)
    manga109  = Benchmark(args.manga109_hr_path, args.manga109_lr_path, scale=args.scale, colors=args.colors, store_in_ram=args.store_in_ram)

    
    train_dataloader = DataLoader(dataset=div2k, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    #train_dataloader = DataLoader(dataset=div2k, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, sampler=DistributedSampler(div2k))
    valid_dataloaders = []
    valid_dataloaders += [{'name': 'set5', 'dataloader': DataLoader(dataset=set5, batch_size=1, shuffle=False)}]
    valid_dataloaders += [{'name': 'set14', 'dataloader': DataLoader(dataset=set14, batch_size=1, shuffle=False)}]
    valid_dataloaders += [{'name': 'b100', 'dataloader': DataLoader(dataset=b100, batch_size=1, shuffle=False)}]
    valid_dataloaders += [{'name': 'u100', 'dataloader': DataLoader(dataset=u100, batch_size=1, shuffle=False)}]
    valid_dataloaders += [{'name': 'manga109', 'dataloader': DataLoader(dataset=manga109, batch_size=1, shuffle=False)}]

    ## definitions of model, loss, and optimizer
    #model = ECBSR(module_nums=10, channel_nums=args.c_ecbsr, with_idt=args.idt_ecbsr, act_type=args.act_type, scale=args.scale, colors=args.colors).to(device)
    #model = ECBSR_5_ada_share()
    #model = ECBSR_5_ada_share_local()
    model = ECBSR_5_ada_share_local_cat()
    #model = ESPCN(args.scale)
    #model = SRCNN()
    print(model)
    device_ids = [0,1]

    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)
    # if args.local_rank >= 1:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids={args.local_rank}, output_device=args.local_rank)
    loss_func = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print('lr: '+ str(args.lr))
    if args.pretrain is not None:
        print("load pretrained model: {}!".format(args.pretrain))
        model.load_state_dict(torch.load(args.pretrain))
    else:
        print("train the model from scratch!")

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

    timer_start = time.time()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        stat_dict['epochs'] = epoch
        model = model.train()
        print("##===========Epoch: {}=============##".format(epoch))
        for iter, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            lr, hr = batch
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = loss_func(sr, hr)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss)

            if (iter + 1) % args.log_every == 0:

                cur_steps = (iter+1)*args.batch_size
                total_steps = len(train_dataloader.dataset)
                fill_width = math.ceil(math.log10(total_steps))
                cur_steps = str(cur_steps).zfill(fill_width)

                epoch_width = math.ceil(math.log10(args.epochs))
                cur_epoch = str(epoch).zfill(epoch_width)

                avg_loss = epoch_loss / (iter + 1)
                stat_dict['losses'].append(avg_loss)

                timer_end = time.time()
                duration = timer_end - timer_start
                timer_start = timer_end
                print("Epoch:{}, {}/{}, loss: {:.4f}, time: {:.3f}".format(cur_epoch, cur_steps, total_steps, avg_loss, duration))

        if (epoch + 1) % args.test_every == 0:
            torch.set_grad_enabled(False)
            test_log = ""
            model = model.eval()
            for valid_dataloader in valid_dataloaders:
                avg_psnr = 0.0
                avg_ssim = 0.0
                name = valid_dataloader['name']
                loader = valid_dataloader['dataloader']
                sum_time = 0
                sum_model_time = 0
                for lr, hr in tqdm(loader, ncols=80):
                    torch.cuda.synchronize()
                    timer_test_start = time.time()
                    a,b,h,w = lr.shape
                    if h <= 128 or w <= 128:
                        lr = lr.to(device)
                        sr = model(lr)# [1,1,256,256]
                    else:
                        crop_sz = 64
                        final_sz = crop_sz*2
                        step = 48
                        pad = 0
                        thres_sz = 0
                        H_num = (h - crop_sz + step)//step
                        W_num = (w - crop_sz + step)//step
                        
                        lr_patch = F.unfold(lr, kernel_size=crop_sz, stride=step,padding=(pad,pad)).permute(2,1,0).contiguous()   
                        lr_patch = lr_patch.view(-1,1,crop_sz,crop_sz).contiguous().to(device)
                        
                        lr_patch_edge_w = F.unfold(lr[:,:,crop_sz*(-1):,:], kernel_size=crop_sz, stride=step,padding=(pad,pad)).permute(2,1,0).contiguous() 
                        lr_patch_edge_w = lr_patch_edge_w.view(-1,1,crop_sz,crop_sz).contiguous().to(device) 
                        lr_patch_edge_h = F.unfold(lr[:,:,:,crop_sz*(-1):], kernel_size=crop_sz, stride=step,padding=(pad,pad)).permute(2,1,0).contiguous() 
                        lr_patch_edge_h = lr_patch_edge_h.view(-1,1,crop_sz,crop_sz).contiguous().to(device) 
                        lr_patch = torch.cat((lr_patch, lr_patch_edge_w, lr_patch_edge_h,lr[:,:,crop_sz*(-1):,crop_sz*(-1):].to(device))).contiguous()
                        #print(torch.sum(lr[0,0,0:128,0:128].to(device) - lr_patch[0,0,:,:]))
                        
                        torch.cuda.synchronize()
                        timer_test_start_1 = time.time()
                        sr_patch = model(lr_patch)
                        torch.cuda.synchronize()
                        timer_test_end_1 = time.time()
                        sum_model_time +=timer_test_end_1 - timer_test_start_1
                        
                        sr_patch_wh = sr_patch[-1,:,:,:].view(1,1,final_sz,final_sz)
                        sr_patch = sr_patch.view(-1,1,final_sz*final_sz)
                        sr_patch = sr_patch.permute(1,2,0).contiguous()
                        final_H = 2*step*H_num + 2*(crop_sz - step)
                        final_W = 2*step*W_num + 2*(crop_sz - step)
                        sr_i = F.fold(input = sr_patch[:,:,0:H_num * W_num],output_size=(final_H, final_W), kernel_size=(final_sz, final_sz), stride=2*step)
                        sr_w = F.fold(input = sr_patch[:,:,H_num * W_num:H_num * W_num+W_num],output_size=(final_sz, final_W), kernel_size=(final_sz, final_sz), stride=2*step)
                        sr_h = F.fold(input = sr_patch[:,:,H_num * W_num+W_num:H_num * W_num+W_num+H_num],output_size=(final_H, final_sz), kernel_size=(final_sz, final_sz), stride=2*step)
                        
                        count_i = torch.ones(1,final_sz*final_sz,H_num * W_num).to(device)
                        count_w = torch.ones(1,final_sz*final_sz, W_num).to(device)
                        count_h = torch.ones(1,final_sz*final_sz, H_num).to(device)
                        sr_count_i = F.fold(input = count_i,output_size=(final_H, final_W), kernel_size=(final_sz, final_sz), stride=2*step)
                        sr_count_w = F.fold(input = count_w,output_size=(final_sz, final_W), kernel_size=(final_sz, final_sz), stride=2*step)
                        sr_count_h = F.fold(input = count_h,output_size=(final_H, final_sz), kernel_size=(final_sz, final_sz), stride=2*step)
                        
                        sr = torch.zeros(1,1,2*h,2*w).to(device)
                        sr_cat_count = torch.zeros(1,1,2*h,2*w).to(device)
                        sr[:,:,0:final_H,0:final_W] += sr_i
                        sr[:,:,final_sz*(-1):,0:final_W] += sr_w
                        sr[:,:,0:final_H,final_sz*(-1):] += sr_h
                        sr[:,:,final_sz*(-1):,final_sz*(-1):] += sr_patch_wh
                        #final_sz*(-1)
                        sr_cat_count[:,:,0:final_H,0:final_W] += sr_count_i
                        sr_cat_count[:,:,final_sz*(-1):,0:final_W] += sr_count_w
                        sr_cat_count[:,:,0:final_H,final_sz*(-1):] += sr_count_h       
                        sr_cat_count[:,:,final_sz*(-1):,final_sz*(-1):] += 1
                        
                        sr = sr/sr_cat_count
                    sr, hr = sr.to(device), hr.to(device)
                    torch.cuda.synchronize()
                    timer_test_end = time.time()
                    sum_time += timer_test_end - timer_test_start
                    # crop
                    hr = hr[:, :, args.scale:-args.scale, args.scale:-args.scale]#[1,1,252,252]
                    sr = sr[:, :, args.scale:-args.scale, args.scale:-args.scale]#[1,1,252,252]
                    # quantize
            
                    hr = hr.clamp(0, 255)
                    sr = sr.clamp(0, 255)
                    # calculate psnr
                    psnr = utils.calc_psnr(sr, hr)       
                    ssim = utils.calc_ssim(sr, hr)     
                    #print(psnr)
                    # psnr = utils_2.calculate_PSNR(sr,hr)   
                    # ssim = utils_2.calculate_SSIM(sr,hr)     
                    avg_psnr += psnr
                    avg_ssim += ssim
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
            # saved_model_path2 = os.path.join(experiment_model_path, 'model_x{}_{}.pth'.format(args.scale, epoch))
            # torch.save(model, saved_model_path2)
            torch.set_grad_enabled(True)
            # save stat dict
            ## save training paramters
            stat_dict_name = os.path.join(experiment_path, 'stat_dict.yml')
            with open(stat_dict_name, 'w') as stat_dict_file:
                yaml.dump(stat_dict, stat_dict_file, default_flow_style=False)