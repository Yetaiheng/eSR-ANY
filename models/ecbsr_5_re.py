from tkinter import Y
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
            
        elif self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        if self.type == 'conv1x1-conv3x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)#思路，y1乘计算出来的系数
        return y1
    
    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.type == 'conv1x1-conv3x3':
            # re-param conv kernel
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1,) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1,) + b1
        return RK, RB

class ECB(nn.Module):#修改后的block
    def __init__(self, inp_planes, out_planes, depth_multiplier, act_type='prelu', with_idt = False):
        super(ECB, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type
        
        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        self.conv3x3_0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv3x3_1 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv3x3_2 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv3x3_3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv3x3_4 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        
        self.RK, self.RN = torch.ones(out_planes,out_planes,5,3,3).cuda(), torch.ones(5,out_planes).cuda()
        #self.RK, self.RN = torch.randn(out_planes,out_planes,5,3,3).cpu(), torch.randn(5,out_planes).cpu()

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def rep_param(self):
        RK0, RN0 = self.conv3x3_0.weight.view(self.out_planes,self.out_planes,1,3,3), self.conv3x3_0.bias.view(1,self.out_planes)
        RK1, RN1 = self.conv3x3_1.weight.view(self.out_planes,self.out_planes,1,3,3), self.conv3x3_1.bias.view(1,self.out_planes)
        RK2, RN2 = self.conv3x3_2.weight.view(self.out_planes,self.out_planes,1,3,3), self.conv3x3_2.bias.view(1,self.out_planes)
        RK3, RN3 = self.conv3x3_3.weight.view(self.out_planes,self.out_planes,1,3,3), self.conv3x3_3.bias.view(1,self.out_planes)
        RK4, RN4 = self.conv3x3_4.weight.view(self.out_planes,self.out_planes,1,3,3), self.conv3x3_4.bias.view(1,self.out_planes)
        RK = torch.cat((RK0,RK1.view(self.out_planes,self.out_planes,1,3,3),RK2.view(self.out_planes,self.out_planes,1,3,3),RK3.view(self.out_planes,self.out_planes,1,3,3),RK4.view(self.out_planes,self.out_planes,1,3,3)),dim=2)
        RN = torch.cat((RN0,RN1.view(1,self.out_planes),RN2.view(1,self.out_planes),RN3.view(1,self.out_planes),RN4.view(1,self.out_planes)),dim=0)
        self.RK.data.copy_(RK)
        self.RN.data.copy_(RN)
        #print(self.RK[0,0,0,0,:])
        return 1

    def forward(self, x,weight):
        B,C,H,W = x.size()
        if self.training:
            y = self.conv3x3_0(x)* weight[:,0].view(B,1,1,1) + \
                self.conv3x3_1(x)* weight[:,1].view(B,1,1,1) + \
                self.conv3x3_2(x)* weight[:,2].view(B,1,1,1) + \
                self.conv3x3_3(x)* weight[:,3].view(B,1,1,1) + \
                self.conv3x3_4(x)* weight[:,4].view(B,1,1,1)
        else:
            RK_patch = torch.sum(self.RK.view(1, self.out_planes,self.out_planes,5,3,3) * weight.view(B,1,1,5,1,1), dim = 3).view(-1,self.out_planes,3,3)
            RB_patch = torch.sum(self.RN.view(1,5,self.out_planes) * weight.view(B,5,1), dim=1).view(-1)
            x = x.view(1,-1,H,W)
            y = F.conv2d(input=x, weight=RK_patch, bias=RB_patch, stride=1, padding=1,groups= B).view(B,-1,H,W)
        if self.with_idt:
            y += x

        if self.act_type != 'linear':
            y = self.act(y)
        return y

class ECB_ori(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier, act_type='prelu', with_idt = False):
        super(ECB_ori, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type
        
        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.inp_planes, self.out_planes, -1)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.inp_planes, self.out_planes, -1)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.inp_planes, self.out_planes, -1)
        
        self.RK, self.RN = torch.ones(out_planes,inp_planes,3,3).cuda(), torch.ones(out_planes).cuda()
        #self.RK, self.RN = torch.ones(out_planes,inp_planes,3,3).cpu(), torch.ones(out_planes).cpu()

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        if self.training:
            y = self.conv3x3(x)     + \
                self.conv1x1_3x3(x) + \
                self.conv1x1_sbx(x) + \
                self.conv1x1_sby(x) + \
                self.conv1x1_lpl(x)
            if self.with_idt:
                y += x
        else:
            y = F.conv2d(input=x, weight=self.RK, bias=self.RN, stride=1, padding=1) ##
        if self.act_type != 'linear':
            y = self.act(y)
        return y
    
    def rep_param(self):
            RK0, RN0 = self.conv3x3.weight.view(self.out_planes,self.inp_planes,3,3), self.conv3x3.bias.view(self.out_planes)
            RK1, RN1 = self.conv1x1_3x3.rep_params()        
            RK2, RN2 = self.conv1x1_sbx.rep_params()
            RK3, RN3 = self.conv1x1_sby.rep_params()
            RK4, RN4 = self.conv1x1_lpl.rep_params()
            RK, RN = (RK0+RK1+RK2+RK3+RK4), (RN0+RN1+RN2+RN3+RN4)
            self.RK.data.copy_(RK)
            self.RN.data.copy_(RN)
            
    def rep_params(self):
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        RK, RB = (K0+K1+K2+K3+K4), (B0+B1+B2+B3+B4)

        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB

class ECBSR_5_one_re(nn.Module):
    def __init__(self):
        super(ECBSR_5_one_re, self).__init__()
        self.module_nums = 4
        self.channel_nums = 8
        self.scale = 2
        self.colors = 1
        self.with_idt = 0
        self.act_type = 'prelu'
        self.backbone = None
        self.upsampler = None
        self.window_size = 64
        self.step = 48
        
        backbone_weight = []
        backbone_weight += [BasicBlock_conv_no_down(1, 8)]
        backbone_weight += [BasicBlock_conv(8, 16)]#8,16
        backbone_weight += [BasicBlock_conv(16, 32)]#8,16
        self.backbone_weight = nn.Sequential(*backbone_weight)
        self.getWeight1 = nn.Linear(32, 20,bias=True)

        backbone = []
        self.head = ECB_ori(self.colors, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)
        for i in range(self.module_nums):
            backbone += [ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)]
        self.tail = ECB_ori(self.channel_nums, self.colors*self.scale*self.scale, depth_multiplier=2.0, act_type='linear', with_idt = self.with_idt)

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = self.step - (h - self.window_size)% self.step
        mod_pad_w = self.step - (w - self.window_size)% self.step
        num_h = (h + mod_pad_h - self.window_size + self.step)//self.step
        num_w = (w + mod_pad_w - self.window_size + self.step)//self.step
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x,num_h,num_w
    
    def forward(self, x):
        B, C, H, W= x.shape
        x,num_h,num_w = self.check_image_size(x)#padding image
        weight = self.backbone_weight(x)#B,32,H,W
        weight = F.avg_pool2d(weight, self.window_size//4, self.step//4)#B,32,n_h,n_w
        weight = weight.permute(0, 2, 3, 1).contiguous().view(-1, 32)#B*n*n,32
        weight = self.getWeight1(weight)#B*n*n,20
       
        x = F.unfold(x, kernel_size=self.window_size, stride=self.step).permute(0,2,1).contiguous()#B,n*n,win*win*1
        x = x.view(-1, 1, self.window_size,self.window_size)#B*n*n,1,win,win
        shortcut = x
        x = self.head(x)
        for i in range(4):
            x = self.backbone[i](x,weight[:,5*i:5*i+5])
        x = self.tail(x)
        y = shortcut + x
        y = self.upsampler(y)##B*n*n,1,2win,2win
        y = y.view(B,-1,2*2*self.window_size*self.window_size)
        y = y.permute(0, 2, 1).contiguous()#B,2*2*win*win,N
        final_H = 2*self.step*num_h + 2*(self.window_size - self.step)
        final_W = 2*self.step*num_w + 2*(self.window_size - self.step)
        count_i = torch.ones(1,4*self.window_size*self.window_size,num_h * num_w).cuda()############################
        final_sz = 2*self.window_size
        sr_count_i = F.fold(input = count_i,output_size=(final_H, final_W), kernel_size=(final_sz, final_sz), stride=2*self.step)
        sr_i = F.fold(input = y,output_size=(final_H, final_W), kernel_size=(final_sz, final_sz), stride=2*self.step)
        sr = sr_i/sr_count_i
        sr = sr[:,:,0:H*self.scale, 0:W*self.scale]
        return sr
    
    def rep_params(self):
        self.head.rep_param()
        self.tail.rep_param()
        for i in range(4):
            self.backbone[i].rep_param()
  
class ECBSR_5_one_re_store(nn.Module):
    def __init__(self):
        super(ECBSR_5_one_re_store, self).__init__()
        self.module_nums = 4
        self.channel_nums = 8
        self.scale = 2
        self.colors = 1
        self.with_idt = 0
        self.act_type = 'prelu'
        self.backbone = None
        self.upsampler = None
        self.window_size = 64
        self.step = 48
        
        backbone_weight = []
        backbone_weight += [BasicBlock_conv_no_down(1, 8)]
        backbone_weight += [BasicBlock_conv(8, 16)]#8,16
        backbone_weight += [BasicBlock_conv(16, 32)]#8,16
        self.backbone_weight = nn.Sequential(*backbone_weight)
        self.getWeight1 = nn.Linear(32, 20,bias=True)

        backbone = []
        self.head = ECB_ori(self.colors, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)
        for i in range(self.module_nums):
            backbone += [ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)]
        self.tail = ECB_ori(self.channel_nums, self.colors*self.scale*self.scale, depth_multiplier=2.0, act_type='linear', with_idt = self.with_idt)

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = self.step - (h - self.window_size)% self.step
        mod_pad_w = self.step - (w - self.window_size)% self.step
        num_h = (h + mod_pad_h - self.window_size + self.step)//self.step
        num_w = (w + mod_pad_w - self.window_size + self.step)//self.step
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x,num_h,num_w
    
    def forward(self, x):
        B, C, H, W= x.shape
        x,num_h,num_w = self.check_image_size(x)#padding image
        weight = self.backbone_weight(x)#B,32,H,W
        weight = F.avg_pool2d(weight, self.window_size//4, self.step//4)#B,32,n_h,n_w
        weight = weight.permute(0, 2, 3, 1).contiguous().view(-1, 32)#B*n*n,32
        weight = self.getWeight1(weight)#B*n*n,20
        # 存储一张图各个不同区域的20个系数
        a = weight.cpu().numpy()
        np.savetxt('txt_param/result0805.txt',a)
       
        x = F.unfold(x, kernel_size=self.window_size, stride=self.step).permute(0,2,1).contiguous()#B,n*n,win*win*1
        x = x.view(-1, 1, self.window_size,self.window_size)#B*n*n,1,win,win
        
        #存储分割后的图像
        N,_,H2,W2 = x.size()
        for i in range(N):
            sr_patch = x[i,:,:,:]
            output_rgb = sr_patch.view(1,H2,W2).expand(3, -1, -1).permute(1, 2, 0).cpu().numpy()
            Image.fromarray(
            np.uint8(np.round(output_rgb))
            ).save(
            'output_image/'  + str(805) + '_' + str(i) +'.png'
            )
        
        shortcut = x
        x = self.head(x)
        for i in range(4):
            x = self.backbone[i](x,weight[:,5*i:5*i+5])
        x = self.tail(x)
        y = shortcut + x
        y = self.upsampler(y)##B*n*n,1,2win,2win
        y = y.view(B,-1,2*2*self.window_size*self.window_size)
        y = y.permute(0, 2, 1).contiguous()#B,2*2*win*win,N
        final_H = 2*self.step*num_h + 2*(self.window_size - self.step)
        final_W = 2*self.step*num_w + 2*(self.window_size - self.step)
        count_i = torch.ones(1,4*self.window_size*self.window_size,num_h * num_w).cuda()############################
        final_sz = 2*self.window_size
        sr_count_i = F.fold(input = count_i,output_size=(final_H, final_W), kernel_size=(final_sz, final_sz), stride=2*self.step)
        sr_i = F.fold(input = y,output_size=(final_H, final_W), kernel_size=(final_sz, final_sz), stride=2*self.step)
        sr = sr_i/sr_count_i
        sr = sr[:,:,0:H*self.scale, 0:W*self.scale]
        return sr
    
    def rep_params(self):
        self.head.rep_param()
        self.tail.rep_param()
        for i in range(4):
            self.backbone[i].rep_param() 
    
class BasicBlock_conv(nn.Module):
    def __init__(self, dim,out_planes, drop_rate=0., layer_scale_init_value=1e-6):# dim = 96
        super().__init__()
        self.dwconv = nn.Conv2d(dim, out_planes, kernel_size=3, stride = 2,padding=1)  # depthwise conv
        self.norm = torch.nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act = torch.nn.LeakyReLU(negative_slope=0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:#[1,96,50,50]
        x = self.dwconv(x)#[1,96,50,50]
        x = self.act(x)#[1,50,50,384]
        x = self.norm(x)#[1,50,50,96]
        return x
    
class BasicBlock_conv_no_down(nn.Module):
    def __init__(self, dim,out_planes, drop_rate=0., layer_scale_init_value=1e-6):# dim = 96
        super().__init__()
        self.dwconv = nn.Conv2d(dim, out_planes, kernel_size=3, stride = 1,padding=1)  # depthwise conv
        self.norm = torch.nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act = torch.nn.LeakyReLU(negative_slope=0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:#[1,96,50,50]
        x = self.dwconv(x)#[1,96,50,50]
        x = self.act(x)#[1,50,50,384]
        x = self.norm(x)#[1,50,50,96]
        return x
   
if __name__ == '__main__':
    z = torch.randn((32, 1, 500, 500))
    #out = F.unfold(z, kernel_size=64, stride=64).permute(2,1,0).contiguous()   
    # out = F.avg_pool2d(z,64,48)
    # model = ECBSR_5_one()
    # out = model(z)
    # print(out.shape)