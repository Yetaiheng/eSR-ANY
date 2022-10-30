import torch
import torch.nn as nn
import torch.nn.functional as F
import time
#三种方法（ECBSR，全局系数提取，局部系数提取）的重参数化模型代码，只做测试使用，适用于全图测试
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
    
class ECB(nn.Module):############################# cpu ，cuda
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
            RK0, RN0 = self.conv3x3.weight.view(self.out_planes,self.inp_planes,3,3), self.conv3x3.bias.view(self.out_planes)
            RK1, RN1 = self.conv1x1_3x3.rep_params()        
            RK2, RN2 = self.conv1x1_sbx.rep_params()
            RK3, RN3 = self.conv1x1_sby.rep_params()
            RK4, RN4 = self.conv1x1_lpl.rep_params()
            RK, RN = (RK0+RK1+RK2+RK3+RK4), (RN0+RN1+RN2+RN3+RN4)
            self.RK.data.copy_(RK)
            self.RN.data.copy_(RN)
            if self.with_idt:
                y += x
        else:
            # timer_start = time.time()
            #RK, RB = self.rep_params()
            # timer_end = time.time()
            # duration = timer_end - timer_start
            # print('重参数化时间：'+ str(duration))
            
            y = F.conv2d(input=x, weight=self.RK, bias=self.RN, stride=1, padding=1) 
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
    
class ECBSR_re(nn.Module):
    def __init__(self, module_nums, channel_nums, with_idt, act_type, scale, colors):
        super(ECBSR_re, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.with_idt = with_idt
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        backbone = []
        backbone += [ECB(self.colors, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)]
        for i in range(self.module_nums):
            backbone += [ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)]
        backbone += [ECB(self.channel_nums, self.colors*self.scale*self.scale, depth_multiplier=2.0, act_type='linear', with_idt = self.with_idt)]

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        # torch.cuda.synchronize()
        # timer_test_start = time.time()
        y = self.backbone(x) + x
        y = self.upsampler(y)
        # torch.cuda.synchronize()
        # timer_test_end = time.time()
        # print(timer_test_end - timer_test_start)
        return y
    
    def rep_params(self):
        for i in range(6):
            self.backbone[i].rep_param()
    
class ECB_train(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier, act_type='prelu', with_idt = False):
        super(ECB_train, self).__init__()

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

        y = self.conv3x3(x)     + \
            self.conv1x1_3x3(x) + \
            self.conv1x1_sbx(x) + \
            self.conv1x1_sby(x) + \
            self.conv1x1_lpl(x)

        if self.act_type != 'linear':
            y = self.act(y)
        return y

class ECBSR_train(nn.Module):
    def __init__(self, module_nums, channel_nums, with_idt, act_type, scale, colors):
        super(ECBSR_train, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.with_idt = with_idt
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        backbone = []
        backbone += [ECB_train(self.colors, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)]
        for i in range(self.module_nums):
            backbone += [ECB_train(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)]
        backbone += [ECB_train(self.channel_nums, self.colors*self.scale*self.scale, depth_multiplier=2.0, act_type='linear', with_idt = self.with_idt)]

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        # torch.cuda.synchronize()
        # timer_test_start = time.time()
        y = self.backbone(x) + x
        y = self.upsampler(y)
        # torch.cuda.synchronize()
        # timer_test_end = time.time()
        # print(timer_test_end - timer_test_start)
        return y
    
class ECB_ada_DW(nn.Module):############################# cpu ，cuda
    def __init__(self, inp_planes, out_planes, depth_multiplier, act_type='prelu', with_idt = False):
        super(ECB_ada_DW, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type
        self.softmax = nn.Softmax(dim=1)

        self.RK, self.RN = torch.ones(out_planes,out_planes,5,3,3).cuda(), torch.ones(5,out_planes).cuda()
        #self.RK, self.RN = torch.randn(out_planes,out_planes,5,3,3).cpu(), torch.randn(5,out_planes).cpu()
        
        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False
        
        self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.inp_planes, self.out_planes, -1)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.inp_planes, self.out_planes, -1)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.inp_planes, self.out_planes, -1)

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
        RK0, RN0 = self.conv3x3.weight.view(self.out_planes,self.out_planes,1,3,3), self.conv3x3.bias.view(1,self.out_planes)
        RK1, RN1 = self.conv1x1_3x3.rep_params()        
        RK2, RN2 = self.conv1x1_sbx.rep_params()
        RK3, RN3 = self.conv1x1_sby.rep_params()
        RK4, RN4 = self.conv1x1_lpl.rep_params()
        RK = torch.cat((RK0,RK1.view(self.out_planes,self.out_planes,1,3,3),RK2.view(self.out_planes,self.out_planes,1,3,3),RK3.view(self.out_planes,self.out_planes,1,3,3),RK4.view(self.out_planes,self.out_planes,1,3,3)),dim=2)
        RN = torch.cat((RN0,RN1.view(1,self.out_planes),RN2.view(1,self.out_planes),RN3.view(1,self.out_planes),RN4.view(1,self.out_planes)),dim=0)
        self.RK.data.copy_(RK)
        self.RN.data.copy_(RN)
        #print(self.RK[0,0,0,0,:])
        return 1

    def forward(self, x,weight):
        B,C,H,W = x.size()
        if self.training:
                y = self.conv3x3(x) * weight[:,0].view(B,1,1,1)    + \
                    self.conv1x1_3x3(x)* weight[:,1].view(B,1,1,1) + \
                    self.conv1x1_sbx(x)* weight[:,2].view(B,1,1,1) + \
                    self.conv1x1_sby(x)* weight[:,3].view(B,1,1,1) + \
                    self.conv1x1_lpl(x)* weight[:,4].view(B,1,1,1)
                RK0, RN0 = self.conv3x3.weight.view(self.out_planes,self.out_planes,1,3,3), self.conv3x3.bias.view(1,self.out_planes)
                RK1, RN1 = self.conv1x1_3x3.rep_params()        
                RK2, RN2 = self.conv1x1_sbx.rep_params()
                RK3, RN3 = self.conv1x1_sby.rep_params()
                RK4, RN4 = self.conv1x1_lpl.rep_params()
                RK = torch.cat((RK0,RK1.view(self.out_planes,self.out_planes,1,3,3),RK2.view(self.out_planes,self.out_planes,1,3,3),RK3.view(self.out_planes,self.out_planes,1,3,3),RK4.view(self.out_planes,self.out_planes,1,3,3)),dim=2)
                RN = torch.cat((RN0,RN1.view(1,self.out_planes),RN2.view(1,self.out_planes),RN3.view(1,self.out_planes),RN4.view(1,self.out_planes)),dim=0)
                self.RK.data.copy_(RK)
                self.RN.data.copy_(RN)
                # print('RK:' + str(self.RK0.shape))
                # print('RN:' + str(self.RN0.shape))
        else:
            
            RK = torch.sum(self.RK * weight.view(1,1,5,1,1), dim=2)
            RB = torch.sum(self.RN * weight.view(5,1), dim=0)
            
            y = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1) 
        if self.with_idt:
            y += x

        if self.act_type != 'linear':
            y = self.act(y)
        return y      
    
class ECBSR_5_ada_share_re(nn.Module):
    def __init__(self):
        super(ECBSR_5_ada_share_re, self).__init__()
        self.module_nums = 4
        self.channel_nums = 8
        self.scale = 2
        self.colors = 1
        self.with_idt = 0
        self.act_type = 'prelu'
        self.backbone = None
        self.upsampler = None
        backbone_weight = []
        backbone_weight += [BasicBlock_conv_no_down(1, 8)]
        backbone_weight += [BasicBlock_conv(8, 20)]
        self.backbone_weight = nn.Sequential(*backbone_weight)
        self.getWeight1 = nn.Linear(20, 20,bias=True)
        
        backbone = []
        self.head= ECB(self.colors, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)
        for i in range(self.module_nums):
            backbone += [ECB_ada_DW(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)]
        self.tail = ECB(self.channel_nums, self.colors*self.scale*self.scale, depth_multiplier=2.0, act_type='linear', with_idt = self.with_idt)

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        weight = self.backbone_weight(x)
        weight = weight.mean([-2, -1])# N,C,H,W  ->  N,C
        weight = self.getWeight1(weight)
        shortcut = x
        x = self.head(x)
        
        for i in range(4):
            x = self.backbone[i](x,weight[:,5*i:5*i+5])
        x = self.tail(x)
        y = shortcut + x
        y = self.upsampler(y)
        return y
    
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

class ECB_ada_DW_local(nn.Module):##################### cuda ,cpu
    def __init__(self, inp_planes, out_planes, depth_multiplier, act_type='prelu', with_idt = False):
        super(ECB_ada_DW_local, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type
        self.softmax = nn.Softmax(dim=1)
        
        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False
        
        backbone = []
        #backbone += [BasicBlock_conv_no_down(1, 8)]
        backbone += [BasicBlock_conv(8, 20)]
        self.backbone = nn.Sequential(*backbone)
        self.getWeight = nn.Linear(20, 5,bias=True)
        
        self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.inp_planes, self.out_planes, -1)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.inp_planes, self.out_planes, -1)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.inp_planes, self.out_planes, -1)
        
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
        RK0, RN0 = self.conv3x3.weight.view(self.out_planes,self.out_planes,1,3,3), self.conv3x3.bias.view(1,self.out_planes)
        RK1, RN1 = self.conv1x1_3x3.rep_params()        
        RK2, RN2 = self.conv1x1_sbx.rep_params()
        RK3, RN3 = self.conv1x1_sby.rep_params()
        RK4, RN4 = self.conv1x1_lpl.rep_params()
        RK = torch.cat((RK0,RK1.view(self.out_planes,self.out_planes,1,3,3),RK2.view(self.out_planes,self.out_planes,1,3,3),RK3.view(self.out_planes,self.out_planes,1,3,3),RK4.view(self.out_planes,self.out_planes,1,3,3)),dim=2)
        RN = torch.cat((RN0,RN1.view(1,self.out_planes),RN2.view(1,self.out_planes),RN3.view(1,self.out_planes),RN4.view(1,self.out_planes)),dim=0)
        self.RK.data.copy_(RK)
        self.RN.data.copy_(RN)
        print(self.RK[0,0,0,0,:])
        return 1
    
    def forward(self, x):
        B,C,H,W = x.size()
        weight = self.backbone(x)
        weight = weight.mean([-2, -1])# N,C,H,W  ->  N,C
        weight = self.getWeight(weight)
        #print(weight.data)
        # #weight = self.softmax(weight)*2
        # weight1, weight2,weight3,weight4,weight5 = torch.split(weight, [1,1,1,1,1], dim=1)
        if self.training:
            y = self.conv3x3(x) * weight[:,0].view(B,1,1,1)    + \
                self.conv1x1_3x3(x)* weight[:,1].view(B,1,1,1) + \
                self.conv1x1_sbx(x)* weight[:,2].view(B,1,1,1) + \
                self.conv1x1_sby(x)* weight[:,3].view(B,1,1,1) + \
                self.conv1x1_lpl(x)* weight[:,4].view(B,1,1,1)
            RK0, RN0 = self.conv3x3.weight.view(self.out_planes,self.out_planes,1,3,3), self.conv3x3.bias.view(1,self.out_planes)
            RK1, RN1 = self.conv1x1_3x3.rep_params()        
            RK2, RN2 = self.conv1x1_sbx.rep_params()
            RK3, RN3 = self.conv1x1_sby.rep_params()
            RK4, RN4 = self.conv1x1_lpl.rep_params()
            RK = torch.cat((RK0,RK1.view(self.out_planes,self.out_planes,1,3,3),RK2.view(self.out_planes,self.out_planes,1,3,3),RK3.view(self.out_planes,self.out_planes,1,3,3),RK4.view(self.out_planes,self.out_planes,1,3,3)),dim=2)
            RN = torch.cat((RN0,RN1.view(1,self.out_planes),RN2.view(1,self.out_planes),RN3.view(1,self.out_planes),RN4.view(1,self.out_planes)),dim=0)
            self.RK.data.copy_(RK)
            self.RN.data.copy_(RN)
            #print(self.RK[0,0,0,0,:])
        else:
            #print(self.RK[0,0,0,0,:])
            
            RK = torch.sum(self.RK * weight.view(1,1,5,1,1), dim=2)
            RB = torch.sum(self.RN * weight.view(5,1), dim=0)
            
            y = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1) 
                
        if self.with_idt:
            y += x

        if self.act_type != 'linear':
            y = self.act(y)
        return y 

class ECBSR_5_ada_share_local_re(nn.Module):
    def __init__(self):
        super(ECBSR_5_ada_share_local_re, self).__init__()
        self.module_nums = 4
        self.channel_nums = 8
        self.scale = 2
        self.colors = 1
        self.with_idt = 0
        self.act_type = 'prelu'
        self.backbone = None
        self.upsampler = None
        
        backbone = []
        self.head= ECB(self.colors, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)
        for i in range(self.module_nums):
            backbone += [ECB_ada_DW_local(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)]
        self.tail = ECB(self.channel_nums, self.colors*self.scale*self.scale, depth_multiplier=2.0, act_type='linear', with_idt = self.with_idt)

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):

        shortcut = x
        x = self.head(x)
        # timer_start = time.time()
        for i in range(4):
            x = self.backbone[i](x)
        #y = self.backbone(x) + x
        # timer_end = time.time()
        # duration = timer_end - timer_start
        # print('主干网络时间：'+ str(duration))
        x = self.tail(x)
        y = shortcut + x
        y = self.upsampler(y)
        return y
    
    def rep_params(self):
        print(1)
        self.head.rep_param()
        self.tail.rep_param()
        for i in range(4):
            self.backbone[i].rep_param()
            
# if __name__ == '__main__':
#     # z = torch.randn((3, 1, 64, 64))
#     model = ECBSR_5_ada_share_local_re()
#     # z = model(z)
#     # print(z.shape)
#     model.rep_params()
    

