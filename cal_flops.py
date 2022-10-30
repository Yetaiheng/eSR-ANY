import torch
# 一次加法和乘法当作一次运算
def ECB_flops(C,O,H,W,K):#input channel,output channel, height, wide, kennel size
    all = O*H*W*(K*K*C)
    return all

def ECB_param(C,O,K):#input channel,output channel, kennel size
    all = C*O*K*K + O
    return all

def ECB_act(O,H,W):
    all = O*H*W
    return all 

#ECBSR 
scale = 2
input_channel = 1
middle_channel = 8
block_num = 4
output_channel = scale * scale

flop1 = ECB_flops(input_channel,middle_channel,1280/scale,720/scale,3)
flop2 = block_num * ECB_flops(middle_channel,middle_channel,1280/scale,720/scale,3)
flop3 = ECB_flops(middle_channel,output_channel,1280/scale,720/scale,3)
activations = (block_num + 1) * ECB_act(middle_channel,1280/scale,720/scale)
flop_sum = flop1 + flop2 + flop3 + activations
print("ECBSR flops:" + str(flop_sum/1e9))

param1 = ECB_param(input_channel,middle_channel,3)
param2 = block_num * ECB_param(middle_channel,middle_channel,3)
param3 = ECB_param(middle_channel,output_channel,3)
param_sum = param1 + param2 + param3
print("ECBSR params:" + str(param_sum/1e3))

# 自适应机制额外的参数量和计算量，全局区域池化，使用ECBSR的多分支卷积，两层(1,8)(8,20)
stride = 2
flop1 = ECB_flops(1,8,1280/scale,720/scale,3)
flop2 = ECB_flops(8,20,1280/scale,720/scale,3)/stride/stride
flop3_pool = 20 * 1280/scale * 720/scale / stride / stride / 2
flop4_FC = 20 * 20
flop_sum = flop1 + flop2 + flop3_pool + flop4_FC
print("extra predict network flops:" + str(flop_sum/1e9))

param1 = ECB_param(1,8,3)
param2 = ECB_param(8,20,3)
param_FC = 20 * 20 + 1
param_sum = param1 + param2 + param_FC
print("extra predict network params:" + str(param_sum/1e3))

param1 = (5 - 3) * ECB_param(1,8,3)
param2 = (5 - 3) * block_num * ECB_param(8,8,3)
param3 = (5 - 3) * ECB_param(8,scale * scale,3)
param_sum = param1 + param2 + param3
print("extra convolution kernel params:" + str(param_sum/1e3))

# 自适应机制额外的参数量和计算量，局部区域池化，使用general conv的普通卷积，三层(1,8)(8,16)（16，32）
stride = 2
pool_multiple = 64/48 #池化窗口/池化步距
flop1 = ECB_flops(1,8,1280/scale,720/scale,3)
flop2 = ECB_flops(8,16,1280/scale,720/scale,3)/stride/stride
flop3 = ECB_flops(16,32,1280/scale,720/scale,3)/stride/stride/stride/stride
flop4_pool = 32 * 1280/scale * 720/scale /stride/stride/stride/stride * pool_multiple * pool_multiple /2
flop5_FC = 32 * 20
flop_sum = flop1 + flop2 + flop3 + flop4_pool + flop5_FC
print("extra predict network flops:" + str(flop_sum/1e9))

flop1 = ECB_flops(input_channel,middle_channel,1280/scale,720/scale,3) * (pool_multiple * pool_multiple - 1)
flop2 = block_num * ECB_flops(middle_channel,middle_channel,1280/scale,720/scale,3) * (pool_multiple * pool_multiple - 1)
flop3 = ECB_flops(middle_channel,output_channel,1280/scale,720/scale,3) * (pool_multiple * pool_multiple - 1)
flop_sum = flop1 + flop2 + flop3
print("extra convolution kernel flops:" + str(flop_sum/1e9))

param1 = ECB_param(1,8,3)
param2 = ECB_param(8,16,3)
param3 = ECB_param(16,32,3)
param_FC = 32 * 20 + 1
param_sum = param1 + param2 + param3 + param_FC
print("extra predict network params:" + str(param_sum/1e3))

param1 = (5 - 1) * ECB_param(input_channel,middle_channel,3)
param2 = (5 - 1) * block_num * ECB_param(middle_channel,middle_channel,3)
param3 = (5 - 1) * ECB_param(middle_channel,scale * scale,3)
param_sum = param1 + param2 + param3
print("extra convolution kernel params:" + str(param_sum/1e3))
