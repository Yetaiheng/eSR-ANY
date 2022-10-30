import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from collections import OrderedDict
from utils import GradCAM, show_cam_on_image, center_crop_img
from ecbsr_5 import ECBSR_5_one_grad

def main():
    # model = models.mobilenet_v3_small(pretrained=True)
    # target_layers = [model.features[-1]]
    
    # cpu or gpu
    #device = torch.device("cuda")
    device = torch.device('cpu')

    model = ECBSR_5_one_grad().to(device)
    print("load pretrained model: {}!".format('model_x2_5_one_256.pt'))
    state_dict = torch.load('model_x2_5_one_256.pt', map_location=device)
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
        #para.requires_grad_(False)
        
    target_layers = [model.backbone_weight]
    

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "805_72.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('L')
    img = np.array(img, dtype=np.float32)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = torch.from_numpy(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)
    input_tensor = torch.unsqueeze(input_tensor, dim=0)
    print(input_tensor.shape)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    for i in range(20):
        target_category = i  # tabby, tabby cat
        # target_category = 254  # pug, pug-dog

        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                        grayscale_cam,
                                        use_rgb=True)
        # plt.imshow(visualization)
        # plt.show()
        
        Image.fromarray(
        np.uint8(np.round(visualization))
        ).save("805_72_" +
        str(target_category) +'.png'
        )


if __name__ == '__main__':
    main()
