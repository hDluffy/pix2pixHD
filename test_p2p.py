
import cv2
import torch
import numpy as np
from PIL import Image
import PIL
import torch.nn.functional as F
from torchvision import transforms
import models.networks as networks
import os
from util import test_util 
import onnxruntime
from thop import profile
# 增加可读性
from thop import clever_format

transformer_G = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

if __name__ == '__main__':
    device = torch.device('cuda:0') 
    imagenet_std    = torch.Tensor([0.5, 0.5, 0.5]).view(3,1,1).to(device)
    imagenet_mean   = torch.Tensor([0.5, 0.5, 0.5]).view(3,1,1).to(device)

    with_mask = 0
    with_domains = 0
    input_w = 512
    input_h = 512
    input_nc = 3
    do_onnx=0
    
    pic_dir='./datasets/face_align/cartoon/'
    if with_mask:
        pic_dir='./datasets/face_align/face_edit/Hairpaint/'
        input_w = 768
        input_h = 896
        input_nc += 3
    elif with_domains:
        pic_dir='./datasets/face_align/face_edit/age_edit/'
        input_nc += 1

    netG = networks.define_G(input_nc, 3, 16, 'global','Unet', 4, 4, 1, 3, 'instance', gpu_ids=[]).to(device)
    test_util.load_network(netG, 'G', 'latest', './checkpoints/2WaterColour/')
    netG.eval()

    if do_onnx:
        pix2pix_input = torch.randn(1, input_nc, input_w, input_h, device=device)
        flops, params = profile(netG, inputs=(pix2pix_input,))
        flops, params = clever_format([flops, params], "%.3f")
        with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False) as prof:
            in0 = torch.ones(1, input_nc, input_w, input_h, device=device)
            out = netG(in0)
        print(prof.table())
        #prof.export_chrome_trace('./profile.json')

        input_names = [ "input"]
        output_names = [ "output" ]
        torch.onnx.export(netG, (pix2pix_input), "2WaterColour.onnx", verbose=True, input_names=input_names, output_names=output_names)
        #torch.onnx.export(netG, (pix2pix_input), "Hairpaint.onnx", verbose=True, input_names=input_names, output_names=output_names,dynamic_axes={'input': {0: 'batch'},'output': {0: 'batch'}})

    with torch.no_grad():
        torch.set_printoptions(edgeitems=768)
        
        pic = pic_dir + '0.jpg'
        input_img = Image.open(pic).convert('RGB')
        input_img = transformer_G(input_img).to(device).unsqueeze(0)
        input_img = F.interpolate(input_img, size=(input_w, input_h), mode='bilinear')
        input_img_ = input_img
        if with_mask:
            pic_mask = pic_dir + '0_mask.jpg'
            input_img_mask = Image.open(pic_mask).convert('RGB')
            input_img_mask = transformer_G(input_img_mask).to(device).unsqueeze(0)
            input_img_mask = F.interpolate(input_img_mask, size=(input_w, input_h), mode='bilinear')
            input_img_ = torch.cat([input_img, input_img_mask], dim=1).detach()
        elif with_domains:
            N,C,H,W = input_img.size()
            tensor_c=(torch.ones(N,1,H,W)*(-0.6)).cuda()
            input_img_ = torch.cat([input_img, tensor_c], dim=1).detach()
        ############## Forward Pass ######################
        img_fake = netG(input_img_)

        row1 = input_img[0]*imagenet_std + imagenet_mean
        row2 = img_fake[0]*imagenet_std + imagenet_mean
        full = torch.cat([row1, row2], dim=2).detach()
        output = full.mul(255).permute(1, 2, 0).cpu().numpy()
        output = output[..., ::-1]
        cv2.imwrite('result_cv.jpg',output)

        if 1:
            imgs = list()
            for i in range(5):
                pic = pic_dir + str(i) + '.jpg'

                input_img = Image.open(pic).convert('RGB')
                input_img = transformer_G(input_img).to(device).unsqueeze(0)
                input_img = F.interpolate(input_img, size=(input_w, input_h), mode='bilinear')
                input_img_ = input_img
                if with_mask:
                    pic_mask = pic_dir + str(i) + '_mask.jpg'
                    input_img_mask = Image.open(pic_mask).convert('RGB')
                    input_img_mask = transformer_G(input_img_mask).to(device).unsqueeze(0)
                    input_img_mask = F.interpolate(input_img_mask, size=(input_w, input_h), mode='bilinear')
                    input_img_ = torch.cat([input_img, input_img_mask], dim=1).detach()
                elif with_domains:
                    N,C,H,W = input_img.size()
                    tensor_c=(torch.ones(N,1,H,W)*(-0.6)).cuda()
                    input_img_ = torch.cat([input_img, tensor_c], dim=1).detach()
                ############## Forward Pass ######################
                img_fake = netG(input_img_)

                row1 = input_img[0]*imagenet_std + imagenet_mean
                row2 = img_fake[0]*imagenet_std + imagenet_mean
                imgs.append(row1.to('cpu'))
                imgs.append(row2.to('cpu'))
            imgs = np.stack(imgs, axis = 0).transpose(0,2,3,1)
            test_util.plot_batch(imgs, 5, 2, os.path.join('./', 'result_all.jpg'))