import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.functional as F
###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, netM, n_downsample_global=3, n_blocks_global=4, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        if netM == 'Unet':
            netG = GlobalGeneratorUnet(input_nc, output_nc, ngf, n_blocks_global, norm_type="none")
        elif netM == 'SG':
            netG = GlobalGeneratorSG(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_type="none")
        else:
            netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)       
    elif netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = torch.abs(error) < clip_delta
    squared_loss = 0.5 * torch.pow(error,2.0)
    linear_loss = clip_delta * (torch.abs(error) - 0.5 * clip_delta)
    return torch.mean(torch.where(cond, squared_loss, linear_loss))

def edge_conv2d(im):
    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    # 定义sobel算子参数，所有值除以3个人觉得出来的图更好些
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 卷积输出通道，这里我设置为3
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
    # 输入图的通道，这里我设置为3
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)

    conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
    # print(conv_op.weight.size())
    # print(conv_op, '\n')

    edge_detect = conv_op(im)
    #print(torch.max(edge_detect))
    # 将输出转换为图片格式
    #edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class GradientLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(GradientLoss, self).__init__()
        self.criterion = nn.L1Loss()
        #self.grad = edge_conv2d()
        #self.criterion = huber_loss()

    def forward(self, x, y):
        x_grad,y_grad = edge_conv2d(x).cuda(),edge_conv2d(y).cuda()
        loss = self.criterion(x_grad, y_grad.detach())
        #loss = huber_loss(x_grad, y_grad.detach())
        return loss

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev
        
class ConvBlock(nn.Module):
    """Conv Block with instance normalization."""
    def __init__(self, dim_in, dim_out, kernel_size_, stride_, padding_,norm_type="none"):
        super(ConvBlock, self).__init__()
        activation=nn.ReLU(inplace=True)
        if norm_type == "batch":
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size_, stride=stride_, padding=padding_, bias=False),
                nn.BatchNorm2d(dim_out),
                activation)
        elif norm_type == "instance":
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size_, stride=stride_, padding=padding_, bias=False),
                nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
                activation)
        else :
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size_, stride=stride_, padding=padding_, bias=False),
                activation)

    def forward(self, x):
        return self.main(x)
        
class UpConvBlock(nn.Module):
    """Conv Block with instance normalization."""
    def __init__(self, dim_in, dim_out, kernel_size_, stride_, padding_,up_type,norm_type="none"):
        super(UpConvBlock, self).__init__()
        
        activation=nn.ReLU(inplace=True)
        self.up_type=up_type
        if up_type=="interpolate":
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 3, 1, 1),
                nn.ReLU(inplace=True))
        else:
            if norm_type == "batch":
                self.main = nn.Sequential(
                    nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size_, stride=stride_, padding=padding_,output_padding=1,bias=False),
                    nn.BatchNorm2d(dim_out),
                    activation)
            elif norm_type == "instance":
                self.main = nn.Sequential(
                    nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size_, stride=stride_, padding=padding_,output_padding=1,bias=False),
                    nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
                    activation)
            else:
                self.main = nn.Sequential(
                    nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size_, stride=stride_, padding=padding_,output_padding=1,bias=False),
                    activation)

    def forward(self, x):
        if self.up_type=="interpolate":
            x=F.interpolate(x,scale_factor=2, mode='nearest')
        return self.main(x)

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, norm_type="none"):
        super(ResidualBlock, self).__init__()
        
        activation=nn.ReLU(inplace=True)
        if norm_type=="batch":
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(dim_out),
                activation,
                nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(dim_out))
        elif norm_type=="instance":
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
                activation,
                nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
        else:
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
                activation,
                nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return x + self.main(x)

class ResidualBottleneckBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock2, self).__init__()
        hidden_dim=dim_in//2
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim_out))

    def forward(self, x):
        return x + self.main(x)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                #nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                #nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                #nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                #nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                #nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
            
class ResidualBlockDW(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock1, self).__init__()
        self.main = nn.Sequential(
            InvertedResidual(dim_in, dim_out,1,1),
            nn.ReLU(inplace=True),
            InvertedResidual(dim_out, dim_out,1,1))

    def forward(self, x):
        return x + self.main(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.PReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
        
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
        
class DulAttention(nn.Module):
    def __init__(self,inc):
        super(DulAttention, self).__init__()

        #self.inplanes = 64
        self.conv = nn.Sequential(
            nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.ca = ChannelAttention(inc)
        self.sa = SpatialAttention()
        self.conv1x1 = nn.Conv2d(inc*2, inc, 1, 1, 0, bias=False)

    def forward(self, x):
        conv_out = self.conv(x)
        x1=self.ca(conv_out)*conv_out
        x2=self.sa(conv_out)*conv_out
        att_out = self.conv1x1(torch.cat([x1, x2], dim=1))
        return att_out+x
        
class GlobalGeneratorSG(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=4, norm_type="none"):
        assert (n_blocks >= 0)
        super(GlobalGeneratorUnet01, self).__init__()
        #activation = nn.LeakyReLU(0.2)
        activation = nn.ReLU(True)
        self.att=False

        dim_in = ngf
        max_conv_dim = 512
        self.from_rgb = ConvBlock(input_nc, dim_in, kernel_size_=7, stride_=1, padding_=3, norm_type=norm_type)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.res_block = nn.ModuleList()
        self.to_rgb = nn.Conv2d(dim_in, output_nc, kernel_size=7, stride=1, padding=3, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # down/up-sampling blocks
        repeat_num = n_downsampling
        for num in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ConvBlock(dim_in, dim_out, kernel_size_=4, stride_=2, padding_=1,norm_type=norm_type))
            if num==repeat_num-1:
                self.decode.insert(
                    0, UpConvBlock(dim_out, dim_in, kernel_size_=4, stride_=2, padding_=1,up_type="interpolate",norm_type=norm_type))  # stack-like
            else:
                self.decode.insert(
                    0, UpConvBlock(dim_out*2, dim_in, kernel_size_=4, stride_=2, padding_=1,up_type="interpolate",norm_type=norm_type))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(n_blocks):
            self.res_block.append(
                ResidualBlock(dim_in=dim_out, dim_out=dim_out,norm_type=norm_type))

    def forward(self, x):
        out = self.from_rgb(x)
        cache = {}
        ind=0
        for block in self.encode:
            ind=ind+1
            out = block(out)
            cache[ind] = out
        for block in self.res_block:
            out = block(out)
        for block in self.decode:
            out = block(out)
            ind = ind - 1
            if ind>0:
                #out = out + cache[ind]
                out=torch.cat((cache[ind], out), dim=1)
        out=self.to_rgb(out)
        out=self.tanh(out)
        if self.att:
            att= self.sigmoid(x)
            out = (out * att + x * (1 - att))
        return out
        
class GlobalGeneratorUnet00(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_type="none"):
        conv_dim = ngf
        repeat_num = 4
        assert (n_blocks >= 0)
        super(GlobalGeneratorUnet00, self).__init__()
        self.left_conv_start = ConvBlock(input_nc, conv_dim, kernel_size_=7, stride_=1, padding_=3,norm_type=norm_type)

        curr_dim = conv_dim
        self.left_conv_1 = ConvBlock(curr_dim, curr_dim * 2, kernel_size_=4, stride_=2, padding_=1,norm_type=norm_type)
        curr_dim = curr_dim * 2
        self.left_conv_2 = ConvBlock(curr_dim, curr_dim * 2, kernel_size_=4, stride_=2, padding_=1,norm_type=norm_type)
        curr_dim = curr_dim * 2
        self.left_conv_3 = ConvBlock(curr_dim, curr_dim * 2, kernel_size_=4, stride_=2, padding_=1,norm_type=norm_type)
        curr_dim = curr_dim * 2
        self.left_conv_4 = ConvBlock(curr_dim, curr_dim * 2, kernel_size_=4, stride_=2, padding_=1,norm_type=norm_type)
        curr_dim = curr_dim * 2
        #self.left_conv_5 = ConvBlock(curr_dim, curr_dim, kernel_size_=4, stride_=2, padding_=1,norm_type=norm_type)
        #self.left_conv_6 = ConvBlock(curr_dim, curr_dim, kernel_size_=4, stride_=2, padding_=1,norm_type=norm_type)

        layers = []
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim,norm_type=norm_type))
        self.res_block = nn.Sequential(*layers)
        # self.left_conv_5 = ConvBlock(in_channels=512, middle_channels=1024, out_channels=1024)

        # 定义右半部分网络
        self.right_conv_1 = UpConvBlock(curr_dim, curr_dim // 2, kernel_size_=4, stride_=2, padding_=1,up_type="interpolate",norm_type=norm_type)
        curr_dim = curr_dim // 2
        self.right_conv_2 = UpConvBlock(curr_dim * 2, curr_dim // 2, kernel_size_=4, stride_=2, padding_=1,up_type="interpolate",norm_type=norm_type)
        curr_dim = curr_dim // 2
        self.right_conv_3 = UpConvBlock(curr_dim * 2, curr_dim // 2, kernel_size_=4, stride_=2, padding_=1,up_type="interpolate",norm_type=norm_type)
        curr_dim = curr_dim // 2
        self.right_conv_4 = UpConvBlock(curr_dim * 2, curr_dim // 2, kernel_size_=4, stride_=2, padding_=1,up_type="interpolate",norm_type=norm_type)
        curr_dim = curr_dim // 2
        #self.right_conv_5 = UpConvBlock(curr_dim * 2, curr_dim // 2, kernel_size_=4, stride_=2, padding_=1,up_type="interpolate",norm_type=norm_type)
        #curr_dim = curr_dim // 2
        #self.right_conv_6 = UpConvBlock(curr_dim * 2, curr_dim // 2, kernel_size_=4, stride_=2, padding_=1,up_type="interpolate",norm_type=norm_type)
        #curr_dim = curr_dim // 2

        self.right_conv_end = nn.Conv2d(curr_dim, output_nc, kernel_size=7, stride=1, padding=3, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.

        # 1：进行编码过程
        feature_0 = self.left_conv_start(x)
        feature_1 = self.left_conv_1(feature_0)
        feature_2 = self.left_conv_2(feature_1)
        feature_3 = self.left_conv_3(feature_2)
        feature_4 = self.left_conv_4(feature_3)
        #feature_5 = self.left_conv_5(feature_4)
        #feature_6 = self.left_conv_6(feature_5)

        feature_res = self.res_block(feature_4)
        # 2：进行解码过程
        de_feature_1 = self.right_conv_1(feature_res)
        temp = torch.cat((feature_3, de_feature_1), dim=1)
        de_feature_2 = self.right_conv_2(temp)
        temp = torch.cat((feature_2, de_feature_2), dim=1)
        de_feature_3 = self.right_conv_3(temp)
        temp = torch.cat((feature_1, de_feature_3), dim=1)
        de_feature_4 = self.right_conv_4(temp)
        #temp = torch.cat((feature_1, de_feature_4), dim=1)
        #de_feature_5 = self.right_conv_5(temp)
        #temp = torch.cat((feature_1, de_feature_5), dim=1)
        #de_feature_6 = self.right_conv_6(temp)

        out_ = self.right_conv_end(de_feature_4)
        out = self.tanh(out_)
        return out 


        
class GlobalGeneratorUnet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_type="none"):
        conv_dim = ngf
        repeat_num = 4
        assert (n_blocks >= 0)
        super(GlobalGeneratorUnet, self).__init__()
        
        activation = nn.ReLU(True)
        self.att=False
        self.left_conv_start = nn.Sequential(
            nn.Conv2d(input_nc, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True))

        curr_dim = 16
        #self.left_conv_1 = ConvBlock(curr_dim, curr_dim * 2, kernel_size_=4, stride_=2, padding_=1, norm_type=norm_type)
        self.left_dwconv_1 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        #self.att1=DulAttention(curr_dim)
        self.left_conv_1 = ConvBlock(curr_dim, curr_dim * 2, kernel_size_=3, stride_=2, padding_=1, norm_type=norm_type)
        curr_dim = curr_dim * 2
        self.left_dwconv_2 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.att2=DulAttention(curr_dim)
        self.left_conv_2 = ConvBlock(curr_dim, curr_dim * 2, kernel_size_=3, stride_=2, padding_=1, norm_type=norm_type)
        curr_dim = curr_dim * 2
        self.left_dwconv_3 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.att3=DulAttention(curr_dim)
        self.left_conv_3 = ConvBlock(curr_dim, curr_dim * 2, kernel_size_=3, stride_=2, padding_=1, norm_type=norm_type)
        curr_dim = curr_dim * 2
        self.left_dwconv_4_0 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.left_dwconv_4_1 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.att4=DulAttention(curr_dim)
        self.left_conv_4 = ConvBlock(curr_dim, curr_dim, kernel_size_=3, stride_=2, padding_=1, norm_type=norm_type)
        curr_dim = curr_dim

        layers = []
        for i in range(repeat_num):
            layers.append(ResidualBlock(curr_dim, curr_dim, norm_type=norm_type))
        self.res_block = nn.Sequential(*layers)

        # 定义右半部分网络
        self.right_conv_1 = UpConvBlock(curr_dim, curr_dim, kernel_size_=3, stride_=2, padding_=1,up_type="interpolate", norm_type=norm_type)
        curr_dim = curr_dim
        self.right_dwconv_1_0 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_1_1 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_1_2 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_conv_2 = UpConvBlock(curr_dim, curr_dim // 2, kernel_size_=3, stride_=2, padding_=1,up_type="interpolate", norm_type=norm_type)
        curr_dim = curr_dim // 2
        self.right_dwconv_2_0 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_2_1 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_2_2 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_conv_3 = UpConvBlock(curr_dim, curr_dim // 2, kernel_size_=3, stride_=2, padding_=1,up_type="interpolate", norm_type=norm_type)
        curr_dim = curr_dim // 2
        self.right_dwconv_3_0 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_3_1 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_3_2 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_conv_4 = UpConvBlock(curr_dim, 16, kernel_size_=3, stride_=2, padding_=1,up_type="interpolate", norm_type=norm_type)
        curr_dim = 16
        self.right_dwconv_4_0 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_4_1 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)
        self.right_dwconv_4_2 = ResidualBlock(curr_dim, curr_dim, norm_type=norm_type)

        self.right_conv_end = nn.Conv2d(curr_dim, output_nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.

        # 1：进行编码过程
        feature_0 = self.left_conv_start(x)
        dw_feature_1 = self.left_dwconv_1(feature_0)
        feature_1 = self.left_conv_1(dw_feature_1)
        dw_feature_2 = self.left_dwconv_2(feature_1)
        feature_2 = self.left_conv_2(dw_feature_2)
        dw_feature_3 = self.left_dwconv_3(feature_2)
        feature_3 = self.left_conv_3(dw_feature_3)
        dw_feature_4 = self.left_dwconv_4_0(feature_3)
        dw_feature_4 = self.left_dwconv_4_1(dw_feature_4)
        feature_4 = self.left_conv_4(dw_feature_4)
        #feature_5 = self.left_conv_5(feature_4)
        #feature_6 = self.left_conv_6(feature_5)

        feature_res = self.res_block(feature_4)
        # 2：进行解码过程
        de_feature_1 = self.right_conv_1(feature_res)
        #temp = torch.cat((self.att4(dw_feature_4), de_feature_1), dim=1)
        temp = self.att4(dw_feature_4) + de_feature_1
        temp = self.right_dwconv_1_0(temp)
        temp = self.right_dwconv_1_1(temp)
        temp = self.right_dwconv_1_2(temp)
        de_feature_2 = self.right_conv_2(temp)
        #temp = torch.cat((self.att3(dw_feature_3), de_feature_2), dim=1)
        temp = self.att3(dw_feature_3) + de_feature_2
        temp = self.right_dwconv_2_0(temp)
        temp = self.right_dwconv_2_1(temp)
        temp = self.right_dwconv_2_2(temp)
        de_feature_3 = self.right_conv_3(temp)
        #temp = torch.cat((self.att2(dw_feature_2), de_feature_3), dim=1)
        temp = self.att2(dw_feature_2) + de_feature_3
        temp = self.right_dwconv_3_0(temp)
        temp = self.right_dwconv_3_1(temp)
        temp = self.right_dwconv_3_2(temp)
        de_feature_4 = self.right_conv_4(temp)
        #temp = torch.cat((dw_feature_1, de_feature_4), dim=1)
        temp = dw_feature_1 + de_feature_4
        temp = self.right_dwconv_4_0(temp)
        temp = self.right_dwconv_4_1(temp)
        temp = self.right_dwconv_4_2(temp)

        out_ = self.right_conv_end(temp)
        out = self.tanh(out_)
        if self.att:
            att = self.sigmoid(x)
            out = (out * att + x * (1 - att))
        return out

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)             
        
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]             
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))        
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4            
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]                    
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)                                        
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat                       
        return outputs_mean

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
