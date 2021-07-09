import torch
import torchvision
import networks
from collections import OrderedDict
from thop import profile
# 增加可读性
from thop import clever_format

def load_model(model, pretrained):
    pretrained_dict = torch.load(pretrained)
    #pretrained_dict = weights['model'].state_dict()  
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
 
model_coder = networks.define_G(3, 3, 32, 'global','Unet', 4, 9, 1, 3, 'instance', gpu_ids={})
load_model(model_coder,"../checkpoints/latest_net_G.pth")
model_coder.eval()
model_coder.cuda()

pix2pix_input = torch.randn(1, 3, 512, 512, device='cuda')

flops, params = profile(model_coder, inputs=(pix2pix_input, ))
flops, params = clever_format([flops, params], "%.3f")

for layer,param in model_coder.state_dict().items(): # param is weight or bias(Tensor) 
    if layer == "from_rgb.weight":
        print(layer,param)

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = [ "input"]
output_names = [ "output" ]
torch.onnx.export(model_coder, pix2pix_input, "pix2pixHD.onnx", verbose=True, input_names=input_names, output_names=output_names)
print("params:",params)
print("flops:",flops)