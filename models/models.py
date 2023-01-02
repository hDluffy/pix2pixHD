import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def create_model(opt):
    if opt.model == 'pix2pixHD':
        if opt.do_nc:
            from .pix2pixHD_model_nc import Pix2PixHDModel, InferenceModel
        else :
            from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    else:
    	from .ui_model import UIModel
    	model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        if opt.use_ddp:
            torch.cuda.set_device(opt.rank)
            device = torch.device('cuda', opt.rank)
            model.to(device)
            model = DDP(model, device_ids=[opt.rank],output_device=opt.rank)
        else:
            model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
