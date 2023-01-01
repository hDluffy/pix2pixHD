import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import torch.distributed as dist
import torch.multiprocessing as mp

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#os.environ["NCCL_DEBUG"] = "INFO"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12138'

# sync   
seed = 0   
torch.manual_seed(seed)   
torch.cuda.manual_seed(seed)   
torch.cuda.manual_seed_all(seed)   
os.environ['PYTHONHASHSEED'] = str(seed)   
torch.backends.cudnn.deterministic = True   
torch.backends.cudnn.benchmark = False
opt = TrainOptions().parse()

def train(gpu, opt):
    opt.use_ddp = True
    opt.rank = opt.nr * opt.gpus + gpu
    torch.cuda.set_device(gpu)
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0

    opt.print_freq = lcm(opt.print_freq, opt.batchSize)
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.niter = 1
        opt.niter_decay = 0
        opt.max_dataset_size = 10

    #torch.manual_seed(100)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=opt.world_size,
        rank=opt.rank)

    # torch.distributed.init_process_group(backend='nccl', init_method='env://')

    data_loader = CreateDataLoader(opt)
    dataset, train_sampler = data_loader.load_data()
    dataset_size = int(len(data_loader)/opt.gpus)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    if opt.rank == 0:
        visualizer = Visualizer(opt)
    if opt.fp16:
        from apex import amp
        model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D],
                                                           opt_level='O1')
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    else:
        optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

    total_steps = (start_epoch - 1) * dataset_size + epoch_iter

    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        train_sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        for i, data in enumerate(dataset, start=epoch_iter):
            if total_steps % opt.print_freq == print_delta:
                iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == display_delta

            ############## Forward Pass ######################
            losses, generated = model(Variable(data['label']), Variable(data['inst']),
                                      Variable(data['image']), Variable(data['feat']), infer=save_fake)

            # sum per device losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)
            if opt.use_grad_loss:
                loss_G += loss_dict.get('G_Grad', 0)
            if opt.use_ssim_loss:
                loss_G += loss_dict.get('G_ssim', 0)
	        if opt.use_tv_loss:
	            loss_G += loss_dict.get('G_tv',0)

            ############### Backward Pass ####################
            # update generator weights
            optimizer_G.zero_grad()
            if opt.fp16:
                with amp.scale_loss(loss_G, optimizer_G) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_G.backward()
            optimizer_G.step()

            # update discriminator weights
            optimizer_D.zero_grad()
            if opt.fp16:
                with amp.scale_loss(loss_D, optimizer_D) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_D.backward()
            optimizer_D.step()

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / (opt.print_freq * opt.gpus)
                if opt.rank == 0:
                    visualizer.print_current_errors(epoch, epoch_iter * opt.gpus, errors, t)
                    visualizer.plot_current_errors(errors, total_steps)
                    # call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

            ### display output images
            if save_fake and opt.rank == 0:
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                       ('synthesized_image', util.tensor2im(generated.data[0])),
                                       ('real_image', util.tensor2im(data['image'][0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta and opt.rank == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.module.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break

        # end of epoch
        iter_end_time = time.time()
        if opt.rank == 0:
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0 and opt.rank == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.module.save('latest')
            model.module.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.module.update_fixed_params()

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.module.update_learning_rate()


if __name__ == '__main__':
    opt.world_size = opt.nodes * opt.gpus
    mp.spawn(train, nprocs=opt.gpus, args=(opt,), join=True)
