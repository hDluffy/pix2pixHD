一、添加新loss需要修改的脚本包括：
1.train_options(控制是否激活该loss)
2.pix2pixHD_model(设置filter,统计loss)
3.networks(新loss实现)

本版本添加了grad_loss、ssim_loss及huber_loss：
ssim_loss是在pytorch_ssim中实现，huber_loss对VGG_loss优化

二、添加网络结构：
在networks中新增结构，通过defin_G函数选择对应的结构

三、多域输入，加条件：



运行脚本：
python train.py --name 2Old --netM 'Unet' --batchSize 32 --gpu_ids 0,1 --label_nc 0 --loadSize 512 --ngf 32 --dataroot /data2/2Old/  --resize_or_crop none --no_instance 
#--continue_train
#--use_ssim_loss 
#--continue_train
#--use_grad_loss

选项：
--netM包括自定义的G网络结构选择(目前常用Unet)
--n_downsample_global定义所选的G网络下采用次数(目前只适用Unet)
--n_blocks_global定义所选的G网络block次数(目前只适用Unet)
--norm定义了归一化层，"none"表示不加归一化层，"batch"表示bn，"instance"表示in