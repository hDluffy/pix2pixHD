CUDA_VISIBLE_DEVICES=0,1 python train_ddp.py --name 2Old --netM 'Unet' \
--batchSize 16 --gpu_ids 0,1 --loadSize 512 --label_nc 0 --ngf 32 \
--dataroot /data2/2Old/  --resize_or_crop none --no_instance 
#--continue_train
#--use_ssim_loss 
#--use_grad_loss
#--use_tv_loss
#--use_online_aug