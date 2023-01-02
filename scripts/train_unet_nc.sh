##训练多个域转换，比如pop漫画分男女2种风格，年龄变换多个年龄域
#注：条件的设置需要在pix2pixHD_model_nc.py中修改
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_nc.py --name age_edit --netM 'Unet' \
--batchSize 16 --gpu_ids 0,1,2,3 --loadSize 512 --label_nc 0 --ngf 32 --input_nc 4 \
--dataroot /home/jiaqing/DataSets/pix2pixHD/age_edit  --resize_or_crop none --no_instance --do_nc --domain_num 2
#--continue_train
#--use_ssim_loss 
#--use_grad_loss
#--use_tv_loss
#--use_online_aug
