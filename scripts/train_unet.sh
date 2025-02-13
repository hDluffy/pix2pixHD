task_num='no_mask'
if [ ${task_num} = 'no_mask' ];then
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name 2Goddess --netM 'Unet' \
    --batchSize 16 --gpu_ids 0,1,2,3 --loadSize 512 --label_nc 0 --ngf 32 \
    --dataroot /home/jiaqing/DataSets/pix2pixHD/2Goddess  --resize_or_crop none --no_instance
fi

if [ ${task_num} = 'with_mask' ];then
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name Hairpaint --netM 'Unet' \
    --batchSize 16 --gpu_ids 0,1,2,3 --loadSize 512 --label_nc 0 --ngf 32 \
    --dataroot /home/jiaqing/DataSets/pix2pixHD/Hairpaint  --resize_or_crop none --no_instance \
    --use_mask
fi


#--continue_train
#--use_ssim_loss 
#--use_grad_loss
#--use_tv_loss
#--use_online_aug
#--use_mask
