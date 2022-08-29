#!/bin/sh
export JOB_DIR=models
export IMAGENET_DIR=/mnt/tianyu/workspace/adv_seo_clip/food-101/images
export OUTPUT_DIR=/mnt/tianyu/workspace/adv_seo_clip/models/pretrain_on_food101_v3
export CKPT_PATH=/mnt/tianyu/workspace/adv_seo_clip/mae_pretrain_vit_base_noise.pth

python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 main_pretrain.py \
    --batch_size 64 \
    --output_dir ${OUTPUT_DIR} \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.2 \
    --resume ${CKPT_PATH} \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-5 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR}


# python  main_pretrain.py \
#     --batch_size 96 \
#     --model mae_vit_base_patch16 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 800 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --data_path ${IMAGENET_DIR}