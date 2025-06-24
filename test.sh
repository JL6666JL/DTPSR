CUDA_VISIBLE_DEVICES=0 python test.py \
--holisdip_model_path /data2/jianglei/HoliSDiP_exp/experiments/ca_hf_lf_fix/checkpoint-160000  \
--image_path /data1/jianglei/work/dataset/HoliSDiP/StableSR_testsets/DrealSRVal_crop128/lq \
--output_dir /data2/jianglei/HoliSDiP/test  \
--save_prompts \
--local_des_pkl_path /data2/jianglei/Holidataset/DrealSRVal_crop128/lfhf_local_descriptions \
--des_path /data2/jianglei/Holidataset/DrealSRVal_crop128/descriptions/descriptions.json  

#!/bin/bash

# # ==== 通用配置 ====
# IMAGE_PATH="/data1/jianglei/work/dataset/HoliSDiP/StableSR_testsets"
# HOLIDATASET="/data2/jianglei/Holidataset"
# CHECKPOINT_DIR="/data2/jianglei/HoliSDiP/experiments/ca_hf_lf"

# datasets=("DrealSRVal_crop128" "RealSRVal_crop128" "DIV2K_V2_val")
# # 推理相关配置
# TOTAL_IMAGES=3000
# IMAGES_PER_PROCESS=375

# # ==== 遍历多个 checkpoint ====
# for STEP in $(seq 160000 10000 160000); do
#     echo "======== 开始处理 checkpoint-$STEP ========"
#     OUTPUT_DIR="/data2/jianglei/HoliSDiP/results/ca_hf_lf_$STEP"
#     for item in "${datasets[@]}"; do
#         NOW_IMAGE_PATH="$IMAGE_PATH/$item/lq"
#         NOW_HFLF_DES_PATH="$HOLIDATASET/$item/lfhf_local_descriptions"
#         NOW_DES_PATH="$HOLIDATASET/$item/descriptions/descriptions.json"

#         HOLISDIP_MODEL="$CHECKPOINT_DIR/checkpoint-$STEP"
#         NOW_OUTPUT_DIR="$OUTPUT_DIR/$item"
#         # 启动8个并行进程
#         for ((i=0; i<8; i++))
#         do
#             START=$((i * IMAGES_PER_PROCESS))
#             END=$(( (i + 1) * IMAGES_PER_PROCESS))
#             if [ $i -eq 7 ]; then
#                 END=$((TOTAL_IMAGES))
#             fi

#             # 分配 GPU
#             if [ $i -lt 2 ]; then
#                 GPU="0"
#             elif [ $i -lt 4 ]; then
#                 GPU="1"
#             elif [ $i -lt 6 ]; then
#                 GPU="2"
#             else
#                 GPU="3"
#             fi

#             # echo "启动进程 $((i+1)): checkpoint-$STEP, 图片 $START-$END (GPU $GPU)"
#             CUDA_VISIBLE_DEVICES=$GPU python test.py \
#                 --holisdip_model_path "$HOLISDIP_MODEL" \
#                 --image_path "$NOW_IMAGE_PATH" \
#                 --output_dir "$NOW_OUTPUT_DIR" \
#                 --save_prompts \
#                 --local_des_pkl_path "$NOW_HFLF_DES_PATH" \
#                 --des_path "$NOW_DES_PATH" \
#                 --start_index "$START" \
#                 --end_index "$END" &
#         done
#     done
#     echo "======== checkpoint-$STEP 处理完成 ========"
#     echo
# done

