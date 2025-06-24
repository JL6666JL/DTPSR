#!/bin/bash

# ==== 通用配置 ====
IMAGE_PATH="/data1/jianglei/work/dataset/HoliSDiP/StableSR_testsets"
HOLIDATASET="/data2/jianglei/Holidataset"
CHECKPOINT_DIR="/data2/jianglei/HoliSDiP_exp/experiments/ca_hf_lf_fix"

datasets=("DrealSRVal_crop128" "RealSRVal_crop128" "DIV2K_V2_val")
# 推理相关配置
TOTAL_IMAGES=3000
IMAGES_PER_PROCESS=375

# ==== 遍历多个 checkpoint ====
for STEP in $(seq 100000 20000 180000); do
    echo "======== 开始处理 checkpoint-$STEP ========"
    OUTPUT_DIR="/data2/jianglei/HoliSDiP/results/ca_hf_lf_fix_CFG_hflfneg_$STEP"
    for item in "${datasets[@]}"; do
        NOW_IMAGE_PATH="$IMAGE_PATH/$item/lq"
        NOW_HFLF_DES_PATH="$HOLIDATASET/$item/lfhf_local_descriptions"
        NOW_DES_PATH="$HOLIDATASET/$item/descriptions/descriptions.json"

        HOLISDIP_MODEL="$CHECKPOINT_DIR/checkpoint-$STEP"
        NOW_OUTPUT_DIR="$OUTPUT_DIR/$item"

        if [[ "$item" == "DIV2K_V2_val" ]]; then
            # 启动8个并行进程
            for ((i=0; i<8; i++))
            do
                START=$((i * IMAGES_PER_PROCESS))
                END=$(( (i + 1) * IMAGES_PER_PROCESS))
                if [ $i -eq 7 ]; then
                    END=$((TOTAL_IMAGES))
                fi

                # 分配 GPU
                if [ $i -lt 2 ]; then
                    GPU="0"
                elif [ $i -lt 4 ]; then
                    GPU="1"
                elif [ $i -lt 6 ]; then
                    GPU="2"
                else
                    GPU="3"
                fi

                # echo "启动进程 $((i+1)): checkpoint-$STEP, 图片 $START-$END (GPU $GPU)"
                CUDA_VISIBLE_DEVICES=$GPU python test.py \
                    --holisdip_model_path "$HOLISDIP_MODEL" \
                    --image_path "$NOW_IMAGE_PATH" \
                    --output_dir "$NOW_OUTPUT_DIR" \
                    --save_prompts \
                    --local_des_pkl_path "$NOW_HFLF_DES_PATH" \
                    --des_path "$NOW_DES_PATH" \
                    --start_index "$START" \
                    --end_index "$END" &
            done
        elif [[ "$item" == "DrealSRVal_crop128" ]]; then
            GPU="1"
            CUDA_VISIBLE_DEVICES=$GPU python test.py \
                --holisdip_model_path "$HOLISDIP_MODEL" \
                --image_path "$NOW_IMAGE_PATH" \
                --output_dir "$NOW_OUTPUT_DIR" \
                --save_prompts \
                --local_des_pkl_path "$NOW_HFLF_DES_PATH" \
                --des_path "$NOW_DES_PATH" &
        else
            GPU="2"
            CUDA_VISIBLE_DEVICES=$GPU python test.py \
                --holisdip_model_path "$HOLISDIP_MODEL" \
                --image_path "$NOW_IMAGE_PATH" \
                --output_dir "$NOW_OUTPUT_DIR" \
                --save_prompts \
                --local_des_pkl_path "$NOW_HFLF_DES_PATH" \
                --des_path "$NOW_DES_PATH" &
        fi
    done
    # 等待所有进程完成再评估
    wait
    echo "✅ checkpoint-$STEP 推理完成，开始评估..."

    CUDA_VISIBLE_DEVICES=2 python evaluate.py \
        --sr_dir="$OUTPUT_DIR" \
        --steps="$STEP"

    echo "🧹 清理输出目录：$OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"

    echo "======== checkpoint-$STEP 处理完成 ========"
    echo
done
