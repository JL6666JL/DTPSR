CUDA_VISIBLE_DEVICES=0 python test.py \
--model_path /data2/jianglei/DTPSR_exp/experiments/ca_hf_lf_fix/checkpoint-110000  \
--image_path /data1/jianglei/work/dataset/DTPSR/StableSR_testsets/DrealSRVal_crop128/lq \
--output_dir /data2/jianglei/DTPSR/test  \
--save_prompts \
--local_des_pkl_path /data2/jianglei/DTPSRdataset/DrealSRVal_crop128/lfhf_local_descriptions \
--des_path /data2/jianglei/DTPSRdataset/DrealSRVal_crop128/descriptions/descriptions.json  



