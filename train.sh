CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch train.py  \
--output_dir /data2/jianglei/HoliSDiP/experiments/ca_hf_lf_fix  \
--enable_xformers_memory_efficient_attention \
--train_batch_size=4 \
--gradient_accumulation_steps=2  